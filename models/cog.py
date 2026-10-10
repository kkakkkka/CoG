import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, feature_dim, ratio=16):
        super().__init__()
        reduced_dim = max(feature_dim // ratio, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.conv1 = nn.Conv1d(feature_dim, reduced_dim, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(reduced_dim, feature_dim, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_t = x.transpose(1, 2)
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x_t))))
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x_t))))
        return self.sigmoid(avg_out + max_out).transpose(1, 2)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class BiasSeparatedDilatedConv1d(nn.Conv1d):
    def forward(self, input):
        if self.dilation[0] <= 1 or self.bias is None:
            return super().forward(input)
        out = F.conv1d(input, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
        return out + self.bias.view(1, -1, 1)


class NativeBiasDilatedConv1dFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, stride, padding, dilation, groups):
        out = F.conv1d(input, weight, None, stride, padding, dilation, groups)
        ctx.save_for_backward(input, weight)
        ctx.bias_defined = bias is not None
        ctx.params = (stride, padding, dilation, groups, weight.shape[0])
        return out if bias is None else out + bias.view(1, -1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        stride, padding, dilation, groups, out_channels = ctx.params
        mask = (ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2] and ctx.bias_defined)
        grad_input, grad_weight, grad_bias = torch.ops.aten.convolution_backward.default(
            grad_output, input, weight, [out_channels] if ctx.bias_defined else None,
            list(stride), list(padding), list(dilation), False, [0], groups, mask
        )
        return grad_input, grad_weight, grad_bias, None, None, None, None


class BiasNativeDilatedConv1d(nn.Conv1d):
    def forward(self, input):
        if self.dilation[0] <= 1 or self.bias is None:
            return super().forward(input)
        return NativeBiasDilatedConv1dFn.apply(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ExpandedDilatedConv1d(nn.Conv1d):
    def forward(self, input):
        if self.dilation[0] <= 1:
            return super().forward(input)
        kernel = self.weight.shape[-1]
        effective = (kernel - 1) * self.dilation[0] + 1
        weight = self.weight.new_zeros((*self.weight.shape[:-1], effective))
        weight[..., :: self.dilation[0]] = self.weight
        return F.conv1d(input, weight, self.bias, self.stride, self.padding, 1, self.groups)


def make_text_conv1d(in_channels, out_channels, kernel_size, padding, dilation, mode="baseline"):
    if dilation <= 1 or mode == "baseline":
        return nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
    if mode == "bias_separate":
        return BiasSeparatedDilatedConv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
    if mode == "bias_native":
        return BiasNativeDilatedConv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
    if mode == "expanded":
        return ExpandedDilatedConv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
    raise ValueError(f"unknown text TCN conv mode: {mode}")


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2, conv_mode="baseline"):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            make_text_conv1d(n_inputs, n_outputs, kernel_size, padding, dilation, conv_mode),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            make_text_conv1d(n_outputs, n_outputs, kernel_size, padding, dilation, conv_mode),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        return F.relu(self.net(x) + residual)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, conv_mode="baseline"):
        super().__init__()
        layers = []
        for i, out_channels in enumerate(num_channels):
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, 2**i, dropout, conv_mode))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TextEncoderTCN(nn.Module):
    def __init__(self, args, n_words, embed_size=300, pre_trained_embedding=None, kernel_size=2, dropout=0.3):
        super().__init__()
        if pre_trained_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(
                torch.FloatTensor(pre_trained_embedding), freeze=args.freeze_wordembed
            )
        else:
            self.embedding = nn.Embedding(n_words, embed_size)
        num_channels = [args.hidden_size] * args.n_layer
        conv_mode = getattr(args, "text_tcn_conv_mode", "baseline")
        self.tcn = TemporalConvNet(embed_size, num_channels, kernel_size, dropout=dropout, conv_mode=conv_mode)
        self.decoder = nn.Linear(num_channels[-1], args.word_f)
        self.drop = nn.Dropout(0.1)

    def forward(self, input):
        emb = self.drop(self.embedding(input))
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        return self.decoder(y).contiguous(), 0


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, ker_size, stride=1, first_dilation=0, downsample=False):
        super().__init__()
        padding = first_dilation
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=ker_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(planes)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=ker_size, padding=ker_size // 2)
        self.bn2 = nn.BatchNorm1d(planes)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.downsample = None
        if downsample or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes, kernel_size=ker_size, stride=stride, padding=padding),
                nn.BatchNorm1d(planes),
            )

    def forward(self, x):
        shortcut = x
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        return self.act2(x + shortcut)


class WavEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.feat_extractor = nn.Sequential(
            BasicBlock(1, 32, 15, 5, first_dilation=1600, downsample=True),
            BasicBlock(32, 32, 15, 6, first_dilation=0, downsample=True),
            BasicBlock(32, 32, 15, 1, first_dilation=7),
            BasicBlock(32, 64, 15, 6, first_dilation=0, downsample=True),
            BasicBlock(64, 64, 15, 1, first_dilation=7),
            BasicBlock(64, out_dim, 15, 6, first_dilation=0, downsample=True),
        )

    def forward(self, wav_data):
        return self.feat_extractor(wav_data.unsqueeze(1)).transpose(1, 2)


class GestureAdaLN(nn.Module):
    """Adaptive layer norm driven by frame-wise style features."""

    def __init__(self, feature_dim, style_dim):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.modulation = nn.Sequential(
            nn.Linear(style_dim, feature_dim * 2),
            nn.SiLU(),
            nn.Linear(feature_dim * 2, feature_dim * 2),
        )

    def forward(self, x, style):
        scale, shift = self.modulation(style).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale) + shift


class FaceGenerator(nn.Module):
    """Speech-to-face branch used as the first stage in CoG."""

    def __init__(self, audio_dim, facial_dim, facial_feature_dim):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(audio_dim, facial_feature_dim),
            nn.LeakyReLU(True),
            nn.Linear(facial_feature_dim, facial_feature_dim),
            nn.LeakyReLU(True),
        )
        self.face_out = nn.Linear(facial_feature_dim, facial_dim)

    def forward(self, audio_feat):
        face_feat = self.feature_net(audio_feat)
        face_pred = self.face_out(face_feat)
        return face_pred, face_feat


class StyleEncoder(nn.Module):
    """Emotion/speaker classifier plus frame-wise style hypernetwork."""

    def __init__(self, args, audio_dim, style_dim):
        super().__init__()
        self.emotion_dims = args.emotion_dims
        self.speaker_dims = args.speaker_dims
        self.emotion_f = args.emotion_f
        self.speaker_f = args.speaker_f

        self.temporal = nn.Sequential(
            nn.Conv1d(audio_dim, audio_dim, kernel_size=9, padding=4),
            nn.BatchNorm1d(audio_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(audio_dim, audio_dim, kernel_size=9, padding=4),
            nn.BatchNorm1d(audio_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.emotion_cls = nn.Linear(audio_dim, self.emotion_dims)
        self.speaker_cls = nn.Linear(audio_dim, self.speaker_dims)
        self.emotion_style = nn.Embedding(self.emotion_dims, self.emotion_f)
        self.speaker_style = nn.Embedding(self.speaker_dims, self.speaker_f)
        self.hyper = nn.Sequential(
            nn.Linear(self.emotion_f + self.speaker_f, style_dim),
            nn.SiLU(),
            nn.Linear(style_dim, style_dim),
        )

    def forward(self, audio_feat):
        h = self.temporal(audio_feat.transpose(1, 2)).transpose(1, 2)
        pooled = h.mean(dim=1)
        emotion_logits = self.emotion_cls(pooled)
        speaker_logits = self.speaker_cls(pooled)
        if not any(parameter.requires_grad for parameter in self.temporal.parameters()):
            emotion_logits = emotion_logits.detach()
            speaker_logits = speaker_logits.detach()

        emotion_prob = F.softmax(emotion_logits, dim=-1)
        speaker_prob = F.softmax(speaker_logits, dim=-1)
        emotion_style = emotion_prob @ self.emotion_style.weight
        speaker_style = speaker_prob @ self.speaker_style.weight
        style = torch.cat([emotion_style, speaker_style], dim=-1)
        style = self.hyper(style).unsqueeze(1).expand(-1, audio_feat.shape[1], -1)
        return style, emotion_logits, speaker_logits

    def style_from_probs(self, emotion_prob, speaker_prob, frames):
        emotion_style = emotion_prob @ self.emotion_style.weight
        speaker_style = speaker_prob @ self.speaker_style.weight
        style = torch.cat([emotion_style, speaker_style], dim=-1)
        style = self.hyper(style).unsqueeze(1).expand(-1, frames, -1)
        return style, emotion_prob, speaker_prob

    def classifier_modules(self):
        return (self.temporal, self.emotion_cls, self.speaker_cls)

    def set_classifier_trainable(self, trainable):
        for module in self.classifier_modules():
            module.train(trainable)
            for param in module.parameters():
                param.requires_grad = trainable


class CoG(nn.Module):
    """Co-speech gesture generator.

    Audio is encoded into face, body, and hand motion. Emotion and speaker
    probabilities set the style of the body and hand decoders.
    """

    def __init__(self, args):
        super().__init__()
        self.pose_dims = args.pose_dims
        self.facial_dims = args.facial_dims
        self.audio_f = args.audio_f
        self.word_f = args.word_f
        self.facial_f = args.facial_f
        self.hidden_size = args.hidden_size
        self.n_layer = args.n_layer
        self.style_dim = getattr(args, "style_f", self.hidden_size)

        self.audio_encoder = WavEncoder(self.audio_f)

        self.text_encoder = None
        if self.word_f != 0:
            with open(f"{args.root_path}{args.train_data_path[:-6]}vocab.pkl", "rb") as f:
                lang_model = pickle.load(f)
                pre_trained_embedding = lang_model.word_embedding_weights
            self.text_encoder = TextEncoderTCN(
                args,
                args.word_index_num,
                args.word_dims,
                pre_trained_embedding=pre_trained_embedding,
                dropout=args.dropout_prob,
            )

        self.face_generator = FaceGenerator(self.audio_f, self.facial_dims, self.facial_f)
        self.style_encoder = StyleEncoder(args, self.audio_f, self.style_dim)

        body_context_dim = self.pose_dims + 1 + self.audio_f + self.word_f + self.facial_f
        self.body_channel_attn = ChannelAttention(body_context_dim)
        self.body_adaln = GestureAdaLN(body_context_dim, self.style_dim)
        self.body_lstm = nn.LSTM(
            body_context_dim,
            hidden_size=self.hidden_size,
            num_layers=self.n_layer,
            batch_first=True,
            bidirectional=True,
            dropout=args.dropout_prob,
        )
        self.body_out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LeakyReLU(True),
            nn.Linear(self.hidden_size // 2, 27),
        )

        hand_context_dim = body_context_dim + self.hidden_size + 27
        self.hand_channel_attn = ChannelAttention(hand_context_dim)
        self.hand_adaln = GestureAdaLN(hand_context_dim, self.style_dim)
        self.hand_lstm = nn.LSTM(
            hand_context_dim,
            hidden_size=self.hidden_size,
            num_layers=self.n_layer,
            batch_first=True,
            bidirectional=True,
            dropout=args.dropout_prob,
        )
        self.hand_out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LeakyReLU(True),
            nn.Linear(self.hidden_size // 2, self.pose_dims - 27),
        )

        rhythm_dim = getattr(args, "rhythm_f", 128)
        self.audio_rhythm_proj = nn.Linear(self.audio_f, rhythm_dim)
        self.face_rhythm_proj = nn.Linear(self.facial_f, rhythm_dim)

    @staticmethod
    def _match_time(x, target_len):
        if x.shape[1] == target_len:
            return x
        if x.shape[1] > target_len:
            return x[:, :target_len]
        pad = x[:, -1:].expand(-1, target_len - x.shape[1], -1)
        return torch.cat([x, pad], dim=1)

    def _merge_pose(self, body_pose, hand_pose):
        out = body_pose.new_zeros((body_pose.shape[0], body_pose.shape[1], self.pose_dims))
        if self.pose_dims == 141:
            out[:, :, 0:18] = body_pose[:, :, 0:18]
            out[:, :, 18:75] = hand_pose[:, :, 0:57]
            out[:, :, 75:84] = body_pose[:, :, 18:27]
            out[:, :, 84:141] = hand_pose[:, :, 57:114]
        else:
            out[:, :, :27] = body_pose
            out[:, :, 27:] = hand_pose
        return out

    def forward(
        self,
        pre_seq,
        in_audio=None,
        in_facial=None,
        in_text=None,
        in_id=None,
        in_emo=None,
        is_test=False,
        tar_pose=None,
        return_dict=False,
        classifier_only=False,
        emotion_prob=None,
        speaker_prob=None,
    ):
        target_len = pre_seq.shape[1]
        audio_feat = self.audio_encoder(in_audio)
        audio_feat = self._match_time(audio_feat, target_len)
        if classifier_only and emotion_prob is None:
            style, emotion_logits, speaker_logits = self.style_encoder(audio_feat)
            return {
                "style": style,
                "emotion_logits": emotion_logits,
                "speaker_logits": speaker_logits,
            }

        if self.text_encoder is not None and in_text is not None:
            text_feat, _ = self.text_encoder(in_text)
            text_feat = self._match_time(text_feat, target_len)
        else:
            text_feat = audio_feat.new_zeros(audio_feat.shape[0], target_len, self.word_f)

        face_pred, face_feat = self.face_generator(audio_feat)
        if emotion_prob is not None and speaker_prob is not None:
            style, emotion_logits, speaker_logits = self.style_encoder.style_from_probs(
                emotion_prob, speaker_prob, target_len
            )
        else:
            style, emotion_logits, speaker_logits = self.style_encoder(audio_feat)

        body_context = torch.cat([pre_seq, audio_feat, text_feat, face_feat], dim=-1)
        body_context = self.body_channel_attn(body_context) * body_context
        body_context = self.body_adaln(body_context, style)
        body_h, _ = self.body_lstm(body_context)
        body_h = body_h[:, :, : self.hidden_size] + body_h[:, :, self.hidden_size :]
        body_pose = self.body_out(body_h.reshape(-1, body_h.shape[-1]))
        body_pose = body_pose.reshape(pre_seq.shape[0], target_len, -1)

        hand_context = torch.cat([body_context, body_h, body_pose], dim=-1)
        hand_context = self.hand_channel_attn(hand_context) * hand_context
        hand_context = self.hand_adaln(hand_context, style)
        hand_h, _ = self.hand_lstm(hand_context)
        hand_h = hand_h[:, :, : self.hidden_size] + hand_h[:, :, self.hidden_size :]
        hand_pose = self.hand_out(hand_h.reshape(-1, hand_h.shape[-1]))
        hand_pose = hand_pose.reshape(pre_seq.shape[0], target_len, -1)

        pose = self._merge_pose(body_pose, hand_pose)
        if not return_dict:
            return pose

        return {
            "pose": pose,
            "face": face_pred,
            "face_feature": face_feat,
            "audio_feature": audio_feat,
            "audio_rhythm": self.audio_rhythm_proj(audio_feat),
            "face_rhythm": self.face_rhythm_proj(face_feat),
            "emotion_logits": emotion_logits,
            "speaker_logits": speaker_logits,
        }


class ConvDiscriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_size = args.pose_dims
        self.hidden_size = 64
        self.pre_conv = nn.Sequential(
            nn.Conv1d(self.input_size, 16, 3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(True),
            nn.Conv1d(16, 8, 3),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(True),
            nn.Conv1d(8, 8, 3),
        )
        self.lstm = nn.LSTM(
            8,
            hidden_size=self.hidden_size,
            num_layers=4,
            bidirectional=True,
            dropout=0.3,
            batch_first=True,
        )
        self.out = nn.Linear(self.hidden_size, 1)
        self.out2 = nn.Linear(34 - 6, 1)

    def forward(self, poses):
        feat = self.pre_conv(poses.transpose(1, 2)).transpose(1, 2)
        output, _ = self.lstm(feat)
        output = output[:, :, : self.hidden_size] + output[:, :, self.hidden_size :]
        output = self.out(output.contiguous().view(-1, output.shape[2]))
        output = output.view(poses.shape[0], -1)
        output = self.out2(output)
        return torch.sigmoid(output)
