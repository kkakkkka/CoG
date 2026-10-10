import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from loguru import logger

import train
from dataloaders import data_tools
from optimizers.loss_factory import get_loss_func
from utils import metric, other_tools


class CustomTrainer(train.BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.word_rep = args.word_rep
        self.emo_rep = args.emo_rep
        self.sem_rep = args.sem_rep
        self.speaker_id = args.speaker_id
        self.alignmenter = metric.alignment(0.3, 2)
        self.face_weight = getattr(args, "face_weight", 10.0)
        self.rhythm_weight = getattr(args, "rhythm_weight", 0.1)
        self.emotion_cls_weight = getattr(args, "emotion_cls_weight", 0.1)
        self.speaker_cls_weight = getattr(args, "speaker_cls_weight", 0.1)
        self.rhythm_temperature = getattr(args, "rhythm_temperature", 0.07)
        self.face_loss_mode = getattr(args, "face_loss_mode", "smooth_l1")
        self.rhythm_mode = getattr(args, "rhythm_mode", "sequence")
        self.pose_rec_mode = getattr(args, "pose_rec_mode", "huber")
        self.hand_rec_weight = getattr(args, "hand_rec_weight", 1.0)
        self.stage_schedule = getattr(args, "stage_schedule", False)
        self._style_classifier_frozen = False
        self.stage1_epochs = getattr(args, "stage1_epochs", 4)
        self.loss_item_batch_cpu = getattr(args, "loss_item_batch_cpu", True)
        self.zero_grad_set_to_none = getattr(args, "zero_grad_set_to_none", True)
        self.stage1_scales = {
            "rec": getattr(args, "stage1_rec_scale", 1.0),
            "face": getattr(args, "stage1_face_scale", 0.2),
            "rhythm": getattr(args, "stage1_rhythm_scale", 1.0),
            "emo_cls": getattr(args, "stage1_emotion_cls_scale", 0.2),
            "speaker_cls": getattr(args, "stage1_speaker_cls_scale", 0.2),
        }
        self.rec_loss = get_loss_func("huber_loss")
        self.loss_meters.update(
            {
                "face": other_tools.AverageMeter("face"),
                "rhythm": other_tools.AverageMeter("rhythm"),
                "emo_cls": other_tools.AverageMeter("emo_cls"),
                "speaker_cls": other_tools.AverageMeter("speaker_cls"),
            }
        )
        if self.rank == 0 and self.stage_schedule:
            logger.info(
                "Stage schedule enabled: "
                f"stage1_epochs={self.stage1_epochs}, stage1_scales={self.stage1_scales}"
            )

    def _stage_scale(self, epoch, name):
        if self.stage_schedule and epoch < self.stage1_epochs:
            return self.stage1_scales[name]
        return 1.0

    def _make_pre_pose(self, tar_pose):
        pre_pose = tar_pose.new_zeros((tar_pose.shape[0], tar_pose.shape[1], tar_pose.shape[2] + 1))
        pre_pose[:, 0 : self.pre_frames, :-1] = tar_pose[:, 0 : self.pre_frames]
        pre_pose[:, 0 : self.pre_frames, -1] = 1
        return pre_pose

    def _unpack_batch(self, batch_data):
        tar_pose = batch_data["pose"].to(self.device, non_blocking=True)
        in_audio = batch_data["audio"].to(self.device, non_blocking=True) if self.audio_rep is not None else None
        in_facial = batch_data["facial"].to(self.device, non_blocking=True) if self.facial_rep is not None else None
        in_id = batch_data["id"].to(self.device, non_blocking=True) if self.speaker_id else None
        in_word = batch_data["word"].to(self.device, non_blocking=True) if self.word_rep is not None else None
        in_emo = batch_data["emo"].to(self.device, non_blocking=True) if self.emo_rep is not None else None
        in_sem = batch_data["sem"].to(self.device, non_blocking=True) if self.sem_rep is not None else None
        return tar_pose, in_audio, in_facial, in_word, in_id, in_emo, in_sem

    def _style_batch(self, batch_data):
        if "style_emo" not in batch_data:
            return None, None
        return (
            batch_data["style_emo"].to(self.device, non_blocking=True),
            batch_data["style_spk"].to(self.device, non_blocking=True),
        )

    def _rhythm_info_nce(self, audio_feature, face_feature):
        if self.rhythm_mode == "frame":
            audio_feature = F.normalize(audio_feature, dim=-1)
            face_feature = F.normalize(face_feature, dim=-1)
            logits = torch.matmul(face_feature, audio_feature.transpose(-1, -2)) / self.rhythm_temperature
            batch, frames, _ = logits.shape
            labels = torch.arange(frames, device=logits.device).expand(batch, frames).reshape(-1)
            return F.cross_entropy(logits.reshape(batch * frames, frames), labels)
        audio_feature = F.normalize(audio_feature.mean(dim=1), dim=-1)
        face_feature = F.normalize(face_feature.mean(dim=1), dim=-1)
        logits = audio_feature @ face_feature.t() / self.rhythm_temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    def _split_body_hand(self, pose):
        if pose.shape[-1] == 141:
            body = torch.cat((pose[..., 0:18], pose[..., 75:84]), dim=-1)
            hand = torch.cat((pose[..., 18:75], pose[..., 84:141]), dim=-1)
            return body, hand
        return pose[..., :27], pose[..., 27:]

    def _losses(self, outputs, tar_pose, in_facial, in_sem, in_id, in_emo, epoch):
        if "pose" not in outputs:
            rec = tar_pose.new_tensor(0.0)
        elif self.pose_rec_mode == "l1_split":
            body_target, hand_target = self._split_body_hand(tar_pose)
            body_pred, hand_pred = self._split_body_hand(outputs["pose"])
            rec = F.l1_loss(body_pred, body_target) + self.hand_rec_weight * F.l1_loss(hand_pred, hand_target)
        elif in_sem is not None:
            rec = self.rec_loss(tar_pose * (in_sem.unsqueeze(2) + 1), outputs["pose"] * (in_sem.unsqueeze(2) + 1))
        else:
            rec = self.rec_loss(tar_pose, outputs["pose"])
        rec = rec * self.rec_weight * self._stage_scale(epoch, "rec")

        face = tar_pose.new_tensor(0.0)
        if in_facial is not None and "face" in outputs and outputs["face"].shape[-1] == in_facial.shape[-1]:
            if self.face_loss_mode == "mse":
                face = F.mse_loss(outputs["face"], in_facial)
            else:
                face = F.smooth_l1_loss(outputs["face"], in_facial)
            face = face * self.face_weight * self._stage_scale(epoch, "face")

        rhythm = tar_pose.new_tensor(0.0)
        if "audio_rhythm" in outputs and "face_rhythm" in outputs:
            rhythm = (
                self._rhythm_info_nce(outputs["audio_rhythm"], outputs["face_rhythm"])
                * self.rhythm_weight
                * self._stage_scale(epoch, "rhythm")
            )

        emo_cls = tar_pose.new_tensor(0.0)
        if in_emo is not None:
            emo_target = in_emo[:, 0].long().clamp(min=0, max=outputs["emotion_logits"].shape[-1] - 1)
            emo_cls = (
                F.cross_entropy(outputs["emotion_logits"], emo_target)
                * self.emotion_cls_weight
                * self._stage_scale(epoch, "emo_cls")
            )

        speaker_cls = tar_pose.new_tensor(0.0)
        if in_id is not None:
            speaker_target = in_id.reshape(in_id.shape[0], -1)[:, 0].long().clamp(
                min=0, max=outputs["speaker_logits"].shape[-1] - 1
            )
            speaker_cls = (
                F.cross_entropy(outputs["speaker_logits"], speaker_target)
                * self.speaker_cls_weight
                * self._stage_scale(epoch, "speaker_cls")
            )

        total = rec + face + rhythm + emo_cls + speaker_cls
        return total, rec, face, rhythm, emo_cls, speaker_cls

    def _norm_or_zero(self, tensors, device):
        vals = [torch.linalg.vector_norm(t.detach()) for t in tensors if t is not None]
        if not vals:
            return torch.zeros((), device=device)
        return torch.linalg.vector_norm(torch.stack(vals))

    def _monitor_metrics(self, pre_clip_norm=None):
        params = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]
        device = self.device
        with torch.no_grad():
            param_tensors = [p for _, p in params]
            grad_tensors = [p.grad for _, p in params if p.grad is not None]
            param_norm = self._norm_or_zero(param_tensors, device)
            post_clip_grad_norm = self._norm_or_zero(grad_tensors, device)
            pre_clip_grad_norm = (
                pre_clip_norm.detach().to(device)
                if pre_clip_norm is not None
                else post_clip_grad_norm
            )
            grad_abs_max = (
                torch.stack([g.detach().abs().max() for g in grad_tensors]).max()
                if grad_tensors
                else torch.zeros((), device=device)
            )
            missing_grad = sum(1 for _, p in params if p.grad is None)
            nonfinite_grad = (
                torch.stack([(~torch.isfinite(g.detach()).all()).to(device=device, dtype=torch.float32) for g in grad_tensors]).sum()
                if grad_tensors
                else torch.zeros((), device=device)
            )
            nonfinite_param = (
                torch.stack([(~torch.isfinite(p.detach()).all()).to(device=device, dtype=torch.float32) for _, p in params]).sum()
                if params
                else torch.zeros((), device=device)
            )
            global_values = torch.stack(
                [
                    pre_clip_grad_norm,
                    post_clip_grad_norm,
                    param_norm,
                    grad_abs_max,
                    post_clip_grad_norm / torch.clamp(param_norm, min=1e-12),
                    nonfinite_grad,
                    nonfinite_param,
                ]
            ).detach().cpu().tolist()
            extra = {
                "monitor/grad_norm_pre_clip": global_values[0],
                "monitor/grad_norm_post_clip": global_values[1],
                "monitor/param_norm": global_values[2],
                "monitor/grad_abs_max": global_values[3],
                "monitor/grad_param_ratio": global_values[4],
                "monitor/nonfinite_grad_tensors": global_values[5],
                "monitor/nonfinite_param_tensors": global_values[6],
                "monitor/missing_grad_tensors": missing_grad,
                "monitor/clip_threshold": float(self.grad_norm),
                "monitor/grad_clip_scale": min(1.0, float(self.grad_norm) / max(global_values[0], 1e-12)) if self.grad_norm else 1.0,
            }
            module_rows = []
            for module_name, module in self.model.named_children():
                module_params = [p for p in module.parameters() if p.requires_grad]
                if not module_params:
                    continue
                module_param_norm = self._norm_or_zero(module_params, device)
                module_grad_norm = self._norm_or_zero([p.grad for p in module_params if p.grad is not None], device)
                module_rows.append((module_name, module_param_norm, module_grad_norm))
            if module_rows:
                module_values = torch.stack(
                    [value for _, pnorm, gnorm in module_rows for value in (pnorm, gnorm)]
                ).detach().cpu().tolist()
                for idx, (module_name, _, _) in enumerate(module_rows):
                    extra[f"monitor/module/{module_name}/param_norm"] = module_values[2 * idx]
                    extra[f"monitor/module/{module_name}/grad_norm"] = module_values[2 * idx + 1]
            return extra

    def _classifier_only(self, epoch):
        return self.stage_schedule and epoch < self.stage1_epochs

    def _freeze_style_classifier(self):
        if self._style_classifier_frozen:
            return
        self.model.style_encoder.set_classifier_trainable(False)
        self._style_classifier_frozen = True
        if self.rank == 0:
            logger.info("Froze emotion/speaker classifier. Style hypernetwork stays trainable.")

    def train(self, epoch):
        classifier_only = self._classifier_only(epoch)
        if self.stage_schedule and not classifier_only:
            self._freeze_style_classifier()
        self.model.train()
        if self._style_classifier_frozen:
            self.model.style_encoder.set_classifier_trainable(False)
        its_len = len(self.train_loader)
        t_start = time.time()
        for its, batch_data in enumerate(self.train_loader):
            t_data = time.time() - t_start
            tar_pose, in_audio, in_facial, in_word, in_id, in_emo, in_sem = self._unpack_batch(batch_data)
            pre_pose = self._make_pre_pose(tar_pose)

            self.opt.zero_grad(set_to_none=self.zero_grad_set_to_none)
            emotion_prob, speaker_prob = self._style_batch(batch_data)
            outputs = self.model(
                pre_pose,
                in_audio=in_audio,
                in_facial=in_facial,
                in_text=in_word,
                in_id=in_id,
                in_emo=in_emo,
                return_dict=True,
                classifier_only=classifier_only,
                emotion_prob=emotion_prob,
                speaker_prob=speaker_prob,
            )
            total, rec, face, rhythm, emo_cls, speaker_cls = self._losses(
                outputs, tar_pose, in_facial, in_sem, in_id, in_emo, epoch
            )
            total.backward()
            grad_total_norm = None
            if self.grad_norm != 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            extra = None
            should_monitor = self.rank == 0 and its % self.log_period == 0
            if should_monitor:
                extra = self._monitor_metrics(grad_total_norm)
            self.opt.step()

            if self.loss_item_batch_cpu:
                loss_values = torch.stack(
                    [
                        total.detach(),
                        rec.detach(),
                        face.detach(),
                        rhythm.detach(),
                        emo_cls.detach(),
                        speaker_cls.detach(),
                    ]
                ).cpu().tolist()
            else:
                loss_values = [
                    total.item(),
                    rec.item(),
                    face.item(),
                    rhythm.item(),
                    emo_cls.item(),
                    speaker_cls.item(),
                ]
            self.loss_meters["all"].update(loss_values[0])
            self.loss_meters["rec"].update(loss_values[1])
            self.loss_meters["face"].update(loss_values[2])
            self.loss_meters["rhythm"].update(loss_values[3])
            self.loss_meters["emo_cls"].update(loss_values[4])
            self.loss_meters["speaker_cls"].update(loss_values[5])

            t_train = time.time() - t_start - t_data
            t_start = time.time()
            mem_cost = torch.cuda.memory_reserved(self.device) / 1e9 if self.device.type == "cuda" else 0.0
            lr_g = self.opt.param_groups[0]["lr"]
            if should_monitor:
                if self.device.type == "cuda":
                    extra.update({
                        "system/gpu_allocated_gb": torch.cuda.memory_allocated(self.device) / 1e9,
                        "system/gpu_peak_allocated_gb": torch.cuda.max_memory_allocated(self.device) / 1e9,
                    })
                self.recording(epoch, its, its_len, self.loss_meters, lr_g, 0.0, t_data, t_train, mem_cost, extra)
        self.opt_s.step(epoch)

    def val(self, epoch):
        self.model.eval()
        if hasattr(self, "eval_model") and getattr(self, "freeze_eval_model", False):
            self.eval_model.eval()
        latent_out_chunks = []
        latent_ori_chunks = []
        rec_sum = 0.0
        rec_count = 0
        val_len = len(self.val_loader)
        if self.rank == 0:
            logger.info(f"[val][{epoch:03d}] start validation: {val_len} batches")
        with torch.no_grad():
            for its, batch_data in enumerate(self.val_loader):
                tar_pose, in_audio, in_facial, in_word, in_id, in_emo, in_sem = self._unpack_batch(batch_data)
                pre_pose = self._make_pre_pose(tar_pose)
                emotion_prob, speaker_prob = self._style_batch(batch_data)
                out_pose = self.model(
                    pre_pose,
                    in_audio=in_audio,
                    in_facial=in_facial,
                    in_text=in_word,
                    in_id=in_id,
                    in_emo=in_emo,
                    emotion_prob=emotion_prob,
                    speaker_prob=speaker_prob,
                )

                if hasattr(self, "eval_model"):
                    latent_out = self.eval_model(out_pose)
                    latent_ori = self.eval_model(tar_pose)
                    latent_out_chunks.append(latent_out.cpu().numpy())
                    latent_ori_chunks.append(latent_ori.cpu().numpy())

                rec = self.rec_loss(tar_pose, out_pose) * self.rec_weight
                batch_size = tar_pose.shape[0]
                rec_sum += rec.item() * batch_size
                rec_count += batch_size
                if self.rank == 0 and (its % self.log_period == 0 or its == val_len - 1):
                    rec_avg = rec_sum / max(rec_count, 1)
                    mem_cost = torch.cuda.memory_reserved(self.device) / 1e9 if self.device.type == "cuda" else 0.0
                    logger.info(
                        f"[val][{epoch:03d}][{its:03d}/{val_len:03d}] "
                        f"rec_avg: {rec_avg:.3f}\t"
                        f"pose: {tuple(tar_pose.shape)}\t"
                        f"audio: {tuple(in_audio.shape) if in_audio is not None else None}\t"
                        f"mem: {mem_cost:.2f}"
                    )

        local_result = {
            "latent_out": np.concatenate(latent_out_chunks, axis=0) if latent_out_chunks else np.empty((0, 0)),
            "latent_ori": np.concatenate(latent_ori_chunks, axis=0) if latent_ori_chunks else np.empty((0, 0)),
            "rec_sum": rec_sum,
            "rec_count": rec_count,
        }
        if self.ddp:
            gathered = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, local_result)
        else:
            gathered = [local_result]

        if self.rank == 0:
            latent_out_items = [item["latent_out"] for item in gathered if item["latent_out"].size]
            latent_ori_items = [item["latent_ori"] for item in gathered if item["latent_ori"].size]
            latent_out_all = np.concatenate(latent_out_items, axis=0) if latent_out_items else np.empty((0, 0))
            latent_ori_all = np.concatenate(latent_ori_items, axis=0) if latent_ori_items else np.empty((0, 0))
            rec_count_all = sum(item["rec_count"] for item in gathered)
            rec_sum_all = sum(item["rec_sum"] for item in gathered)
            if rec_count_all > 0:
                self.loss_meters["rec_val"].update(rec_sum_all / rec_count_all, rec_count_all)
            if latent_out_all.size and latent_ori_all.size:
                fid = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
                self.loss_meters["fid_val"].update(fid)
            self.val_recording(epoch, self.loss_meters)

    def test(self, epoch):
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if not os.path.exists(results_save_path):
            os.makedirs(results_save_path)
        test_seq_list = os.listdir(self.test_demo)
        test_seq_list.sort()
        total_length = 0
        self.model.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                tar_pose, in_audio, in_facial, in_word, in_id, in_emo, _ = self._unpack_batch(batch_data)
                pre_pose = self._make_pre_pose(tar_pose)
                in_audio = in_audio.reshape(1, -1)
                emotion_prob, speaker_prob = self._style_batch(batch_data)
                out_pose = self.model(
                    pre_pose,
                    in_audio=in_audio,
                    in_facial=in_facial,
                    in_text=in_word,
                    in_id=in_id,
                    in_emo=in_emo,
                    is_test=True,
                    emotion_prob=emotion_prob,
                    speaker_prob=speaker_prob,
                )
                out_final = (out_pose.cpu().numpy().reshape(-1, self.pose_dims) * self.std_pose) + self.mean_pose
                total_length += out_final.shape[0]
                with open(f"{results_save_path}result_raw_{test_seq_list[its]}", "w+") as f_real:
                    for line_id in range(out_final.shape[0]):
                        line_data = np.array2string(
                            out_final[line_id], max_line_width=np.inf, precision=6, suppress_small=False, separator=" "
                        )
                        f_real.write(line_data[1:-2] + "\n")
        data_tools.result2target_vis(self.pose_version, results_save_path, results_save_path, self.test_demo, False)
        logger.info(f"total inference time for {int(total_length / self.pose_fps)} s motion")
