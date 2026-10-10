import json
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from loguru import logger

import train
from dataloaders import data_tools
from optimizers.loss_factory import get_loss_func
from utils import config, logger_tools, metric, other_tools


class TestRunner:
    def __init__(self, args):
        self.args = args
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.device = args.device
        self.pose_version = args.pose_version
        self.pose_rep = args.pose_rep
        self.pose_fps = args.pose_fps
        self.pose_dims = args.pose_dims
        self.audio_rep = args.audio_rep
        self.facial_rep = args.facial_rep
        self.word_rep = args.word_rep
        self.emo_rep = args.emo_rep
        self.sem_rep = args.sem_rep
        self.speaker_id = args.speaker_id
        self.pose_length = args.pose_length
        self.stride = args.stride
        self.pre_frames = args.pre_frames
        self.rec_weight = args.rec_weight
        self.eval_split = args.eval_split
        self.checkpoint_path = args.root_path + args.out_root_path + "custom/" + args.name + args.notes + "/"
        self.results_save_path = os.path.join(self.checkpoint_path, "test_results")

        self.mean_pose = np.load(args.root_path + args.mean_pose_path + f"{args.pose_rep}/bvh_mean.npy")
        self.std_pose = np.load(args.root_path + args.mean_pose_path + f"{args.pose_rep}/bvh_std.npy")
        self.test_demo = args.root_path + args.test_data_path + f"{args.pose_rep}_vis/"

        self.rec_loss = get_loss_func("huber_loss")
        self.alignmenter = metric.alignment(0.3, 2)
        self.srgr_calculator = metric.SRGR(4, 47)
        self.l1_calculator = metric.L1div()

        loader_type = "val" if self.eval_split == "val" else "test"
        test_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, loader_type)
        self.test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=args.batch_size if self.eval_split == "val" else 1,
            shuffle=False,
            num_workers=args.loader_workers,
            drop_last=False,
        )
        logger.info(f"Init {self.eval_split} dataloader success: {len(self.test_loader)} batches")

        model_module = __import__(f"models.{args.model}", fromlist=["something"])
        self.model = getattr(model_module, args.g_name)(args)
        self.load_report = self._load_generator_checkpoint()
        self.model = train.wrap_model_for_runtime(self.model, args)
        logger.info(f"init {args.g_name} success")

        self.eval_model = None
        self.eval_load_report = None
        if args.e_name is not None:
            eval_path = other_tools.resolve_path(args.root_path, args.e_path)
            if os.path.exists(eval_path):
                eval_model_module = __import__(f"models.{args.eval_model}", fromlist=["something"])
                self.eval_model = getattr(eval_model_module, args.e_name)(args)
                self.eval_load_report = other_tools.load_checkpoints(self.eval_model, eval_path, args.e_name)
                self.eval_model = train.wrap_model_for_runtime(self.eval_model, args)
                self.eval_model.eval()
                logger.info(f"init {args.e_name} success")
            else:
                logger.warning(f"eval checkpoint not found, skip FID: {eval_path}")

    def _load_generator_checkpoint(self):
        if self.args.test_ckpt:
            return other_tools.load_checkpoints(
                self.model,
                other_tools.resolve_path(self.args.root_path, self.args.test_ckpt),
                self.args.g_name,
                strict=True,
            )
        if self.args.pretrained_ckpt:
            return other_tools.load_checkpoints(
                self.model,
                other_tools.resolve_path(self.args.root_path, self.args.pretrained_ckpt),
                self.args.g_name,
                strict=self.args.pretrained_load_mode == "strict",
                include_prefixes=self.args.pretrained_prefixes,
            )
        logger.warning("No --test_ckpt or --pretrained_ckpt was provided; evaluating randomly initialized model.")
        return {"mode": "random_init", "loaded": 0}

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

    def _window_pose(self, out_pose, tar_pose):
        num_divs = (tar_pose.shape[1] - self.pose_length) // self.stride + 1
        cat_results = []
        cat_targets = []
        for i in range(num_divs):
            start = i * self.stride
            end = start + self.pose_length
            cat_results.append(out_pose[:, start:end, :])
            cat_targets.append(tar_pose[:, start:end, :])
        return torch.cat(cat_results, 0), torch.cat(cat_targets, 0)

    def evaluate(self):
        start_time = time.time()
        total_length = 0
        rec_sum = 0.0
        rec_count = 0
        align = 0.0
        align_enabled = True
        latent_out_chunks = []
        latent_ori_chunks = []
        test_seq_list = sorted(os.listdir(self.test_demo)) if self.eval_split == "test" else []
        t_start = 10
        t_end = 500

        if self.args.save_test_results:
            os.makedirs(self.results_save_path, exist_ok=True)

        self.model.eval()
        max_batches = self.args.test_max_batches
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                if max_batches >= 0 and its >= max_batches:
                    break
                tar_pose, in_audio, in_facial, in_word, in_id, in_emo, in_sem = self._unpack_batch(batch_data)
                pre_pose = self._make_pre_pose(tar_pose)
                model_audio = in_audio.reshape(1, -1) if in_audio is not None and self.eval_split == "test" else in_audio
                emotion_prob = speaker_prob = None
                if "style_emo" in batch_data:
                    emotion_prob = batch_data["style_emo"].to(self.device, non_blocking=True)
                    speaker_prob = batch_data["style_spk"].to(self.device, non_blocking=True)
                out_pose = self.model(
                    pre_pose,
                    in_audio=model_audio,
                    in_facial=in_facial,
                    in_text=in_word,
                    in_id=in_id,
                    in_emo=in_emo,
                    is_test=True,
                    emotion_prob=emotion_prob,
                    speaker_prob=speaker_prob,
                )

                rec = self.rec_loss(tar_pose, out_pose) * self.rec_weight
                rec_sum += rec.item() * tar_pose.shape[0]
                rec_count += tar_pose.shape[0]

                if self.eval_split == "val":
                    cat_results, cat_targets = out_pose, tar_pose
                else:
                    cat_results, cat_targets = self._window_pose(out_pose, tar_pose)
                if self.eval_model is not None:
                    latent_out_chunks.append(self.eval_model(cat_results).cpu().numpy())
                    latent_ori_chunks.append(self.eval_model(cat_targets).cpu().numpy())

                if self.eval_split == "test":
                    np_cat_results = (cat_results.cpu().numpy().reshape(-1, self.pose_dims) * self.std_pose) + self.mean_pose
                    _ = self.l1_calculator.run(np_cat_results)

                out_final = (out_pose.cpu().numpy().reshape(-1, self.pose_dims) * self.std_pose) + self.mean_pose
                np_cat_targets = (tar_pose.cpu().numpy().reshape(-1, self.pose_dims) * self.std_pose) + self.mean_pose
                if self.eval_split == "test" and in_sem is not None:
                    _ = self.srgr_calculator.run(out_final, np_cat_targets, in_sem.cpu().numpy())

                total_length += out_final.shape[0]
                if self.eval_split == "test" and in_audio is not None and align_enabled:
                    try:
                        onset_raw, onset_bt, onset_bt_rms = self.alignmenter.load_audio(
                            in_audio.cpu().numpy().reshape(-1), t_start, t_end, True
                        )
                        beats = self.alignmenter.load_pose(out_final, t_start, t_end, self.pose_fps, True)
                        align += self.alignmenter.calculate_align(
                            onset_raw,
                            onset_bt,
                            onset_bt_rms,
                            *beats,
                            self.pose_fps,
                        )
                    except ImportError as exc:
                        logger.warning(f"skip align metric: {exc}")
                        align_enabled = False

                if self.eval_split == "test" and self.args.save_test_results:
                    seq_name = test_seq_list[its] if its < len(test_seq_list) else f"{its:04d}.txt"
                    with open(os.path.join(self.results_save_path, f"result_raw_{seq_name}"), "w+") as f_real:
                        for line_id in range(out_final.shape[0]):
                            line_data = np.array2string(
                                out_final[line_id],
                                max_line_width=np.inf,
                                precision=6,
                                suppress_small=False,
                                separator=" ",
                            )
                            f_real.write(line_data[1:-2] + "\n")

                if its % max(self.args.log_period, 1) == 0 or its == len(self.test_loader) - 1:
                    logger.info(
                        f"[test][{its:03d}/{len(self.test_loader):03d}] "
                        f"rec_avg: {rec_sum / max(rec_count, 1):.3f}"
                    )

        evaluated = rec_count
        metrics = {
            "checkpoint": self.load_report,
            "eval_checkpoint": self.eval_load_report,
            "eval_split": self.eval_split,
            "num_batches": evaluated,
            "rec_test": rec_sum / max(rec_count, 1),
            "align": align / max(evaluated, 1) if self.eval_split == "test" and align_enabled else None,
            "srgr": self.srgr_calculator.avg() if self.eval_split == "test" and self.sem_rep is not None else None,
            "l1div": self.l1_calculator.avg() if self.eval_split == "test" else None,
            "fid": None,
            "fid_status": "skipped_no_eval_model",
            "total_motion_seconds": total_length / self.pose_fps,
            "total_inference_seconds": time.time() - start_time,
        }
        if latent_out_chunks and latent_ori_chunks:
            latent_out_all = np.concatenate(latent_out_chunks, axis=0)
            latent_ori_all = np.concatenate(latent_ori_chunks, axis=0)
            metrics["fid"] = float(data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all))
            metrics["fid_status"] = "invalid_sentinel" if metrics["fid"] >= 1e10 else "ok"

        metrics_path = os.path.join(self.checkpoint_path, "test_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, ensure_ascii=False)
        logger.info(f"test metrics: {json.dumps(metrics, ensure_ascii=False, indent=2)}")
        logger.info(f"wrote test metrics to {metrics_path}")
        if self.args.save_test_results:
            data_tools.result2target_vis(
                self.pose_version,
                self.results_save_path,
                self.results_save_path,
                self.test_demo,
                False,
            )
        return metrics


@logger.catch
def main_worker(rank, world_size, args):
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    train.setup_runtime(rank, world_size, args)
    logger_tools.set_args_and_logger(args, rank)
    other_tools.set_random_seed(args)
    other_tools.print_exp_info(args)
    runner = TestRunner(args)
    runner.evaluate()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    args = config.parse_args()
    os.environ["MASTER_PORT"] = args.test_port
    if args.ddp:
        mp.set_start_method("spawn", force=True)
        mp.spawn(main_worker, args=(len(args.gpus), args), nprocs=len(args.gpus))
    else:
        main_worker(0, 1, args)
