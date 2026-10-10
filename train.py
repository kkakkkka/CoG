# Copyright (c) HuaWei, Inc. and its affiliates.
# liu.haiyang@huawei.com
# Train script for audio2pose

import glob
import json
import math
import os
import signal
import time
import csv
import sys
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import numpy as np
import time
import pprint
from loguru import logger
try:
    import wandb
except ImportError:
    from utils import wandb_stub as wandb

from utils import config, logger_tools, other_tools
from dataloaders import data_tools
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return None
    return number


def setup_runtime(rank, world_size, args):
    if torch.cuda.is_available():
        if len(args.gpus) == 0:
            raise RuntimeError("CUDA is available but no GPU ids were provided in --gpus.")
        if rank >= len(args.gpus):
            raise RuntimeError(f"rank {rank} has no matching GPU in args.gpus={args.gpus}.")
        device = torch.device("cuda", args.gpus[rank])
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    args.rank = rank
    args.world_size = world_size
    args.device = device
    if args.ddp or world_size > 1:
        dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def wrap_model_for_runtime(model, args):
    model = model.to(args.device)
    if args.ddp:
        if args.device.type != "cuda":
            return DDP(model, find_unused_parameters=False)
        return DDP(
            model,
            device_ids=[args.device.index],
            output_device=args.device.index,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    if args.device.type == "cuda" and len(args.gpus) > 1:
        return torch.nn.DataParallel(model, device_ids=args.gpus).cuda(args.device)
    return model


def optimized_loader_kwargs(args):
    kwargs = {}
    if getattr(args, "loader_pin_memory", True) and getattr(args, "device", None) is not None and args.device.type == "cuda":
        kwargs["pin_memory"] = True
    prefetch_factor = getattr(args, "loader_prefetch_factor", None)
    if getattr(args, "loader_workers", 0) > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


def should_watch_wandb(args):
    env_value = os.environ.get("COG_WANDB_WATCH")
    if env_value is not None:
        return env_value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(getattr(args, "wandb_watch", False))


def watch_model(model, args):
    if should_watch_wandb(args):
        wandb.watch(
            model,
            log="all",
            log_freq=max(int(getattr(args, "wandb_histogram_period", 500)), 1),
            log_graph=False,
        )


# def init_model(args, _device):
def init_model(args):
    raise RuntimeError("Diffusion mode was removed from the cleaned CoG package.")


class BaseTrainer(object):
    def __init__(self, args):
        self.notes = args.notes
        self.ddp = args.ddp
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.device = args.device
        self.freeze_eval_model = bool(getattr(args, "freeze_eval_model", False))
        self.checkpoint_path = args.root_path+args.out_root_path + "custom/" + args.name + args.notes + "/" #wandb.run.dir #args.root_path+args.out_root_path+"/"+args.name
        self.batch_size = args.batch_size
        self.gpus = len(args.gpus)
        
        self.best_epochs = {
            'fid_val': [np.inf, 0],
            'rec_val': [np.inf, 0],
                           }
        self.loss_meters = {
            'fid_val': other_tools.AverageMeter('fid_val'),
            'rec_val': other_tools.AverageMeter('rec_val'),
            'all': other_tools.AverageMeter('all'),
            'rec': other_tools.AverageMeter('rec'), 
            'gen': other_tools.AverageMeter('gen'),
            'dis': other_tools.AverageMeter('dis'), 
        } 
        self.pose_version = args.pose_version
        # data and path
        self.mean_pose = np.load(args.root_path+args.mean_pose_path+f"{args.pose_rep}/bvh_mean.npy")
        self.std_pose = np.load(args.root_path+args.mean_pose_path+f"{args.pose_rep}/bvh_std.npy")
        
        # pose
        self.pose_rep = args.pose_rep 
        self.pose_fps = args.pose_fps
        self.pose_dims = args.pose_dims
        # audio
        self.audio_rep = args.audio_rep
        self.audio_fps = args.audio_fps
        #self.audio_dims = args.audio_dims
        # facial
        self.facial_rep = args.facial_rep
        self.facial_fps = args.facial_fps
        self.facial_dims = args.facial_dims
        
        # model para    
        self.pre_frames = args.pre_frames
        self.rec_loss = get_loss_func("huber_loss")
        self.adv_loss = get_loss_func("bce_loss")
        self.fid_loss = get_loss_func("l2_loss")
        self.vel_loss = get_loss_func("l2_loss")
        self.acc_loss = get_loss_func("l2_loss")
        # TODO: 
        # self.pos_loss        
        self.rec_weight = args.rec_weight
        self.adv_weight = args.adv_weight
        self.fid_weight = args.fid_weight
        self.vel_weight = args.vel_weight
        self.acc_weight = args.acc_weight
        self.grad_norm = args.grad_norm 
      
        self.no_adv_epochs = args.no_adv_epochs
        self.log_period = max(args.log_period // max(args.batch_size // 256, 1), 1)
        self.test_demo = args.root_path + args.test_data_path + f"{args.pose_rep}_vis/"
        
        self.train_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "train")
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data, 
            batch_size=args.batch_size,  
            shuffle=False if self.ddp else True,  
            num_workers=args.loader_workers,
            drop_last=True,
            sampler=torch.utils.data.distributed.DistributedSampler(self.train_data) if self.ddp else None, 
            **optimized_loader_kwargs(args),
        )
        self.train_length = len(self.train_loader)
        logger.info(f"Init train dataloader success")
       
        self.val_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "val")  
        self.val_loader = torch.utils.data.DataLoader(
            self.val_data, 
            batch_size=args.batch_size,  
            shuffle=False,  
            num_workers=args.loader_workers,
            drop_last=False,
            sampler=torch.utils.data.distributed.DistributedSampler(self.val_data) if self.ddp else None, 
            **optimized_loader_kwargs(args),
        )
        logger.info(f"Init val dataloader success")
        self.test_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "test")
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data, 
            batch_size=1,  
            shuffle=False,  
            num_workers=args.loader_workers,
            drop_last=False,
            **optimized_loader_kwargs(args),
        )
        logger.info(f"Init test dataloader success")
        
        model_module = __import__(f"models.{args.model}", fromlist=["something"])
        
        if self.ddp:
            self.model = getattr(model_module, args.g_name)(args)
            if args.pretrained_ckpt:
                other_tools.load_checkpoints(
                    self.model,
                    other_tools.resolve_path(args.root_path, args.pretrained_ckpt),
                    args.g_name,
                    strict=args.pretrained_load_mode == "strict",
                    include_prefixes=args.pretrained_prefixes,
                )
            if self.device.type == "cuda":
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = wrap_model_for_runtime(self.model, args)
        else: 
            if args.use_df:
                self.model = init_model(args)
            else:
                self.model = getattr(model_module, args.g_name)(args)
                if args.pretrained_ckpt:
                    other_tools.load_checkpoints(
                        self.model,
                        other_tools.resolve_path(args.root_path, args.pretrained_ckpt),
                        args.g_name,
                        strict=args.pretrained_load_mode == "strict",
                        include_prefixes=args.pretrained_prefixes,
                    )
            self.model = wrap_model_for_runtime(self.model, args)
        if self.rank == 0:
            logger.info(self.model)
            watch_model(self.model, args)
            logger.info(f"init {args.g_name} success")
        
        if args.d_name:
            if self.ddp:
                self.d_model = getattr(model_module, args.d_name)(args)
                if self.device.type == "cuda":
                    self.d_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.d_model)
                self.d_model = wrap_model_for_runtime(self.d_model, args)
            else:    
                self.d_model = wrap_model_for_runtime(getattr(model_module, args.d_name)(args), args)
            if self.rank == 0:
                logger.info(self.d_model)
                watch_model(self.d_model, args)
                logger.info(f"init {args.d_name} success")
            self.opt_d = create_optimizer(args, self.d_model, lr_weight=args.d_lr_weight)
            self.opt_d_s = create_scheduler(args, self.opt_d)
            
        if args.e_name is not None:
            eval_model_module = __import__(f"models.{args.eval_model}", fromlist=["something"])
            self.eval_model = getattr(eval_model_module, args.e_name)(args)
            other_tools.load_checkpoints(self.eval_model, other_tools.resolve_path(args.root_path, args.e_path), args.e_name)
            if self.ddp and self.device.type == "cuda":
                self.eval_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.eval_model)
            self.eval_model = wrap_model_for_runtime(self.eval_model, args)
            if self.freeze_eval_model:
                self.eval_model.eval()
                for parameter in self.eval_model.parameters():
                    parameter.requires_grad_(False)
            if self.rank == 0:
                logger.info(self.eval_model)
                if self.freeze_eval_model:
                    logger.info("Frozen eval_model for validation/FID: eval() mode and requires_grad=False")
                logger.info(f"init {args.e_name} success")
        
        if getattr(args, "frozen_style_dir", ""):
            style_model = self.model.module if hasattr(self.model, "module") else self.model
            style_model.style_encoder.set_classifier_trainable(False)
            if self.rank == 0:
                logger.info("External emotion/speaker distributions are frozen. The internal classifier is not trained.")
        self.opt = create_optimizer(args, self.model)
        self.opt_s = create_scheduler(args, self.opt)
       
    def recording(self, epoch, its, its_len, loss_meters, lr_g, lr_d, t_data, t_train, mem_cost, extra_metrics=None):
        if self.rank == 0:
            pstr = "[%03d][%03d/%03d]  "%(epoch, its, its_len)
            payload = {}
            for name, loss_meter in self.loss_meters.items():
                if "val" not in name:
                    if loss_meter.count > 0:
                        pstr += "{}: {:.3f}\t".format(loss_meter.name, loss_meter.avg)
                        payload[f"train/{loss_meter.name}"] = loss_meter.avg
                        loss_meter.reset()
            pstr += "glr: {:.1e}\t".format(lr_g)
            pstr += "dlr: {:.1e}\t".format(lr_d)
            payload.update({'train/lr_g': lr_g, 'train/lr_d': lr_d, 'system/data_time_s': t_data, 'system/step_time_s': t_train, 'system/samples_per_s': self.batch_size / max(t_train + t_data, 1e-12), 'system/gpu_reserved_gb': mem_cost*self.gpus})
            if extra_metrics:
                payload.update(extra_metrics)
                if "monitor/grad_norm_pre_clip" in extra_metrics:
                    pstr += "gpre: {:.1f}\tgpost: {:.1f}\tpnorm: {:.1f}\tgp: {:.3e}\t".format(
                        extra_metrics["monitor/grad_norm_pre_clip"],
                        extra_metrics["monitor/grad_norm_post_clip"],
                        extra_metrics["monitor/param_norm"],
                        extra_metrics["monitor/grad_param_ratio"],
                    )
            wandb.log(payload, step=epoch*self.train_length+its)
            self._append_monitor({"split": "train", "epoch": int(epoch), "step": int(its), "step_count": int(its_len), **payload})
            pstr += "dtime: %04d\t"%(t_data*1000)        
            pstr += "ntime: %04d\t"%(t_train*1000)
            pstr += "mem: {:.2f} ".format(mem_cost*self.gpus)
            logger.info(pstr)

    def _append_monitor(self, row):
        if self.rank != 0:
            return
        path = os.path.join(self.checkpoint_path, "monitor.jsonl")
        try:
            os.makedirs(self.checkpoint_path, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(_json_ready(row), ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning(f"monitor.jsonl write failed: {exc}")

    def _save_metric_best(self, metric_name, epoch):
        directory = self.checkpoint_path
        other_tools.save_checkpoints(os.path.join(directory, f"best_{metric_name}.bin"), self.model, opt=None, epoch=None, lrs=None)
        current = os.path.join(directory, f"best_{metric_name}_epoch_{epoch:03d}.bin")
        for old in glob.glob(os.path.join(directory, f"best_{metric_name}_epoch_*.bin")):
            if os.path.abspath(old) != os.path.abspath(current):
                try:
                    os.remove(old)
                except OSError as exc:
                    logger.warning(f"could not remove old best snapshot {old}: {exc}")
        other_tools.save_checkpoints(current, self.model, opt=None, epoch=None, lrs=None)
     
    def val_recording(self, epoch, metrics):
        if self.rank == 0: 
            pstr_curr = "Curr info >>>>  "
            pstr_best = "Best info >>>>  "

            val_row = {"split": "val", "epoch": int(epoch)}
            for name, metric in metrics.items():
                if "val" in name:
                    if metric.count > 0:
                        pstr_curr += "{}: {:.3f}     \t".format(metric.name, metric.avg)
                        val_row[name] = float(metric.avg)
                        wandb.log({f"val/{metric.name}": metric.avg}, step=epoch*self.train_length)
                        if metric.avg < self.best_epochs[metric.name][0]:
                            self.best_epochs[metric.name][0] = metric.avg
                            self.best_epochs[metric.name][1] = epoch
                            self._save_metric_best(metric.name, epoch)
                        metric.reset()
            self._append_monitor(val_row)
            for k, v in self.best_epochs.items():
                pstr_best += "{}: {:.3f}({:03d})\t".format(k, v[0], v[1])
            logger.info(pstr_curr)
            logger.info(pstr_best)  

@logger.catch(reraise=True)
def main_worker(rank, world_size, args):
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    setup_runtime(rank, world_size, args)
    logger_tools.set_args_and_logger(args, rank)
    other_tools.set_random_seed(args)
    other_tools.print_exp_info(args)
    if args.resume and args.pretrained_ckpt:
        logger.info(
            "Full-state resume requested with pretrained_ckpt set in the config; "
            "skipping model-only warm-start load and restoring the full training state."
        )
        args.pretrained_ckpt = None
      
    # return one intance of trainer
    trainer = __import__(f"{args.trainer}_trainer", fromlist=["something"]).CustomTrainer(args) if args.trainer != "base" else BaseTrainer(args)
    start_epoch = 0
    global_step = 0
    if args.resume:
        if args.ddp:
            raise RuntimeError("Full-state cog-training-v2 resume is currently supported only for single-GPU training")
        resumed = other_tools.load_training_checkpoint(other_tools.resolve_path(args.root_path, args.resume), trainer, args)
        start_epoch = resumed["next_epoch"]
        global_step = resumed["global_step"]
        logger.info(f"Full-state epoch-boundary resume from {resumed['path']} at epoch {start_epoch}, global_step {global_step}")
    else:
        if args.pretrained_ckpt:
            logger.info(
                "Warm-start training from model-only checkpoint "
                f"{other_tools.resolve_path(args.root_path, args.pretrained_ckpt)} "
                f"with load_mode={args.pretrained_load_mode}, prefixes={args.pretrained_prefixes}; "
                "optimizer/scheduler/RNG/loss meters start fresh."
            )
        else:
            logger.info("Training from scratch ...")
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        if trainer.ddp: trainer.val_loader.sampler.set_epoch(epoch)
        # trainer.test(epoch)
        trainer.val(epoch)
        epoch_time = time.time()-start_time
        if trainer.rank == 0: logger.info("Time info >>>>  elapsed: %.2f mins\t"%(epoch_time/60)+"remain: %.2f mins"%((args.epochs/(epoch+1e-7)-1)*epoch_time/60))
        if args.stop_after_val_epoch >= 0 and epoch >= args.stop_after_val_epoch:
            if trainer.rank == 0:
                logger.info(f"Stopping after validation epoch {epoch} as requested.")
            break
        if trainer.ddp: trainer.train_loader.sampler.set_epoch(epoch)
        trainer.train(epoch)
        global_step += trainer.train_length
        if ((epoch+1) % args.test_period == 0):
            if rank == 0:
                trainer.test(epoch)
                other_tools.save_checkpoints(os.path.join(trainer.checkpoint_path, f"last_{epoch+1}.bin"), trainer.model, opt=None, epoch=None, lrs=None)
        if rank == 0 and ((epoch + 1) % max(args.save_full_checkpoint_every, 1) == 0):
            other_tools.save_training_checkpoint(
                os.path.join(trainer.checkpoint_path, f"training_state_epoch_{epoch+1}.pt"),
                trainer, epoch, global_step, args,
            )
            other_tools.save_training_checkpoint(
                os.path.join(trainer.checkpoint_path, "training_state_latest.pt"),
                trainer, epoch, global_step, args,
            )
            
    for k, v in trainer.best_epochs.items():
        wandb.log({f"{k}_best": v[0], f"{k}_epoch": v[1]})
    
    if rank == 0:
        wandb.finish()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    
            
if __name__ == "__main__":
    os.environ["MASTER_ADDR"]='localhost'
    args = config.parse_args()
    os.environ["MASTER_PORT"]=args.port
    if args.ddp:
        mp.set_start_method("spawn", force=True)
        mp.spawn(
            main_worker,
            args=(len(args.gpus), args,),
            nprocs=len(args.gpus),
                )
    else:
        main_worker(0, 1, args)
