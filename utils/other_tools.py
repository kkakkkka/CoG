import os
import random
import tempfile
import numpy as np
import torch
import csv
import pprint
from loguru import logger
from collections import OrderedDict


def resolve_path(root_path, path):
    if not path:
        return path
    if os.path.isabs(path):
        if os.path.exists(path):
            return path
        joined = os.path.join(root_path, path.lstrip("/"))
        return joined
    return os.path.join(root_path, path)

def print_exp_info(args):
    logger.info(pprint.pformat(vars(args)))
    logger.info(f"# ------------ {args.name} ----------- #")
    logger.info("PyTorch version: {}".format(torch.__version__))
    logger.info("CUDA version: {}".format(torch.version.cuda))
    logger.info("{} GPUs".format(torch.cuda.device_count()))
    logger.info(f"Random Seed: {args.random_seed}")

def args2csv(args, get_head=False, list4print=[]):
    for k, v in args.items():
        if isinstance(args[k], dict):
            args2csv(args[k], get_head, list4print)
        else: list4print.append(k) if get_head else list4print.append(v)
    return list4print

def record_trial(args, csv_path, best_metric, best_epoch):
    metric_name = []
    metric_value = []
    metric_epoch = []
    list4print = []
    name4print = []
    for k, v in vars(args).items():
        list4print.append(v)
        name4print.append(k)
    
    for k, v in best_metric.items():
        metric_name.append(k)
        metric_value.append(v)
        metric_epoch.append(best_epoch[k])
    
    if not os.path.exists(csv_path):
        with open(csv_path, "a+") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow([*metric_name, *metric_name, *name4print])
            
    with open(csv_path, "a+") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([*metric_value,*metric_epoch, *list4print])
        

def set_random_seed(args):
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = args.deterministic #args.CUDNN_DETERMINISTIC
    torch.backends.cudnn.benchmark = args.benchmark
    torch.backends.cudnn.enabled = args.cudnn_enabled
    

def save_checkpoints(save_path, model, opt=None, epoch=None, lrs=None):
    if lrs is not None:
        states = { 'model_state': model.state_dict(),
                'epoch': epoch + 1,
                'opt_state': opt.state_dict(),
                'lrs':lrs.state_dict(),}
    elif opt is not None:
        states = { 'model_state': model.state_dict(),
                'epoch': epoch + 1,
                'opt_state': opt.state_dict(),}
    else:
        states = { 'model_state': model.state_dict(),}
    torch.save(states, save_path)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_training_checkpoint(path, trainer, epoch, global_step, args):
    """Atomically save an epoch-boundary checkpoint suitable for full-state resume."""
    def meter_state():
        return {
            name: {
                "name": meter.name,
                "fmt": meter.fmt,
                "val": meter.val,
                "avg": meter.avg,
                "sum": meter.sum,
                "count": meter.count,
            }
            for name, meter in trainer.loss_meters.items()
        }

    state = {
        "format": "cog-training-v2",
        "model_state": _unwrap_model(trainer.model).state_dict(),
        "model_training": bool(_unwrap_model(trainer.model).training),
        "optimizer_state": trainer.opt.state_dict(),
        "scheduler_state": trainer.opt_s.state_dict(),
        "eval_model_state": _unwrap_model(trainer.eval_model).state_dict() if hasattr(trainer, "eval_model") else None,
        "eval_model_training": bool(_unwrap_model(trainer.eval_model).training) if hasattr(trainer, "eval_model") else None,
        "discriminator_state": _unwrap_model(trainer.d_model).state_dict() if hasattr(trainer, "d_model") else None,
        "discriminator_training": bool(_unwrap_model(trainer.d_model).training) if hasattr(trainer, "d_model") else None,
        "discriminator_optimizer_state": trainer.opt_d.state_dict() if hasattr(trainer, "opt_d") else None,
        "discriminator_scheduler_state": trainer.opt_d_s.state_dict() if hasattr(trainer, "opt_d_s") else None,
        "epoch_completed": int(epoch),
        "next_epoch": int(epoch) + 1,
        "global_step": int(global_step),
        "best_epochs": trainer.best_epochs,
        "loss_meters": meter_state(),
        "rng_state": capture_rng_state(),
        "protocol": training_protocol(args),
    }
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".checkpoint-", suffix=".tmp", dir=os.path.dirname(path))
    os.close(fd)
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def load_training_checkpoint(path, trainer, args):
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or state.get("format") != "cog-training-v2":
        raise RuntimeError(
            f"{path} is not a full cog-training-v2 checkpoint. "
            "Use --pretrained_ckpt for a model-only warm start; full-state resume is impossible from that file."
        )
    missing_fields = [
        key for key in ("eval_model_state", "loss_meters", "protocol", "rng_state")
        if key not in state
    ]
    if missing_fields:
        raise RuntimeError(f"resume checkpoint missing full-state fields {missing_fields}; legacy partial resume is refused")
    expected = training_protocol(args)
    actual = state.get("protocol", {})
    mismatches = {}
    for key, value in expected.items():
        actual_value = actual.get(key)
        if key == "freeze_eval_model" and actual_value is None:
            actual_value = False
        if actual_value != value:
            mismatches[key] = (actual_value, value)
    if "e_path" in mismatches and state.get("eval_model_state") is not None:
        mismatches.pop("e_path")
    if mismatches:
        raise RuntimeError(f"resume protocol mismatch: {mismatches}")
    _unwrap_model(trainer.model).load_state_dict(state["model_state"], strict=True)
    _unwrap_model(trainer.model).train(bool(state.get("model_training", True)))
    trainer.opt.load_state_dict(state["optimizer_state"])
    trainer.opt_s.load_state_dict(state["scheduler_state"])
    if hasattr(trainer, "eval_model"):
        if state.get("eval_model_state") is None:
            raise RuntimeError("resume checkpoint has no eval_model_state but trainer has eval_model")
        _unwrap_model(trainer.eval_model).load_state_dict(state["eval_model_state"], strict=True)
        _unwrap_model(trainer.eval_model).train(bool(state.get("eval_model_training", True)))
    elif state.get("eval_model_state") is not None:
        raise RuntimeError("resume checkpoint has eval_model_state but trainer has no eval_model")
    if state.get("discriminator_state") is not None:
        _unwrap_model(trainer.d_model).load_state_dict(state["discriminator_state"], strict=True)
        _unwrap_model(trainer.d_model).train(bool(state.get("discriminator_training", True)))
        trainer.opt_d.load_state_dict(state["discriminator_optimizer_state"])
        trainer.opt_d_s.load_state_dict(state["discriminator_scheduler_state"])
    trainer.best_epochs = state["best_epochs"]
    for name, saved in state["loss_meters"].items():
        if name not in trainer.loss_meters:
            raise RuntimeError(f"resume checkpoint has unexpected loss meter {name}")
        meter = trainer.loss_meters[name]
        for field in ("val", "avg", "sum", "count"):
            setattr(meter, field, saved[field])
    restore_rng_state(state["rng_state"])
    return {"next_epoch": int(state["next_epoch"]), "global_step": int(state["global_step"]), "path": str(path)}


def training_protocol(args):
    keys = [
        "model", "g_name", "d_name", "trainer", "dataset",
        "train_data_path", "val_data_path", "test_data_path",
        "audio_rep", "word_rep", "emo_rep", "sem_rep", "facial_rep", "pose_rep", "speaker_id",
        "pose_fps", "pose_dims", "pose_length", "pre_frames", "audio_f", "facial_f", "word_f",
        "speaker_dims", "emotion_dims", "style_f", "rhythm_f", "hidden_size", "n_layer",
        "dropout_prob", "batch_size", "opt", "opt_betas", "weight_decay", "lr_base", "lr_policy",
        "lr_min", "warmup_lr", "warmup_epochs", "decay_epochs", "decay_rate", "milestones",
        "grad_norm", "rec_weight", "face_weight", "rhythm_weight", "emotion_cls_weight",
        "speaker_cls_weight", "stage_schedule", "stage1_epochs", "stage1_rec_scale",
        "stage1_face_scale", "stage1_rhythm_scale", "stage1_emotion_cls_scale",
        "stage1_speaker_cls_scale", "text_tcn_conv_mode", "eval_model", "e_name", "e_path",
        "freeze_eval_model",
        "vae_length", "random_seed", "test_period", "deterministic", "benchmark", "cudnn_enabled",
        "use_aug", "pose_norm", "audio_norm", "facial_norm", "freeze_wordembed",
        "emotion_f", "speaker_f", "word_dims", "audio_fps", "facial_fps", "variational_encoding",
        "loss_item_batch_cpu", "zero_grad_set_to_none",
    ]
    return {key: _protocol_value(getattr(args, key, None)) for key in keys}


def _protocol_value(value):
    if isinstance(value, (list, tuple)):
        return [_protocol_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _checkpoint_state_dict(save_path):
    states = torch.load(save_path, map_location="cpu")
    state_dict = states["model_state"] if isinstance(states, dict) and "model_state" in states else states
    new_weights = OrderedDict()
    stripped = False
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_weights[k[7:]] = v
            stripped = True
        else:
            new_weights[k] = v
    return new_weights, stripped


def load_checkpoints(
    model,
    save_path,
    load_name="model",
    strict=True,
    include_prefixes=None,
):
    state_dict, stripped = _checkpoint_state_dict(save_path)
    if strict:
        model.load_state_dict(state_dict)
        report = {
            "path": save_path,
            "load_name": load_name,
            "mode": "strict",
            "stripped_module_prefix": stripped,
            "loaded": len(state_dict),
            "skipped": 0,
            "missing": 0,
            "unexpected": 0,
            "shape_mismatch": 0,
        }
        logger.info(f"load checkpoints for {load_name}: {report}")
        return report

    model_state = model.state_dict()
    include_prefixes = tuple(include_prefixes or [])
    loadable = OrderedDict()
    skipped_prefix = []
    unexpected = []
    shape_mismatch = []
    for k, v in state_dict.items():
        if include_prefixes and not k.startswith(include_prefixes):
            skipped_prefix.append(k)
            continue
        if k not in model_state:
            unexpected.append(k)
            continue
        if tuple(v.shape) != tuple(model_state[k].shape):
            shape_mismatch.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        loadable[k] = v

    merged_state = OrderedDict(model_state)
    merged_state.update(loadable)
    model.load_state_dict(merged_state)
    missing = [k for k in model_state if k not in loadable]
    report = {
        "path": save_path,
        "load_name": load_name,
        "mode": "partial",
        "stripped_module_prefix": stripped,
        "loaded": len(loadable),
        "skipped_prefix": len(skipped_prefix),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "shape_mismatch": len(shape_mismatch),
        "loaded_sample": list(loadable.keys())[:12],
        "unexpected_sample": unexpected[:12],
        "shape_mismatch_sample": shape_mismatch[:12],
    }
    logger.info(f"partial load checkpoints for {load_name}: {report}")
    return report


def model_complexity(model, args):
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model,  (args.T_GLOBAL._DIM, args.TRAIN.CROP, args.TRAIN), 
        as_strings=False, print_per_layer_stat=False)
    logging.info('{:<30}  {:<8} BFlops'.format('Computational complexity: ', flops / 1e9))
    logging.info('{:<30}  {:<8} MParams'.format('Number of parameters: ', params / 1e6))
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
