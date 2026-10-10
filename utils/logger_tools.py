import os
import inspect
import sys
import yaml
try:
    import wandb
except ImportError:
    from utils import wandb_stub as wandb
import torch.distributed as dist
from loguru import logger


def setup_logger(save_dir, distributed_rank=0, filename="log.txt", mode="a"):
    """setup logger for training and testing.
    Args:
        save_dir(str): location to save log file
        distributed_rank(int): device rank when multi-gpu environment
        filename (string): log save name.
        mode(str): log file write mode, `append` or `override`. default is `a`.

    Return:
        logger instance.
    """
    loguru_format = (
        "<blue>{time: MM-DD HH:mm:ss}</blue> | "
        #"<level>{level: <8}</level> | "
        #"<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        "<level>{message}</level>"
    )

    logger.remove()
    save_file = os.path.join(save_dir, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)
    # only keep logger in rank0 process
    if distributed_rank == 0:
        logger.add(
            sys.stderr,
            format=loguru_format,
            level="INFO",
            enqueue=True,
        )
        logger.add(save_file,
            format=loguru_format,       
                  )


def set_args_and_logger(args, rank):
    """
    set logger file and print args
    """
    args_name_dir = args.root_path+args.out_root_path + "custom/" + args.name + args.notes + "/"
    os.makedirs(args_name_dir, exist_ok=True)
    args_name = args_name_dir + "/" + args.name +".yaml"
    
    os.environ["WANDB_MODE"] = "offline"
    setup_logger(args_name_dir, rank, filename=f"{args.name}.txt")
    serializable_args = {k: (str(v) if k == "device" else v) for k, v in args.__dict__.items()}
    
    if rank == 0:
        wandb.init(project=args.project, dir=args.root_path+args.out_root_path, name=args.name[12:]+ args.notes, mode="offline")
        wandb.config.update(serializable_args)
        if os.path.exists(args_name):
            logger.warning("Args file already exists; preserving random_seed for reproducible resume")
        else:
            with open(args_name, "w+") as f:
                yaml.dump(serializable_args, f)

    if dist.is_available() and dist.is_initialized():
        seed_payload = [args.random_seed]
        dist.broadcast_object_list(seed_payload, src=0)
        args.random_seed = seed_payload[0]
        logger.info("init barrier...")
        dist.barrier()
