'''
Author: xiehua xiehua@akulaku.com
Date: 2023-05-16 18:08:50
LastEditors: xiehua xiehua@akulaku.com
LastEditTime: 2023-05-17 10:53:59
FilePath: /BEAT/codes/audio2pose/optimizers/timm/multistep_lr.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import torch

from .scheduler import Scheduler


class MultiStepLRScheduler(Scheduler):
    """Decay the LR by a factor every time the validation loss plateaus."""

    def __init__(self,
                 optimizer,
                 decay_rate=0.1,
                 milestones=[9999],
                 warmup_t=0,
                 warmup_lr_init=0,
                 noise_range_t=None,
                 noise_type='normal',
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=None,
                 initialize=True,
                 ):
        super().__init__(optimizer, 'lr', initialize=initialize)

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=milestones, 
            gamma=decay_rate
        )

        self.noise_range = noise_range_t
        self.noise_pct = noise_pct
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.noise_seed = noise_seed if noise_seed is not None else 42
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]
        self.restore_lr = None

    def state_dict(self):
        return {
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'restore_lr': self.restore_lr,
        }

    def load_state_dict(self, state_dict):
        self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
        self.restore_lr = state_dict.get('restore_lr')

    # override the base class step fn completely
    def step(self, epoch):
        if epoch <= self.warmup_t:
            lrs = [self.warmup_lr_init + epoch * s for s in self.warmup_steps]
            super().update_groups(lrs)
        else:
            if self.restore_lr is not None:
                # restore actual LR from before our last noise perturbation before stepping base
                for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] = self.restore_lr[i]
                self.restore_lr = None

            self.lr_scheduler.step(epoch)  # step the base scheduler

            if self.noise_range is not None:
                if isinstance(self.noise_range, (list, tuple)):
                    apply_noise = self.noise_range[0] <= epoch < self.noise_range[1]
                else:
                    apply_noise = epoch >= self.noise_range
                if apply_noise:
                    self._apply_noise(epoch)

    # def _apply_noise(self, epoch):
    #     g = torch.Generator()
    #     g.manual_seed(self.noise_seed + epoch)
    #     if self.noise_type == 'normal':
    #         while True:
    #             # resample if noise out of percent limit, brute force but shouldn't spin much
    #             noise = torch.randn(1, generator=g).item()
    #             if abs(noise) < self.noise_pct:
    #                 break
    #     else:
    #         noise = 2 * (torch.rand(1, generator=g).item() - 0.5) * self.noise_pct

    #     # apply the noise on top of previous LR, cache the old value so we can restore for normal
    #     # stepping of base scheduler
    #     restore_lr = []
    #     for i, param_group in enumerate(self.optimizer.param_groups):
    #         old_lr = float(param_group['lr'])
    #         restore_lr.append(old_lr)
    #         new_lr = old_lr + old_lr * noise
    #         param_group['lr'] = new_lr
    #     self.restore_lr = restore_lr
