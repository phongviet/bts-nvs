"""Splatfacto with gsplat's MCMCStrategy densifier instead of nerfstudio's default.

Why this exists: docs/strategy.md's Week 2 backend-locking step calls for an
"MCMC (cap_max sweep)" densifier, following Kheradmand et al. NeurIPS 2024
(3D Gaussian Splatting as Markov Chain Monte Carlo). But the installed
nerfstudio version (see models/splatfacto.py) hardcodes
`gsplat.strategy.DefaultStrategy` with no CLI switch to gsplat's
`MCMCStrategy` -- this file is that missing switch, implemented as a
SplatfactoModel subclass registered as a new `ns-train` method
("splatfacto-mcmc") via src/register_custom_methods.py.

Differences from DefaultStrategy handled here (see gsplat.strategy.MCMCStrategy
source for ground truth):
  - `initialize_state()` takes no `scene_scale` kwarg (DefaultStrategy's does).
  - `step_post_backward(...)` takes `lr` (current "means" optimizer LR) instead
    of `packed`; used to scale the position-perturbation noise.
  - MCMCStrategy has no `absgrad` field (DefaultStrategy does, and
    SplatfactoModel.get_outputs reads `self.strategy.absgrad` when building the
    rasterization call) -- set manually to False after construction (MCMC's
    own reference training loop does not use absgrad).
  - No `step_pre_backward` override (inherits the no-op base implementation),
    so no change needed there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from gsplat.strategy import MCMCStrategy
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig


@dataclass
class SplatfactoMCMCModelConfig(SplatfactoModelConfig):
    """Config for SplatfactoMCMCModel -- adds MCMC-specific knobs on top of the
    base splatfacto config. `warmup_length`, `stop_split_at`, and
    `refine_every` are reused as MCMC's refine_start_iter/refine_stop_iter/
    refine_every so the two strategies stay comparable in an A/B."""

    _target: Type = field(default_factory=lambda: SplatfactoMCMCModel)
    cap_max: int = 1_000_000
    """Maximum number of Gaussians MCMC will grow to (the knob docs/strategy.md calls out to sweep)."""
    mcmc_noise_lr: float = 5e5
    """MCMC position-perturbation noise learning rate (gsplat default 5e5)."""
    mcmc_min_opacity: float = 0.005
    """Gaussians with opacity below this are pruned/relocated by MCMC."""


class SplatfactoMCMCModel(SplatfactoModel):
    config: SplatfactoMCMCModelConfig

    def populate_modules(self):
        super().populate_modules()
        # Replace the DefaultStrategy set up by the parent with MCMCStrategy.
        self.strategy = MCMCStrategy(
            cap_max=self.config.cap_max,
            noise_lr=self.config.mcmc_noise_lr,
            refine_start_iter=self.config.warmup_length,
            refine_stop_iter=self.config.stop_split_at,
            refine_every=self.config.refine_every,
            min_opacity=self.config.mcmc_min_opacity,
            verbose=True,
        )
        self.strategy.absgrad = False  # MCMCStrategy has no absgrad field; get_outputs() reads this.
        self.strategy_state = self.strategy.initialize_state()

    def step_post_backward(self, step):
        assert step == self.step
        # self.optimizers is already the raw {name: torch.optim.Optimizer} dict
        # (SplatfactoModel.get_training_callbacks sets self.optimizers = optimizers.optimizers),
        # matching what gsplat's strategy API expects -- not nerfstudio's Optimizers wrapper.
        means_lr = self.optimizers["means"].param_groups[0]["lr"]
        self.strategy.step_post_backward(
            params=self.gauss_params,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=self.step,
            info=self.info,
            lr=means_lr,
        )
