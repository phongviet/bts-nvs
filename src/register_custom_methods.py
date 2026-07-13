"""Registers custom `ns-train` methods via nerfstudio's NERFSTUDIO_METHOD_CONFIGS
env-var plugin mechanism (no pip install needed).

Methods (comma-separate multiple entries in the env var):
  splatfacto-mcmc             gsplat MCMCStrategy densifier (cap_max sweep, exp006)
  splatfacto-perceptual       + LPIPS(VGG) loss term (exp009 fine-tune)
  splatfacto-mcmc-perceptual  both of the above (exp009 on an MCMC checkpoint)
  splatfacto-tpw              test-pose-weighted train sampling (exp013)

Usage:
    export PYTHONPATH="$PWD:$PYTHONPATH"
    export NERFSTUDIO_METHOD_CONFIGS="\
splatfacto-mcmc=src.register_custom_methods:splatfacto_mcmc_method,\
splatfacto-perceptual=src.register_custom_methods:splatfacto_perceptual_method,\
splatfacto-mcmc-perceptual=src.register_custom_methods:splatfacto_mcmc_perceptual_method,\
splatfacto-tpw=src.register_custom_methods:splatfacto_tpw_method"
    ns-train splatfacto-mcmc --data <staging_dir> ... colmap --eval-mode all --colmap-path sparse/0

Each method clones nerfstudio's method_configs["splatfacto"] TrainerConfig
(same optimizers/schedulers/eval cadence) and only swaps the model config
(and, for -tpw, the datamanager config).
"""
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.models.splatfacto import SplatfactoModelConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from src.data.weighted_datamanager import WeightedFullImageDatamanagerConfig
from src.models.splatfacto_mcmc import SplatfactoMCMCModelConfig
from src.models.splatfacto_perceptual import (
    SplatfactoMCMCPerceptualModelConfig,
    SplatfactoPerceptualModelConfig,
)
from src.models.splatfacto_periphery import SplatfactoPeripheryModelConfig
from src.models.splatfacto_legs import SplatfactoLEGSModelConfig
from src.models.splatfacto_fregs import SplatfactoFreGSModelConfig


def make_trainer_config(method_name: str, model_config, datamanager_config=None) -> TrainerConfig:
    if datamanager_config is None:
        datamanager_config = FullImageDatamanagerConfig(
            dataparser=NerfstudioDataParserConfig(load_3D_points=True),
            cache_images_type="uint8",
        )
    return TrainerConfig(
        method_name=method_name,
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=datamanager_config,
            model=model_config,
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1.6e-6, max_steps=30000),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
            "bilateral_grid": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    )


splatfacto_mcmc_method = MethodSpecification(
    config=make_trainer_config("splatfacto-mcmc", SplatfactoMCMCModelConfig()),
    description="Splatfacto with gsplat MCMCStrategy densification (cap_max sweep) instead of DefaultStrategy.",
)

splatfacto_perceptual_method = MethodSpecification(
    config=make_trainer_config("splatfacto-perceptual", SplatfactoPerceptualModelConfig()),
    description="Splatfacto + LPIPS(VGG) loss term for the exp009 perceptual fine-tune.",
)

splatfacto_mcmc_perceptual_method = MethodSpecification(
    config=make_trainer_config("splatfacto-mcmc-perceptual", SplatfactoMCMCPerceptualModelConfig()),
    description="Splatfacto MCMC densifier + LPIPS(VGG) loss term (exp009 on an MCMC checkpoint).",
)

splatfacto_tpw_method = MethodSpecification(
    config=make_trainer_config(
        "splatfacto-tpw",
        SplatfactoModelConfig(),
        datamanager_config=WeightedFullImageDatamanagerConfig(
            dataparser=NerfstudioDataParserConfig(load_3D_points=True),
            cache_images_type="uint8",
        ),
    ),
    description="Splatfacto with test-pose-weighted train-image sampling (exp013).",
)

splatfacto_prw_method = MethodSpecification(
    config=make_trainer_config("splatfacto-prw", SplatfactoPeripheryModelConfig()),
    description="Splatfacto with periphery-region-weighted (top-biased) L1 loss (exp022).",
)

splatfacto_legs_method = MethodSpecification(
    config=make_trainer_config("splatfacto-legs", SplatfactoLEGSModelConfig()),
    description="Splatfacto with LEGS Laplacian-weighted L1 loss (exp023, arXiv:2606.07932).",
)

splatfacto_fregs_method = MethodSpecification(
    config=make_trainer_config("splatfacto-fregs", SplatfactoFreGSModelConfig()),
    description="Splatfacto with FreGS progressive frequency-regularization loss (exp024, arXiv:2403.06908).",
)
