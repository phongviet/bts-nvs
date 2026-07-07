"""Registers `splatfacto-mcmc` as an `ns-train` method via nerfstudio's
NERFSTUDIO_METHOD_CONFIGS env-var plugin mechanism (no pip install needed).

Usage (see scripts/run_backend_ablation_hcm0034.sh for the wired-up example):
    export PYTHONPATH="$PWD:$PYTHONPATH"
    export NERFSTUDIO_METHOD_CONFIGS="splatfacto-mcmc=src.register_custom_methods:splatfacto_mcmc_method"
    ns-train splatfacto-mcmc --data <staging_dir> --output-dir <out> \
        --pipeline.model.cap-max 1000000 \
        colmap --eval-mode all --colmap-path sparse/0

This clones nerfstudio's own method_configs["splatfacto"] TrainerConfig
(same optimizers/schedulers/eval cadence) and only swaps the model config to
SplatfactoMCMCModelConfig (src/models/splatfacto_mcmc.py), which replaces
gsplat's DefaultStrategy densifier with MCMCStrategy -- see that file's
docstring for exactly what differs between the two strategies.
"""
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from src.models.splatfacto_mcmc import SplatfactoMCMCModelConfig

splatfacto_mcmc_config = TrainerConfig(
    method_name="splatfacto-mcmc",
    steps_per_eval_image=100,
    steps_per_eval_batch=0,
    steps_per_save=2000,
    steps_per_eval_all_images=1000,
    max_num_iterations=30000,
    mixed_precision=False,
    pipeline=VanillaPipelineConfig(
        datamanager=FullImageDatamanagerConfig(
            dataparser=NerfstudioDataParserConfig(load_3D_points=True),
            cache_images_type="uint8",
        ),
        model=SplatfactoMCMCModelConfig(),
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
    config=splatfacto_mcmc_config,
    description="Splatfacto with gsplat MCMCStrategy densification (cap_max sweep) instead of DefaultStrategy.",
)
