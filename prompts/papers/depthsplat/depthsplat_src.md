# prompts/papers/depthsplat/src/config.py

``` py
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Type, TypeVar

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

from .dataset.data_module import DataLoaderCfg, DatasetCfg
from .loss import LossCfgWrapper
from .model.decoder import DecoderCfg
from .model.encoder import EncoderCfg
from .model.model_wrapper import OptimizerCfg, TestCfg, TrainCfg


@dataclass
class CheckpointingCfg:
    load: Optional[str]  # Not a path, since it could be something like wandb://...
    every_n_train_steps: int
    save_top_k: int
    pretrained_model: Optional[str]
    pretrained_monodepth: Optional[str]
    pretrained_mvdepth: Optional[str]
    pretrained_depth: Optional[str]
    no_strict_load: bool
    resume: bool


@dataclass
class ModelCfg:
    decoder: DecoderCfg
    encoder: EncoderCfg


@dataclass
class TrainerCfg:
    max_steps: int
    val_check_interval: int | float | None
    gradient_clip_val: int | float | None
    num_sanity_val_steps: int
    num_nodes: int


@dataclass
class RootCfg:
    wandb: dict
    mode: Literal["train", "test"]
    dataset: DatasetCfg
    data_loader: DataLoaderCfg
    model: ModelCfg
    optimizer: OptimizerCfg
    checkpointing: CheckpointingCfg
    trainer: TrainerCfg
    loss: list[LossCfgWrapper]
    test: TestCfg
    train: TrainCfg
    seed: int
    use_plugins: bool


TYPE_HOOKS = {
    Path: Path,
}


T = TypeVar("T")


def load_typed_config(
    cfg: DictConfig,
    data_class: Type[T],
    extra_type_hooks: dict = {},
) -> T:
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg),
        config=Config(type_hooks={**TYPE_HOOKS, **extra_type_hooks}),
    )


def separate_loss_cfg_wrappers(joined: dict) -> list[LossCfgWrapper]:
    # The dummy allows the union to be converted.
    @dataclass
    class Dummy:
        dummy: LossCfgWrapper

    return [
        load_typed_config(DictConfig({"dummy": {k: v}}), Dummy).dummy
        for k, v in joined.items()
    ]


def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    return load_typed_config(
        cfg,
        RootCfg,
        {list[LossCfgWrapper]: separate_loss_cfg_wrappers},
    )


```

# prompts/papers/depthsplat/src/dataset/__init__.py

``` py
from torch.utils.data import Dataset

from ..misc.step_tracker import StepTracker
from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg
from .dataset_dl3dv import DatasetDL3DV, DatasetDL3DVCfg
from .types import Stage
from .view_sampler import get_view_sampler

DATASETS: dict[str, Dataset] = {
    "re10k": DatasetRE10k,
    "dl3dv": DatasetDL3DV,
}


DatasetCfg = DatasetRE10kCfg | DatasetDL3DVCfg


def get_dataset(
    cfg: DatasetCfg,
    stage: Stage,
    step_tracker: StepTracker | None,
) -> Dataset:
    view_sampler = get_view_sampler(
        cfg.view_sampler,
        stage,
        cfg.overfit_to_scene is not None,
        cfg.cameras_are_circular,
        step_tracker,
    )
    return DATASETS[cfg.name](cfg, stage, view_sampler)


```

# prompts/papers/depthsplat/src/dataset/data_module.py

``` py
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch import Generator, nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

from ..misc.step_tracker import StepTracker
from . import DatasetCfg, get_dataset
from .types import DataShim, Stage
from .validation_wrapper import ValidationWrapper


def get_data_shim(encoder: nn.Module) -> DataShim:
    """Get functions that modify the batch. It's sometimes necessary to modify batches
    outside the data loader because GPU computations are required to modify the batch or
    because the modification depends on something outside the data loader.
    """

    shims: list[DataShim] = []
    if hasattr(encoder, "get_data_shim"):
        shims.append(encoder.get_data_shim())

    def combined_shim(batch):
        for shim in shims:
            batch = shim(batch)
        return batch

    return combined_shim


@dataclass
class DataLoaderStageCfg:
    batch_size: int
    num_workers: int
    persistent_workers: bool
    seed: int | None


@dataclass
class DataLoaderCfg:
    train: DataLoaderStageCfg
    test: DataLoaderStageCfg
    val: DataLoaderStageCfg


DatasetShim = Callable[[Dataset, Stage], Dataset]


def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


class DataModule(LightningDataModule):
    dataset_cfg: DatasetCfg
    data_loader_cfg: DataLoaderCfg
    step_tracker: StepTracker | None
    dataset_shim: DatasetShim
    global_rank: int

    def __init__(
        self,
        dataset_cfg: DatasetCfg,
        data_loader_cfg: DataLoaderCfg,
        step_tracker: StepTracker | None = None,
        dataset_shim: DatasetShim = lambda dataset, _: dataset,
        global_rank: int = 0,
    ) -> None:
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.data_loader_cfg = data_loader_cfg
        self.step_tracker = step_tracker
        self.dataset_shim = dataset_shim
        self.global_rank = global_rank

    def get_persistent(self, loader_cfg: DataLoaderStageCfg) -> bool | None:
        return None if loader_cfg.num_workers == 0 else loader_cfg.persistent_workers

    def get_generator(self, loader_cfg: DataLoaderStageCfg) -> torch.Generator | None:
        if loader_cfg.seed is None:
            return None
        generator = Generator()
        generator.manual_seed(loader_cfg.seed + self.global_rank)
        return generator

    def train_dataloader(self):
        dataset = get_dataset(self.dataset_cfg, "train", self.step_tracker)
        dataset = self.dataset_shim(dataset, "train")
        return DataLoader(
            dataset,
            self.data_loader_cfg.train.batch_size,
            shuffle=not isinstance(dataset, IterableDataset),
            num_workers=self.data_loader_cfg.train.num_workers,
            generator=self.get_generator(self.data_loader_cfg.train),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.train),
        )

    def val_dataloader(self):
        dataset = get_dataset(self.dataset_cfg, "val", self.step_tracker)
        dataset = self.dataset_shim(dataset, "val")
        return DataLoader(
            ValidationWrapper(dataset, 1),
            self.data_loader_cfg.val.batch_size,
            num_workers=self.data_loader_cfg.val.num_workers,
            generator=self.get_generator(self.data_loader_cfg.val),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.val),
        )

    def test_dataloader(self, dataset_cfg=None):
        dataset = get_dataset(
            self.dataset_cfg if dataset_cfg is None else dataset_cfg,
            "test",
            self.step_tracker,
        )
        dataset = self.dataset_shim(dataset, "test")
        return DataLoader(
            dataset,
            self.data_loader_cfg.test.batch_size,
            num_workers=self.data_loader_cfg.test.num_workers,
            generator=self.get_generator(self.data_loader_cfg.test),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.test),
            shuffle=False,
        )


```

# prompts/papers/depthsplat/src/dataset/dataset.py

``` py
from dataclasses import dataclass

from .view_sampler import ViewSamplerCfg


@dataclass
class DatasetCfgCommon:
    image_shape: list[int]
    background_color: list[float]
    cameras_are_circular: bool
    overfit_to_scene: str | None
    view_sampler: ViewSamplerCfg


```

# prompts/papers/depthsplat/src/dataset/dataset_dl3dv.py

``` py
import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset
import numpy as np

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetDL3DVCfg(DatasetCfgCommon):
    name: Literal["dl3dv"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    train_times_per_scene: int
    test_times_per_scene: int
    ori_image_shape: list[int]
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    no_mix_test_set: bool = True
    load_depth: bool = False
    min_views: int = 0
    max_views: int = 0
    highres: bool = False
    sort_target_index: Optional[bool] = False
    overfit_max_views: Optional[int] = None
    sort_context_index: Optional[bool] = False
    use_index_to_load_chunk: Optional[bool] = False


class DatasetDL3DV(IterableDataset):
    cfg: DatasetDL3DVCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetDL3DVCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        # Collect chunks.
        self.chunks = []
        for i, root in enumerate(cfg.roots):
            root = root / self.data_stage
            if self.cfg.use_index_to_load_chunk:
                with open(root / "index.json", "r") as f:
                    json_dict = json.load(f)
                root_chunks = sorted(list(set(json_dict.values())))
            else:
                root_chunks = sorted(
                    [path for path in root.iterdir() if path.suffix == ".torch"]
                )

            self.chunks.extend(root_chunks)
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)
        if self.stage == "test":
            # fast testing
            self.chunks = self.chunks[:: cfg.test_chunk_interval]
        if self.stage == "val":
            self.chunks = self.chunks * int(1e6 // len(self.chunks))

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:
            # Load the chunk.
            chunk = torch.load(chunk_path)

            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"]
                        == self.cfg.overfit_to_scene]
                assert len(item) == 1
                if self.stage == "test":
                    chunk = item
                else:
                    chunk = item * len(chunk)

            if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
                chunk = self.shuffle(chunk)

            times_per_scene = (
                self.cfg.test_times_per_scene
                if self.stage == "test"
                else self.cfg.train_times_per_scene
            )

            for run_idx in range(int(times_per_scene * len(chunk))):
                example = chunk[run_idx // times_per_scene]

                extrinsics, intrinsics = self.convert_poses(example["cameras"])

                scene = example["key"]

                try:
                    extra_kwargs = {}
                    if self.cfg.overfit_to_scene is not None and self.stage != "test":
                        extra_kwargs.update(
                            {
                                "max_num_views": (
                                    148
                                    if self.cfg.overfit_max_views is None
                                    else self.cfg.overfit_max_views
                                )
                            }
                        )

                    out_data = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics,
                        min_context_views=self.cfg.min_views,
                        max_context_views=self.cfg.max_views,
                        **extra_kwargs,
                    )
                    if isinstance(out_data, tuple):
                        context_indices, target_indices = out_data[:2]
                        c_list = [
                            (
                                context_indices.sort()[0]
                                if self.cfg.sort_context_index
                                else context_indices
                            )
                        ]
                        t_list = [
                            (
                                target_indices.sort()[0]
                                if self.cfg.sort_target_index
                                else target_indices
                            )
                        ]
                    if isinstance(out_data, list):
                        c_list = [
                            (
                                a.context.sort()[0]
                                if self.cfg.sort_context_index
                                else a.context
                            )
                            for a in out_data
                        ]
                        t_list = [
                            (
                                a.target.sort()[0]
                                if self.cfg.sort_target_index
                                else a.target
                            )
                            for a in out_data
                        ]

                except ValueError:
                    # Skip because the example doesn't have enough frames.
                    continue

                # Skip the example if the field of view is too wide.
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    continue

                for context_indices, target_indices in zip(c_list, t_list):
                    # Load the images.
                    context_images = [
                        example["images"][index.item()] for index in context_indices
                    ]

                    try:
                        context_images = self.convert_images(context_images)
                    except OSError:
                        # some data might be corrupted
                        continue

                    target_images = [
                        example["images"][index.item()] for index in target_indices
                    ]

                    try:
                        target_images = self.convert_images(target_images)
                    except OSError:
                        # some data might be corrupted
                        continue

                    # Skip the example if the images don't have the right shape.
                    expected_shape = tuple(
                        [3, *self.cfg.ori_image_shape]
                    )  # (3, 270, 480) or (3, 540, 960)

                    context_image_invalid = context_images.shape[1:] != expected_shape
                    target_image_invalid = target_images.shape[1:] != expected_shape

                    if self.cfg.skip_bad_shape and (
                        context_image_invalid or target_image_invalid
                    ):
                        print(
                            f"Skipped bad example {example['key']}. Context shape was "
                            f"{context_images.shape}, target shape was "
                            f"{target_images.shape}, and expected shape was {expected_shape}"
                        )
                        continue

                    # check the extrinsics
                    if any(torch.isnan(torch.det(extrinsics[context_indices][:, :3, :3]))):
                        # print('invalid extrinsics')
                        continue

                    if any(torch.isnan(torch.det(extrinsics[target_indices][:, :3, :3]))):
                        # print('invalid extrinsics')
                        continue

                    # check the extrinsics: translation could be very large
                    # https://github.com/DL3DV-10K/Dataset/issues/34
                    if (extrinsics[context_indices][:, :3, 3] > 1e3).any():
                        # print('extremely large camera translation')
                        continue

                    if (extrinsics[target_indices][:, :3, 3] > 1e3).any():
                        # print('extremely large camera translation')
                        continue

                    if not torch.allclose(torch.det(extrinsics[context_indices][:, :3, :3]), torch.det(extrinsics[context_indices][:, :3, :3]).new_tensor(1)):
                        # print('invalid extrinsics')
                        continue
                    if not torch.allclose(torch.det(extrinsics[target_indices][:, :3, :3]), torch.det(extrinsics[target_indices][:, :3, :3]).new_tensor(1)):
                        # print('invalid extrinsics')
                        continue

                    nf_scale = 1.0
                    example_out = {
                        "context": {
                            "extrinsics": extrinsics[context_indices],
                            "intrinsics": intrinsics[context_indices],
                            "image": context_images,
                            "near": self.get_bound("near", len(context_indices))
                            / nf_scale,
                            "far": self.get_bound("far", len(context_indices))
                            / nf_scale,
                            "index": context_indices,
                        },
                        "target": {
                            "extrinsics": extrinsics[target_indices],
                            "intrinsics": intrinsics[target_indices],
                            "image": target_images,
                            "near": self.get_bound("near", len(target_indices))
                            / nf_scale,
                            "far": self.get_bound("far", len(target_indices))
                            / nf_scale,
                            "index": target_indices,
                        },
                        "scene": scene,
                    }

                    if self.stage == "train" and self.cfg.augment:
                        example_out = apply_augmentation_shim(example_out)
                    if self.cfg.image_shape == list(context_images.shape[2:]):
                        yield example_out
                    else:
                        yield apply_crop_shim(example_out, tuple(self.cfg.image_shape))

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style C2W matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32),
                     "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for i, root in enumerate(self.cfg.roots):
                if not (root / data_stage).is_dir():
                    continue

                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v)
                         for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        if self.stage in ['train', 'test']:
            return (
                min(
                    len(self.index.keys()) * self.cfg.test_times_per_scene,
                    self.cfg.test_len,
                )
                if self.stage == "test" and self.cfg.test_len > 0
                else len(self.index.keys()) * self.cfg.train_times_per_scene
            )
        else:
            # set a very large value here to ensure the validation keep going
            # and do not exhaust; it will be wrap to length 1 anyway.
            return int(1e10)



```

# prompts/papers/depthsplat/src/dataset/dataset_re10k.py

``` py
import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetRE10kCfg(DatasetCfgCommon):
    name: Literal["re10k"]
    roots: list[Path]
    baseline_epsilon: float
    max_fov: float
    make_baseline_1: bool
    augment: bool
    test_len: int
    test_chunk_interval: int
    skip_bad_shape: bool = True
    near: float = -1.0
    far: float = -1.0
    baseline_scale_bounds: bool = True
    shuffle_val: bool = True
    train_times_per_scene: int = 1
    highres: bool = False
    use_index_to_load_chunk: Optional[bool] = False


class DatasetRE10k(IterableDataset):
    cfg: DatasetRE10kCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetRE10kCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        # Collect chunks.
        self.chunks = []
        for i, root in enumerate(cfg.roots):
            root = root / self.data_stage
            if self.cfg.use_index_to_load_chunk:
                with open(root / "index.json", "r") as f:
                    json_dict = json.load(f)
                root_chunks = sorted(list(set(json_dict.values())))
            else:
                root_chunks = sorted(
                    [path for path in root.iterdir() if path.suffix == ".torch"]
                )

            self.chunks.extend(root_chunks)
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)
        if self.stage == "test":
            # testing on a subset for fast speed
            self.chunks = self.chunks[::cfg.test_chunk_interval]

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:
            # Load the chunk.
            chunk = torch.load(chunk_path)

            if self.cfg.overfit_to_scene is not None:
                item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
                assert len(item) == 1
                chunk = item * len(chunk)

            if self.stage in (("train", "val") if self.cfg.shuffle_val else ("train")):
                chunk = self.shuffle(chunk)

            times_per_scene = (
                1
                if self.stage == "test"
                else self.cfg.train_times_per_scene
            )

            for run_idx in range(int(times_per_scene * len(chunk))):
                example = chunk[run_idx // times_per_scene]
                extrinsics, intrinsics = self.convert_poses(example["cameras"])
                scene = example["key"]

                try:
                    context_indices, target_indices = self.view_sampler.sample(
                        scene,
                        extrinsics,
                        intrinsics,
                    )
                except ValueError:
                    # Skip because the example doesn't have enough frames.
                    continue

                # Skip the example if the field of view is too wide.
                if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                    continue

                # Load the images.
                context_images = [
                    example["images"][index.item()] for index in context_indices
                ]
                context_images = self.convert_images(context_images)
                target_images = [
                    example["images"][index.item()] for index in target_indices
                ]
                target_images = self.convert_images(target_images)

                # Skip the example if the images don't have the right shape.
                if self.cfg.highres:
                    expected_shape = (3, 720, 1280)
                else:
                    expected_shape = (3, 360, 640)
                context_image_invalid = context_images.shape[1:] != expected_shape
                target_image_invalid = target_images.shape[1:] != expected_shape
                if self.cfg.skip_bad_shape and (context_image_invalid or target_image_invalid):
                    print(
                        f"Skipped bad example {example['key']}. Context shape was "
                        f"{context_images.shape} and target shape was "
                        f"{target_images.shape}."
                    )
                    continue

                nf_scale = 1.0
                example = {
                    "context": {
                        "extrinsics": extrinsics[context_indices],
                        "intrinsics": intrinsics[context_indices],
                        "image": context_images,
                        "near": self.get_bound("near", len(context_indices)) / nf_scale,
                        "far": self.get_bound("far", len(context_indices)) / nf_scale,
                        "index": context_indices,
                    },
                    "target": {
                        "extrinsics": extrinsics[target_indices],
                        "intrinsics": intrinsics[target_indices],
                        "image": target_images,
                        "near": self.get_bound("near", len(target_indices)) / nf_scale,
                        "far": self.get_bound("far", len(target_indices)) / nf_scale,
                        "index": target_indices,
                    },
                    "scene": scene,
                }

                if self.stage == "train" and self.cfg.augment:
                    example = apply_augmentation_shim(example)
                yield apply_crop_shim(example, tuple(self.cfg.image_shape))

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style C2W matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)
    
    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for i, root in enumerate(self.cfg.roots):
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    def __len__(self) -> int:
        return (
            min(len(self.index.keys()), self.cfg.test_len)
            if self.stage == "test" and self.cfg.test_len > 0
            else len(self.index.keys()) * self.cfg.train_times_per_scene
        )


```

# prompts/papers/depthsplat/src/dataset/shims/augmentation_shim.py

``` py
import torch
from jaxtyping import Float
from torch import Tensor

from ..types import AnyExample, AnyViews


def reflect_extrinsics(
    extrinsics: Float[Tensor, "*batch 4 4"],
) -> Float[Tensor, "*batch 4 4"]:
    reflect = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    reflect[0, 0] = -1
    return reflect @ extrinsics @ reflect


def reflect_views(views: AnyViews) -> AnyViews:
    return {
        **views,
        "image": views["image"].flip(-1),
        "extrinsics": reflect_extrinsics(views["extrinsics"]),
    }


def apply_augmentation_shim(
    example: AnyExample,
    generator: torch.Generator | None = None,
) -> AnyExample:
    """Randomly augment the training images."""
    # Do not augment with 50% chance.
    if torch.rand(tuple(), generator=generator) < 0.5:
        return example

    return {
        **example,
        "context": reflect_views(example["context"]),
        "target": reflect_views(example["target"]),
    }


```

# prompts/papers/depthsplat/src/dataset/shims/bounds_shim.py

``` py
import torch
from einops import einsum, reduce, repeat
from jaxtyping import Float
from torch import Tensor

from ..types import BatchedExample


def compute_depth_for_disparity(
    extrinsics: Float[Tensor, "batch view 4 4"],
    intrinsics: Float[Tensor, "batch view 3 3"],
    image_shape: tuple[int, int],
    disparity: float,
    delta_min: float = 1e-6,  # This prevents motionless scenes from lacking depth.
) -> Float[Tensor, " batch"]:
    """Compute the depth at which moving the maximum distance between cameras
    corresponds to the specified disparity (in pixels).
    """

    # Use the furthest distance between cameras as the baseline.
    origins = extrinsics[:, :, :3, 3]
    deltas = (origins[:, None, :, :] - origins[:, :, None, :]).norm(dim=-1)
    deltas = deltas.clip(min=delta_min)
    baselines = reduce(deltas, "b v ov -> b", "max")

    # Compute a single pixel's size at depth 1.
    h, w = image_shape
    pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=extrinsics.device)
    pixel_size = einsum(
        intrinsics[..., :2, :2].inverse(), pixel_size, "... i j, j -> ... i"
    )

    # This wouldn't make sense with non-square pixels, but then again, non-square pixels
    # don't make much sense anyway.
    mean_pixel_size = reduce(pixel_size, "b v xy -> b", "mean")

    return baselines / (disparity * mean_pixel_size)


def apply_bounds_shim(
    batch: BatchedExample,
    near_disparity: float,
    far_disparity: float,
) -> BatchedExample:
    """Compute reasonable near and far planes (lower and upper bounds on depth). This
    assumes that all of an example's views are of roughly the same thing.
    """

    context = batch["context"]
    _, cv, _, h, w = context["image"].shape

    # Compute near and far planes using the context views.
    near = compute_depth_for_disparity(
        context["extrinsics"],
        context["intrinsics"],
        (h, w),
        near_disparity,
    )
    far = compute_depth_for_disparity(
        context["extrinsics"],
        context["intrinsics"],
        (h, w),
        far_disparity,
    )

    target = batch["target"]
    _, tv, _, _, _ = target["image"].shape
    return {
        **batch,
        "context": {
            **context,
            "near": repeat(near, "b -> b v", v=cv),
            "far": repeat(far, "b -> b v", v=cv),
        },
        "target": {
            **target,
            "near": repeat(near, "b -> b v", v=tv),
            "far": repeat(far, "b -> b v", v=tv),
        },
    }


```

# prompts/papers/depthsplat/src/dataset/shims/crop_shim.py

``` py
import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from torch import Tensor
import torch.nn.functional as F

from ..types import AnyExample, AnyViews


def rescale(
    image: Float[Tensor, "3 h_in w_in"],
    shape: tuple[int, int],
) -> Float[Tensor, "3 h_out w_out"]:
    h, w = shape
    image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
    image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
    image_new = Image.fromarray(image_new)
    image_new = image_new.resize((w, h), Image.LANCZOS)
    image_new = np.array(image_new) / 255
    image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
    return rearrange(image_new, "h w c -> c h w")


def center_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
    depths: None | Float[Tensor, "*#batch h w"],
) -> (
    tuple[
        Float[Tensor, "*#batch c h_out w_out"],  # updated images
        Float[Tensor, "*#batch 3 3"],  # updated intrinsics
    ]
    | tuple[
        Float[Tensor, "*#batch c h_out w_out"],  # updated images
        Float[Tensor, "*#batch 3 3"],  # updated intrinsics
        Float[Tensor, "*#batch h_out w_out"],  # updated depths
    ]
):
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    images = images[..., :, row : row + h_out, col : col + w_out]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    if depths is not None:
        depths = depths[..., :, row : row + h_out, col : col + w_out]
        return images, intrinsics, depths

    return images, intrinsics


def rescale_and_crop(
    images: Float[Tensor, "*#batch c h w"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
    depths: None | Float[Tensor, "*#batch h w"],
) -> (
    tuple[
        Float[Tensor, "*#batch c h_out w_out"],  # updated images
        Float[Tensor, "*#batch 3 3"],  # updated intrinsics
    ]
    | tuple[
        Float[Tensor, "*#batch c h_out w_out"],  # updated images
        Float[Tensor, "*#batch 3 3"],  # updated intrinsics
        Float[Tensor, "*#batch h_out w_out"],  # updated depths
    ]
):
    *_, h_in, w_in = images.shape
    h_out, w_out = shape
    assert h_out <= h_in and w_out <= w_in

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    # Reshape the images to the correct size. Assume we don't have to worry about
    # changing the intrinsics based on how the images are rounded.
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])
    images = images.reshape(*batch, c, h_scaled, w_scaled)

    # reshape and crop depth as well when available
    if depths is not None:
        depths = F.interpolate(
            depths.unsqueeze(1),
            size=(h_scaled, w_scaled),
            mode="bilinear",
            align_corners=True,
        ).squeeze(1)

    return center_crop(images, intrinsics, shape, depths=depths)


def apply_crop_shim_to_views(views: AnyViews, shape: tuple[int, int]) -> AnyViews:
    images, intrinsics = rescale_and_crop(
        views["image"], views["intrinsics"], shape, depths=None
    )
    return {
        **views,
        "image": images,
        "intrinsics": intrinsics,
    }


def apply_crop_shim(example: AnyExample, shape: tuple[int, int]) -> AnyExample:
    """Crop images in the example."""
    return {
        **example,
        "context": apply_crop_shim_to_views(example["context"], shape),
        "target": apply_crop_shim_to_views(example["target"], shape),
    }


```

# prompts/papers/depthsplat/src/dataset/shims/patch_shim.py

``` py
from ..types import BatchedExample, BatchedViews


def apply_patch_shim_to_views(views: BatchedViews, patch_size: int) -> BatchedViews:
    _, _, _, h, w = views["image"].shape

    # Image size must be even so that naive center-cropping does not cause misalignment.
    assert h % 2 == 0 and w % 2 == 0

    h_new = (h // patch_size) * patch_size
    row = (h - h_new) // 2
    w_new = (w // patch_size) * patch_size
    col = (w - w_new) // 2

    # Center-crop the image.
    image = views["image"][:, :, :, row : row + h_new, col : col + w_new]

    # Adjust the intrinsics to account for the cropping.
    intrinsics = views["intrinsics"].clone()
    intrinsics[:, :, 0, 0] *= w / w_new  # fx
    intrinsics[:, :, 1, 1] *= h / h_new  # fy

    return {
        **views,
        "image": image,
        "intrinsics": intrinsics,
    }


def apply_patch_shim(batch: BatchedExample, patch_size: int) -> BatchedExample:
    """Crop images in the batch so that their dimensions are cleanly divisible by the
    specified patch size.
    """
    return {
        **batch,
        "context": apply_patch_shim_to_views(batch["context"], patch_size),
        "target": apply_patch_shim_to_views(batch["target"], patch_size),
    }


```

# prompts/papers/depthsplat/src/dataset/types.py

``` py
from typing import Callable, Literal, TypedDict

from jaxtyping import Float, Int64
from torch import Tensor

Stage = Literal["train", "val", "test"]


# The following types mainly exist to make type-hinted keys show up in VS Code. Some
# dimensions are annotated as "_" because either:
# 1. They're expected to change as part of a function call (e.g., resizing the dataset).
# 2. They're expected to vary within the same function call (e.g., the number of views,
#    which differs between context and target BatchedViews).


class BatchedViews(TypedDict, total=False):
    extrinsics: Float[Tensor, "batch _ 4 4"]  # batch view 4 4
    intrinsics: Float[Tensor, "batch _ 3 3"]  # batch view 3 3
    image: Float[Tensor, "batch _ _ _ _"]  # batch view channel height width
    near: Float[Tensor, "batch _"]  # batch view
    far: Float[Tensor, "batch _"]  # batch view
    index: Int64[Tensor, "batch _"]  # batch view


class BatchedExample(TypedDict, total=False):
    target: BatchedViews
    context: BatchedViews
    scene: list[str]


class UnbatchedViews(TypedDict, total=False):
    extrinsics: Float[Tensor, "_ 4 4"]
    intrinsics: Float[Tensor, "_ 3 3"]
    image: Float[Tensor, "_ 3 height width"]
    near: Float[Tensor, " _"]
    far: Float[Tensor, " _"]
    index: Int64[Tensor, " _"]


class UnbatchedExample(TypedDict, total=False):
    target: UnbatchedViews
    context: UnbatchedViews
    scene: str


# A data shim modifies the example after it's been returned from the data loader.
DataShim = Callable[[BatchedExample], BatchedExample]

AnyExample = BatchedExample | UnbatchedExample
AnyViews = BatchedViews | UnbatchedViews


```

# prompts/papers/depthsplat/src/dataset/validation_wrapper.py

``` py
from typing import Iterator, Optional

import torch
from torch.utils.data import Dataset, IterableDataset


class ValidationWrapper(Dataset):
    """Wraps a dataset so that PyTorch Lightning's validation step can be turned into a
    visualization step.
    """

    dataset: Dataset
    dataset_iterator: Optional[Iterator]
    length: int

    def __init__(self, dataset: Dataset, length: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.length = length
        self.dataset_iterator = None

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        if isinstance(self.dataset, IterableDataset):
            if self.dataset_iterator is None:
                self.dataset_iterator = iter(self.dataset)
            return next(self.dataset_iterator)

        random_index = torch.randint(0, len(self.dataset), tuple())
        return self.dataset[random_index.item()]


```

# prompts/papers/depthsplat/src/dataset/view_sampler/__init__.py

``` py
from typing import Any

from ...misc.step_tracker import StepTracker
from ..types import Stage
from .view_sampler import ViewSampler
from .view_sampler_all import ViewSamplerAll, ViewSamplerAllCfg
from .view_sampler_arbitrary import ViewSamplerArbitrary, ViewSamplerArbitraryCfg
from .view_sampler_bounded import ViewSamplerBounded, ViewSamplerBoundedCfg
from .view_sampler_evaluation import ViewSamplerEvaluation, ViewSamplerEvaluationCfg
from .view_sampler_bounded_v2 import ViewSamplerBoundedV2, ViewSamplerBoundedV2Cfg


VIEW_SAMPLERS: dict[str, ViewSampler[Any]] = {
    "all": ViewSamplerAll,
    "arbitrary": ViewSamplerArbitrary,
    "bounded": ViewSamplerBounded,
    "evaluation": ViewSamplerEvaluation,
    "boundedv2": ViewSamplerBoundedV2,
}

ViewSamplerCfg = (
    ViewSamplerArbitraryCfg
    | ViewSamplerBoundedCfg
    | ViewSamplerEvaluationCfg
    | ViewSamplerAllCfg
    | ViewSamplerBoundedV2Cfg
)


def get_view_sampler(
    cfg: ViewSamplerCfg,
    stage: Stage,
    overfit: bool,
    cameras_are_circular: bool,
    step_tracker: StepTracker | None,
) -> ViewSampler[Any]:
    return VIEW_SAMPLERS[cfg.name](
        cfg,
        stage,
        overfit,
        cameras_are_circular,
        step_tracker,
    )


```

# prompts/papers/depthsplat/src/dataset/view_sampler/view_sampler.py

``` py
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from ...misc.step_tracker import StepTracker
from ..types import Stage

T = TypeVar("T")


class ViewSampler(ABC, Generic[T]):
    cfg: T
    stage: Stage
    is_overfitting: bool
    cameras_are_circular: bool
    step_tracker: StepTracker | None

    def __init__(
        self,
        cfg: T,
        stage: Stage,
        is_overfitting: bool,
        cameras_are_circular: bool,
        step_tracker: StepTracker | None,
    ) -> None:
        self.cfg = cfg
        self.stage = stage
        self.is_overfitting = is_overfitting
        self.cameras_are_circular = cameras_are_circular
        self.step_tracker = step_tracker

    @abstractmethod
    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        pass

    @property
    @abstractmethod
    def num_target_views(self) -> int:
        pass

    @property
    @abstractmethod
    def num_context_views(self) -> int:
        pass

    @property
    def global_step(self) -> int:
        return 0 if self.step_tracker is None else self.step_tracker.get_step()


```

# prompts/papers/depthsplat/src/dataset/view_sampler/view_sampler_all.py

``` py
from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerAllCfg:
    name: Literal["all"]


class ViewSamplerAll(ViewSampler[ViewSamplerAllCfg]):
    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        v, _, _ = extrinsics.shape
        all_frames = torch.arange(v, device=device)
        return all_frames, all_frames

    @property
    def num_context_views(self) -> int:
        return 0

    @property
    def num_target_views(self) -> int:
        return 0


```

# prompts/papers/depthsplat/src/dataset/view_sampler/view_sampler_arbitrary.py

``` py
from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerArbitraryCfg:
    name: Literal["arbitrary"]
    num_context_views: int
    num_target_views: int
    context_views: list[int] | None
    target_views: list[int] | None


class ViewSamplerArbitrary(ViewSampler[ViewSamplerArbitraryCfg]):
    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        """Arbitrarily sample context and target views."""
        num_views, _, _ = extrinsics.shape

        index_context = torch.randint(
            0,
            num_views,
            size=(self.cfg.num_context_views,),
            device=device,
        )

        # Allow the context views to be fixed.
        if self.cfg.context_views is not None:
            assert len(self.cfg.context_views) == self.cfg.num_context_views
            index_context = torch.tensor(
                self.cfg.context_views, dtype=torch.int64, device=device
            )

        index_target = torch.randint(
            0,
            num_views,
            size=(self.cfg.num_target_views,),
            device=device,
        )

        # Allow the target views to be fixed.
        if self.cfg.target_views is not None:
            assert len(self.cfg.target_views) == self.cfg.num_target_views
            index_target = torch.tensor(
                self.cfg.target_views, dtype=torch.int64, device=device
            )

        return index_context, index_target

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views


```

# prompts/papers/depthsplat/src/dataset/view_sampler/view_sampler_bounded.py

``` py
from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerBoundedCfg:
    name: Literal["bounded"]
    num_context_views: int
    num_target_views: int
    min_distance_between_context_views: int
    max_distance_between_context_views: int
    min_distance_to_context_views: int
    warm_up_steps: int
    initial_min_distance_between_context_views: int
    initial_max_distance_between_context_views: int


class ViewSamplerBounded(ViewSampler[ViewSamplerBoundedCfg]):
    def schedule(self, initial: int, final: int) -> int:
        fraction = self.global_step / self.cfg.warm_up_steps
        return min(initial + int((final - initial) * fraction), final)

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        min_view_dist: int | None = None,
        max_view_dist: int | None = None,
        **kwargs,
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        num_views, _, _ = extrinsics.shape

        # Compute the context view spacing based on the current global step.
        if self.stage == "test":
            # When testing, always use the full gap.
            max_gap = self.cfg.max_distance_between_context_views
            min_gap = self.cfg.max_distance_between_context_views
        elif self.cfg.warm_up_steps > 0:
            max_gap = self.schedule(
                self.cfg.initial_max_distance_between_context_views,
                self.cfg.max_distance_between_context_views,
            )
            min_gap = self.schedule(
                self.cfg.initial_min_distance_between_context_views,
                self.cfg.min_distance_between_context_views,
            )
        else:
            max_gap = self.cfg.max_distance_between_context_views
            min_gap = self.cfg.min_distance_between_context_views

        # Pick the gap between the context views.
        if not self.cameras_are_circular:
            max_gap = min(num_views - 1, max_gap)
        min_gap = max(2 * self.cfg.min_distance_to_context_views, min_gap)

        # overwrite min_gap and max_gap, useful for mixed dataset training
        # use different view distance for different dataset
        if min_view_dist is not None:
            min_gap = min_view_dist

        if max_view_dist is not None:
            max_gap = max_view_dist

        if max_gap < min_gap:
            raise ValueError("Example does not have enough frames!")
        context_gap = torch.randint(
            min_gap,
            max_gap + 1,
            size=tuple(),
            device=device,
        ).item()

        # Pick the left and right context indices.
        index_context_left = torch.randint(
            num_views if self.cameras_are_circular else num_views - context_gap,
            size=tuple(),
            device=device,
        ).item()
        if self.stage == "test":
            index_context_left = index_context_left * 0
        index_context_right = index_context_left + context_gap

        if self.is_overfitting:
            index_context_left *= 0
            index_context_right *= 0
            index_context_right += max_gap

        # Pick the target view indices.
        if self.stage == "test":
            # When testing, pick all.
            index_target = torch.arange(
                index_context_left,
                index_context_right + 1,
                device=device,
            )
        else:
            # When training or validating (visualizing), pick at random.
            index_target = torch.randint(
                index_context_left + self.cfg.min_distance_to_context_views,
                index_context_right + 1 - self.cfg.min_distance_to_context_views,
                size=(self.cfg.num_target_views,),
                device=device,
            )

        # Apply modulo for circular datasets.
        if self.cameras_are_circular:
            index_target %= num_views
            index_context_right %= num_views

        return (
            torch.tensor((index_context_left, index_context_right)),
            index_target,
        )

    @property
    def num_context_views(self) -> int:
        return 2

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views


```

# prompts/papers/depthsplat/src/dataset/view_sampler/view_sampler_bounded_v2.py

``` py
'''
Modifiedy from latentSplat and pixelSplat to handle extrapolate and more context views
'''

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from jaxtyping import Float, Int64
from torch import Tensor
import random

from .view_sampler import ViewSampler


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """

    device = xyz.device
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10

    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    barycenter = torch.sum((xyz), 1)
    barycenter = barycenter / xyz.shape[1]
    barycenter = barycenter.view(B, 1, 3)

    dist = torch.sum((xyz - barycenter) ** 2, -1)
    farthest = torch.max(dist, 1)[1]

    for i in range(npoint):
        # print("The %d farthest pts %s " % (i, farthest))
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


@dataclass
class ViewSamplerBoundedV2Cfg:
    name: Literal["boundedv2"]
    num_context_views: int
    num_target_views: int
    min_distance_between_context_views: int
    max_distance_between_context_views: int
    max_distance_to_context_views: int
    context_gap_warm_up_steps: int
    target_gap_warm_up_steps: int
    initial_min_distance_between_context_views: int
    initial_max_distance_between_context_views: int
    initial_max_distance_to_context_views: int
    extra_views_sampling_strategy: Optional[Literal["random", "farthest_point", "equal"]] = "random"
    target_views_replace_sample: Optional[bool] = True


class ViewSamplerBoundedV2(ViewSampler[ViewSamplerBoundedV2Cfg]):
    def schedule(self, initial: int, final: int, steps: int) -> int:
        fraction = self.global_step / steps
        return min(initial + int((final - initial) * fraction), final)

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        max_num_views: Optional[int] = None,
        min_context_views: int = 0,
        max_context_views: int = 0,
        min_view_dist: int | None = None,
        max_view_dist: int | None = None,
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        num_views, _, _ = extrinsics.shape
        if max_num_views is not None:
            num_views = min(num_views, max_num_views)

        if min_context_views > 0 and max_context_views > 0 and self.stage != "test":
            random_num_views = random.randint(min_context_views, max_context_views)
        else:
            random_num_views = None

        # Compute the context view spacing based on the current global step.
        if self.stage == "test":
            # When testing, always use the full gap.
            max_context_gap = self.cfg.max_distance_between_context_views
            min_context_gap = self.cfg.max_distance_between_context_views
        elif self.cfg.context_gap_warm_up_steps > 0:
            max_context_gap = self.schedule(
                self.cfg.initial_max_distance_between_context_views,
                self.cfg.max_distance_between_context_views,
                self.cfg.context_gap_warm_up_steps,
            )
            min_context_gap = self.schedule(
                self.cfg.initial_min_distance_between_context_views,
                self.cfg.min_distance_between_context_views,
                self.cfg.context_gap_warm_up_steps,
            )
        else:
            max_context_gap = self.cfg.max_distance_between_context_views
            min_context_gap = self.cfg.min_distance_between_context_views

        if min_view_dist is not None and max_view_dist is not None:
            # for mixed dataset training, with different sampling distance
            min_context_gap = min_view_dist
            max_context_gap = max_view_dist

        if random_num_views is not None:
            # smaller context gap accordingly
            scale_factor = max(max_context_views // random_num_views, 1)
            max_context_gap = max_context_gap // scale_factor
            min_context_gap = min_context_gap // scale_factor

        if not self.cameras_are_circular:
            max_context_gap = min(
                num_views - 1, max_context_gap
            )

        # Compute the margin from context window to target window based on the current global step
        if self.stage != "test" and self.cfg.target_gap_warm_up_steps > 0:
            max_target_gap = self.schedule(
                self.cfg.initial_max_distance_to_context_views,
                self.cfg.max_distance_to_context_views,
                self.cfg.target_gap_warm_up_steps,
            )
        else:
            max_target_gap = self.cfg.max_distance_to_context_views

        # Pick the gap between the context views.
        if max_context_gap < min_context_gap:
            raise ValueError("Example does not have enough frames!")
        context_gap = torch.randint(
            min_context_gap,
            max_context_gap + 1,
            size=tuple(),
            device=device,
        ).item()

        # Pick the left and right context indices.
        index_context_left = torch.randint(
            low=0,
            high=num_views if self.cameras_are_circular else num_views - context_gap,
            size=tuple(),
            device=device,
        ).item()
        if self.stage == "test":
            index_context_left = index_context_left * 0
        index_context_right = index_context_left + context_gap

        index_target_left = index_context_left - max_target_gap
        index_target_right = index_context_right + max_target_gap

        if not self.cameras_are_circular:
            index_target_left = max(0, index_target_left)
            index_target_right = min(num_views - 1, index_target_right)

        # Pick the target view indices.
        if self.stage == "test":
            # When testing, pick all.
            index_target = torch.arange(
                index_target_left,
                index_target_right + 1,
                device=device,
            )
        else:
            # When training or validating (visualizing), pick at random.
            if self.cfg.target_views_replace_sample:
                index_target = torch.randint(
                    index_target_left,
                    index_target_right + 1,
                    size=(self.cfg.num_target_views,),
                    device=device,
                )
            else:  # sample without replacement
                # similarly, ok to index
                index_target_candidates = torch.arange(
                    index_target_left,
                    index_target_right + 1,
                    device=device,
                )
                indices = torch.randperm(index_target_right + 1 - index_target_left, device=device)[
                    : self.cfg.num_target_views
                ]
                index_target = index_target_candidates[indices]

        # Apply modulo for circular datasets.
        if self.cameras_are_circular:
            index_target %= num_views
            index_context_right %= num_views

        # If more than two context views are desired, pick extra context views between
        # the left and right ones.
        if random_num_views is not None:
            total_num_views = random_num_views
        else:
            total_num_views = self.cfg.num_context_views

        if total_num_views > 2:
            num_extra_views = total_num_views - 2
            extra_views = []
            if self.cfg.extra_views_sampling_strategy == 'random':
                while len(set(extra_views)) != num_extra_views:
                    extra_views = torch.randint(
                        index_context_left + 1,
                        index_context_right,
                        (num_extra_views,),
                    ).tolist()
            elif self.cfg.extra_views_sampling_strategy == 'farthest_point':
                context_bounded_index = torch.arange(index_context_left, index_context_right + 1)
                candidate_views_position = extrinsics[context_bounded_index, :3, -1].unsqueeze(0)
                index_context_local = farthest_point_sample(
                    candidate_views_position, total_num_views
                ).squeeze(0)
                # remap context index back to global scene based index
                index_context = context_bounded_index[index_context_local]
                index_context_left = index_context[0].item()
                index_context_right = index_context[-1].item()
                extra_views = index_context[1:-1].tolist()
            elif self.cfg.extra_views_sampling_strategy == 'equal':
                pass

            # sort the index
            extra_views = sorted(extra_views)
        else:
            extra_views = []

        return (
            torch.tensor((index_context_left, *extra_views, index_context_right)),
            index_target,
        )

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views


```

# prompts/papers/depthsplat/src/dataset/view_sampler/view_sampler_evaluation.py

``` py
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from dacite import Config, from_dict
from jaxtyping import Float, Int64
from torch import Tensor

from ...evaluation.evaluation_index_generator import IndexEntry
from ...misc.step_tracker import StepTracker
from ..types import Stage
from .view_sampler import ViewSampler


@dataclass
class ViewSamplerEvaluationCfg:
    name: Literal["evaluation"]
    index_path: Path
    num_context_views: int


class ViewSamplerEvaluation(ViewSampler[ViewSamplerEvaluationCfg]):
    index: dict[str, IndexEntry | None]

    def __init__(
        self,
        cfg: ViewSamplerEvaluationCfg,
        stage: Stage,
        is_overfitting: bool,
        cameras_are_circular: bool,
        step_tracker: StepTracker | None,
    ) -> None:
        super().__init__(cfg, stage, is_overfitting, cameras_are_circular, step_tracker)

        dacite_config = Config(cast=[tuple])
        with cfg.index_path.open("r") as f:
            self.index = {
                k: None if v is None else from_dict(IndexEntry, v, dacite_config)
                for k, v in json.load(f).items()
            }

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        entry = self.index.get(scene)
        if entry is None:
            raise ValueError(f"No indices available for scene {scene}.")
        context_indices = torch.tensor(entry.context, dtype=torch.int64, device=device)
        target_indices = torch.tensor(entry.target, dtype=torch.int64, device=device)
        return context_indices, target_indices

    @property
    def num_context_views(self) -> int:
        return 0

    @property
    def num_target_views(self) -> int:
        return 0


```

# prompts/papers/depthsplat/src/evaluation/evaluation_cfg.py

``` py
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MethodCfg:
    name: str
    key: str
    path: Path


@dataclass
class SceneCfg:
    scene: str
    target_index: int


@dataclass
class EvaluationCfg:
    methods: list[MethodCfg]
    side_by_side_path: Path | None
    animate_side_by_side: bool
    highlighted: list[SceneCfg]


```

# prompts/papers/depthsplat/src/evaluation/evaluation_index_generator.py

``` py
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from einops import rearrange
from pytorch_lightning import LightningModule
from tqdm import tqdm

from ..geometry.epipolar_lines import project_rays
from ..geometry.projection import get_world_rays, sample_image_grid
from ..misc.image_io import save_image
from ..visualization.annotation import add_label
from ..visualization.layout import add_border, hcat


@dataclass
class EvaluationIndexGeneratorCfg:
    num_target_views: int
    min_distance: int
    max_distance: int
    min_overlap: float
    max_overlap: float
    output_path: Path
    save_previews: bool
    seed: int


@dataclass
class IndexEntry:
    context: tuple[int, ...]
    target: tuple[int, ...]


class EvaluationIndexGenerator(LightningModule):
    generator: torch.Generator
    cfg: EvaluationIndexGeneratorCfg
    index: dict[str, IndexEntry | None]

    def __init__(self, cfg: EvaluationIndexGeneratorCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.generator = torch.Generator()
        self.generator.manual_seed(cfg.seed)
        self.index = {}

    def test_step(self, batch, batch_idx):
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1
        extrinsics = batch["target"]["extrinsics"][0]
        intrinsics = batch["target"]["intrinsics"][0]
        scene = batch["scene"][0]

        context_indices = torch.randperm(v, generator=self.generator)
        for context_index in tqdm(context_indices, "Finding context pair"):
            xy, _ = sample_image_grid((h, w), self.device)
            context_origins, context_directions = get_world_rays(
                rearrange(xy, "h w xy -> (h w) xy"),
                extrinsics[context_index],
                intrinsics[context_index],
            )

            # Step away from context view until the minimum overlap threshold is met.
            valid_indices = []
            for step in (1, -1):
                min_distance = self.cfg.min_distance
                max_distance = self.cfg.max_distance
                current_index = context_index + step * min_distance

                while 0 <= current_index.item() < v:
                    # Compute overlap.
                    current_origins, current_directions = get_world_rays(
                        rearrange(xy, "h w xy -> (h w) xy"),
                        extrinsics[current_index],
                        intrinsics[current_index],
                    )
                    projection_onto_current = project_rays(
                        context_origins,
                        context_directions,
                        extrinsics[current_index],
                        intrinsics[current_index],
                    )
                    projection_onto_context = project_rays(
                        current_origins,
                        current_directions,
                        extrinsics[context_index],
                        intrinsics[context_index],
                    )
                    overlap_a = projection_onto_context["overlaps_image"].float().mean()
                    overlap_b = projection_onto_current["overlaps_image"].float().mean()

                    overlap = min(overlap_a, overlap_b)
                    delta = (current_index - context_index).abs()

                    min_overlap = self.cfg.min_overlap
                    max_overlap = self.cfg.max_overlap
                    if min_overlap <= overlap <= max_overlap:
                        valid_indices.append(
                            (current_index.item(), overlap_a, overlap_b)
                        )

                    # Stop once the camera has panned away too much.
                    if overlap < min_overlap or delta > max_distance:
                        break

                    current_index += step

            if valid_indices:
                # Pick a random valid view. Index the resulting views.
                num_options = len(valid_indices)
                chosen = torch.randint(
                    0, num_options, size=tuple(), generator=self.generator
                )
                chosen, overlap_a, overlap_b = valid_indices[chosen]

                context_left = min(chosen, context_index.item())
                context_right = max(chosen, context_index.item())
                delta = context_right - context_left

                # Pick non-repeated random target views.
                while True:
                    target_views = torch.randint(
                        context_left,
                        context_right + 1,
                        (self.cfg.num_target_views,),
                        generator=self.generator,
                    )
                    if (target_views.unique(return_counts=True)[1] == 1).all():
                        break

                target = tuple(sorted(target_views.tolist()))
                self.index[scene] = IndexEntry(
                    context=(context_left, context_right),
                    target=target,
                )

                # Optionally, save a preview.
                if self.cfg.save_previews:
                    preview_path = self.cfg.output_path / "previews"
                    preview_path.mkdir(exist_ok=True, parents=True)
                    a = batch["target"]["image"][0, chosen]
                    a = add_label(a, f"Overlap: {overlap_a * 100:.1f}%")
                    b = batch["target"]["image"][0, context_index]
                    b = add_label(b, f"Overlap: {overlap_b * 100:.1f}%")
                    vis = add_border(add_border(hcat(a, b)), 1, 0)
                    vis = add_label(vis, f"Distance: {delta} frames")
                    save_image(add_border(vis), preview_path / f"{scene}.png")
                break
        else:
            # This happens if no starting frame produces a valid evaluation example.
            self.index[scene] = None

    def save_index(self) -> None:
        self.cfg.output_path.mkdir(exist_ok=True, parents=True)
        with (self.cfg.output_path / "evaluation_index.json").open("w") as f:
            json.dump(
                {k: None if v is None else asdict(v) for k, v in self.index.items()}, f
            )


```

# prompts/papers/depthsplat/src/evaluation/metric_computer.py

``` py
import os
from pathlib import Path

import torch
from pytorch_lightning import LightningModule
from tabulate import tabulate

from ..misc.image_io import load_image, save_image
from ..visualization.annotation import add_label
from ..visualization.layout import add_border, hcat
from .evaluation_cfg import EvaluationCfg
from .metrics import compute_lpips, compute_psnr, compute_ssim


class MetricComputer(LightningModule):
    cfg: EvaluationCfg

    def __init__(self, cfg: EvaluationCfg) -> None:
        super().__init__()
        self.cfg = cfg

    def test_step(self, batch, batch_idx):
        scene = batch["scene"][0]
        b, cv, _, _, _ = batch["context"]["image"].shape
        assert b == 1 and cv == 2
        _, v, _, _, _ = batch["target"]["image"].shape

        # Skip scenes.
        for method in self.cfg.methods:
            if not (method.path / scene).exists():
                print(f'Skipping "{scene}".')
                return

        # Load the images.
        all_images = {}
        try:
            for method in self.cfg.methods:
                images = [
                    load_image(method.path / scene / f"color/{index.item():0>6}.png")
                    for index in batch["target"]["index"][0]
                ]
                all_images[method.key] = torch.stack(images).to(self.device)
        except FileNotFoundError:
            print(f'Skipping "{scene}".')
            return

        # Compute metrics.
        all_metrics = {}
        rgb_gt = batch["target"]["image"][0]
        for key, images in all_images.items():
            all_metrics = {
                **all_metrics,
                f"lpips_{key}": compute_lpips(rgb_gt, images).mean(),
                f"ssim_{key}": compute_ssim(rgb_gt, images).mean(),
                f"psnr_{key}": compute_psnr(rgb_gt, images).mean(),
            }
        self.log_dict(all_metrics)
        self.print_preview_metrics(all_metrics)

        # Skip the rest if no side-by-side is needed.
        if self.cfg.side_by_side_path is None:
            return

        # Create side-by-side.
        scene_key = f"{batch_idx:0>6}_{scene}"
        for i in range(v):
            true_index = batch["target"]["index"][0, i]
            row = [add_label(batch["target"]["image"][0, i], "Ground Truth")]
            for method in self.cfg.methods:
                image = all_images[method.key][i]
                image = add_label(image, method.name)
                row.append(image)
            start_frame = batch["target"]["index"][0, 0]
            end_frame = batch["target"]["index"][0, -1]
            label = f"Scene {batch['scene'][0]} (frames {start_frame} to {end_frame})"
            row = add_border(add_label(hcat(*row), label, font_size=16))
            save_image(
                row,
                self.cfg.side_by_side_path / scene_key / f"{true_index:0>6}.png",
            )

        # Create an animation.
        if self.cfg.animate_side_by_side:
            (self.cfg.side_by_side_path / "videos").mkdir(exist_ok=True, parents=True)
            command = (
                'ffmpeg -y -framerate 30 -pattern_type glob -i "*.png"  -c:v libx264 '
                '-pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"'
            )
            os.system(
                f"cd {self.cfg.side_by_side_path / scene_key} && {command} "
                f"{Path.cwd()}/{self.cfg.side_by_side_path}/videos/{scene_key}.mp4"
            )

    def print_preview_metrics(self, metrics: dict[str, float]) -> None:
        if getattr(self, "running_metrics", None) is None:
            self.running_metrics = metrics
            self.running_metric_steps = 1
        else:
            s = self.running_metric_steps
            self.running_metrics = {
                k: ((s * v) + metrics[k]) / (s + 1)
                for k, v in self.running_metrics.items()
            }
            self.running_metric_steps += 1

        table = []
        for method in self.cfg.methods:
            row = [
                f"{self.running_metrics[f'{metric}_{method.key}']:.3f}"
                for metric in ("psnr", "lpips", "ssim")
            ]
            table.append((method.key, *row))

        table = tabulate(table, ["Method", "PSNR (dB)", "LPIPS", "SSIM"])
        print(table)


```

# prompts/papers/depthsplat/src/evaluation/metrics.py

``` py
from functools import cache

import torch
from einops import reduce
from jaxtyping import Float
from lpips import LPIPS
from skimage.metrics import structural_similarity
from torch import Tensor


@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
    return -10 * mse.log10()


@cache
def get_lpips(device: torch.device) -> LPIPS:
    return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    value = get_lpips(predicted.device).forward(ground_truth, predicted, normalize=True)
    return value[:, 0, 0, 0]


@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "batch channel height width"],
    predicted: Float[Tensor, "batch channel height width"],
) -> Float[Tensor, " batch"]:
    ssim = [
        structural_similarity(
            gt.detach().cpu().numpy(),
            hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for gt, hat in zip(ground_truth, predicted)
    ]
    return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)


```

# prompts/papers/depthsplat/src/geometry/epipolar_lines.py

``` py
import itertools
from typing import Iterable, Literal, Optional, TypedDict

import torch
from einops import einsum, repeat
from jaxtyping import Bool, Float
from torch import Tensor
from torch.utils.data.dataloader import default_collate

from .projection import (
    get_world_rays,
    homogenize_points,
    homogenize_vectors,
    intersect_rays,
    project_camera_space,
)


def _is_in_bounds(
    xy: Float[Tensor, "*batch 2"],
    epsilon: float = 1e-6,
) -> Bool[Tensor, " *batch"]:
    """Check whether the specified XY coordinates are within the normalized image plane,
    which has a range from 0 to 1 in each direction.
    """
    return (xy >= -epsilon).all(dim=-1) & (xy <= 1 + epsilon).all(dim=-1)


def _is_in_front_of_camera(
    xyz: Float[Tensor, "*batch 3"],
    epsilon: float = 1e-6,
) -> Bool[Tensor, " *batch"]:
    """Check whether the specified points in camera space are in front of the camera."""
    return xyz[..., -1] > -epsilon


def _is_positive_t(
    t: Float[Tensor, " *batch"],
    epsilon: float = 1e-6,
) -> Bool[Tensor, " *batch"]:
    """Check whether the specified t value is positive."""
    return t > -epsilon


class PointProjection(TypedDict):
    t: Float[Tensor, " *batch"]  # ray parameter, as in xyz = origin + t * direction
    xy: Float[Tensor, "*batch 2"]  # image-space xy (normalized to 0 to 1)

    # A "valid" projection satisfies two conditions:
    # 1. It is in front of the camera (i.e., its 3D Z coordinate is positive).
    # 2. It is within the image frame (i.e., its 2D coordinates are between 0 and 1).
    valid: Bool[Tensor, " *batch"]


def _intersect_image_coordinate(
    intrinsics: Float[Tensor, "*#batch 3 3"],
    origins: Float[Tensor, "*#batch 3"],
    directions: Float[Tensor, "*#batch 3"],
    dimension: Literal["x", "y"],
    coordinate_value: float,
) -> PointProjection:
    """Compute the intersection of the projection of a camera-space ray with a line
    that's parallel to the image frame, either horizontally or vertically.
    """

    # Define shorthands.
    dim = "xy".index(dimension)
    other_dim = 1 - dim
    fs = intrinsics[..., dim, dim]  # focal length, same coordinate
    fo = intrinsics[..., other_dim, other_dim]  # focal length, other coordinate
    cs = intrinsics[..., dim, 2]  # principal point, same coordinate
    co = intrinsics[..., other_dim, 2]  # principal point, other coordinate
    os = origins[..., dim]  # ray origin, same coordinate
    oo = origins[..., other_dim]  # ray origin, other coordinate
    ds = directions[..., dim]  # ray direction, same coordinate
    do = directions[..., other_dim]  # ray direction, other coordinate
    oz = origins[..., 2]  # ray origin, z coordinate
    dz = directions[..., 2]  # ray direction, z coordinate
    c = (coordinate_value - cs) / fs  # coefficient (computed once and factored out)

    # Compute the value of t at the intersection.
    # Note: Infinite values of t are fine. No need to handle division by zero.
    t_numerator = c * oz - os
    t_denominator = ds - c * dz
    t = t_numerator / t_denominator

    # Compute the value of the other coordinate at the intersection.
    # Note: Infinite coordinate values are fine. No need to handle division by zero.
    coordinate_numerator = fo * (oo * (c * dz - ds) + do * (os - c * oz))
    coordinate_denominator = dz * os - ds * oz
    coordinate_other = co + coordinate_numerator / coordinate_denominator
    coordinate_same = torch.ones_like(coordinate_other) * coordinate_value
    xy = [coordinate_same]
    xy.insert(other_dim, coordinate_other)
    xy = torch.stack(xy, dim=-1)
    xyz = origins + t[..., None] * directions

    # These will all have exactly the same batch shape (no broadcasting necessary). In
    # terms of jaxtyping annotations, they all match *batch, not just *#batch.
    return {
        "t": t,
        "xy": xy,
        "valid": _is_in_bounds(xy) & _is_in_front_of_camera(xyz) & _is_positive_t(t),
    }


def _compare_projections(
    intersections: Iterable[PointProjection],
    reduction: Literal["min", "max"],
) -> PointProjection:
    intersections = {k: v.clone() for k, v in default_collate(intersections).items()}
    t = intersections["t"]
    xy = intersections["xy"]
    valid = intersections["valid"]

    # Make sure out-of-bounds values are not chosen.
    lowest_priority = {
        "min": torch.inf,
        "max": -torch.inf,
    }[reduction]
    t[~valid] = lowest_priority

    # Run the reduction (either t.min() or t.max()).
    reduced, selector = getattr(t, reduction)(dim=0)

    # Index the results.
    return {
        "t": reduced,
        "xy": xy.gather(0, repeat(selector, "... -> () ... xy", xy=2))[0],
        "valid": valid.gather(0, selector[None])[0],
    }


def _compute_point_projection(
    xyz: Float[Tensor, "*#batch 3"],
    t: Float[Tensor, "*#batch"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> PointProjection:
    xy = project_camera_space(xyz, intrinsics)
    return {
        "t": t,
        "xy": xy,
        "valid": _is_in_bounds(xy) & _is_in_front_of_camera(xyz) & _is_positive_t(t),
    }


class RaySegmentProjection(TypedDict):
    t_min: Float[Tensor, " *batch"]  # ray parameter
    t_max: Float[Tensor, " *batch"]  # ray parameter
    xy_min: Float[Tensor, "*batch 2"]  # image-space xy (normalized to 0 to 1)
    xy_max: Float[Tensor, "*batch 2"]  # image-space xy (normalized to 0 to 1)

    # Whether the segment overlaps the image. If not, the above values are meaningless.
    overlaps_image: Bool[Tensor, " *batch"]


def project_rays(
    origins: Float[Tensor, "*#batch 3"],
    directions: Float[Tensor, "*#batch 3"],
    extrinsics: Float[Tensor, "*#batch 4 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    near: Optional[Float[Tensor, "*#batch"]] = None,
    far: Optional[Float[Tensor, "*#batch"]] = None,
    epsilon: float = 1e-6,
) -> RaySegmentProjection:
    # Transform the rays into camera space.
    world_to_cam = torch.linalg.inv(extrinsics)
    origins = homogenize_points(origins)
    origins = einsum(world_to_cam, origins, "... i j, ... j -> ... i")
    directions = homogenize_vectors(directions)
    directions = einsum(world_to_cam, directions, "... i j, ... j -> ... i")
    origins = origins[..., :3]
    directions = directions[..., :3]

    # Compute intersections with the image's frame.
    frame_intersections = (
        _intersect_image_coordinate(intrinsics, origins, directions, "x", 0.0),
        _intersect_image_coordinate(intrinsics, origins, directions, "x", 1.0),
        _intersect_image_coordinate(intrinsics, origins, directions, "y", 0.0),
        _intersect_image_coordinate(intrinsics, origins, directions, "y", 1.0),
    )
    frame_intersection_min = _compare_projections(frame_intersections, "min")
    frame_intersection_max = _compare_projections(frame_intersections, "max")

    if near is None:
        # Compute the ray's projection at zero depth. If an origin's depth (z value) is
        # within epsilon of zero, this can mean one of two things:
        # 1. The origin is at the camera's position. In this case, use the direction
        #    instead (the ray is probably coming from the camera).
        # 2. The origin isn't at the camera's position, and randomly happens to be on
        #    the plane at zero depth. In this case, its projection is outside the image
        #    plane, and is thus marked as invalid.
        origins_for_projection = origins.clone()
        mask_depth_zero = origins_for_projection[..., -1] < epsilon
        mask_at_camera = origins_for_projection.norm(dim=-1) < epsilon
        origins_for_projection[mask_at_camera] = directions[mask_at_camera]
        projection_at_zero = _compute_point_projection(
            origins_for_projection,
            torch.zeros_like(frame_intersection_min["t"]),
            intrinsics,
        )
        projection_at_zero["valid"][mask_depth_zero & ~mask_at_camera] = False
    else:
        # If a near plane is specified, use it instead.
        t_near = near.broadcast_to(frame_intersection_min["t"].shape)
        projection_at_zero = _compute_point_projection(
            origins + near[..., None] * directions,
            t_near,
            intrinsics,
        )

    if far is None:
        # Compute the ray's projection at infinite depth. Using the projection function
        # with directions (vectors) instead of points may seem wonky, but is equivalent
        # to projecting the point at (origins + infinity * directions).
        projection_at_infinity = _compute_point_projection(
            directions,
            torch.ones_like(frame_intersection_min["t"]) * torch.inf,
            intrinsics,
        )
    else:
        # If a far plane is specified, use it instead.
        t_far = far.broadcast_to(frame_intersection_min["t"].shape)
        projection_at_infinity = _compute_point_projection(
            origins + far[..., None] * directions,
            t_far,
            intrinsics,
        )

    # Build the result by handling cases for ray intersection.
    result = {
        "t_min": torch.empty_like(projection_at_zero["t"]),
        "t_max": torch.empty_like(projection_at_infinity["t"]),
        "xy_min": torch.empty_like(projection_at_zero["xy"]),
        "xy_max": torch.empty_like(projection_at_infinity["xy"]),
        "overlaps_image": torch.empty_like(projection_at_zero["valid"]),
    }

    for min_valid, max_valid in itertools.product([True, False], [True, False]):
        min_mask = projection_at_zero["valid"] ^ (not min_valid)
        max_mask = projection_at_infinity["valid"] ^ (not max_valid)
        mask = min_mask & max_mask
        min_value = projection_at_zero if min_valid else frame_intersection_min
        max_value = projection_at_infinity if max_valid else frame_intersection_max
        result["t_min"][mask] = min_value["t"][mask]
        result["t_max"][mask] = max_value["t"][mask]
        result["xy_min"][mask] = min_value["xy"][mask]
        result["xy_max"][mask] = max_value["xy"][mask]
        result["overlaps_image"][mask] = (min_value["valid"] & max_value["valid"])[mask]

    return result


class RaySegmentProjection(TypedDict):
    t_min: Float[Tensor, " *batch"]  # ray parameter
    t_max: Float[Tensor, " *batch"]  # ray parameter
    xy_min: Float[Tensor, "*batch 2"]  # image-space xy (normalized to 0 to 1)
    xy_max: Float[Tensor, "*batch 2"]  # image-space xy (normalized to 0 to 1)

    # Whether the segment overlaps the image. If not, the above values are meaningless.
    overlaps_image: Bool[Tensor, " *batch"]


def lift_to_3d(
    origins: Float[Tensor, "*#batch 3"],
    directions: Float[Tensor, "*#batch 3"],
    xy: Float[Tensor, "*#batch 2"],
    extrinsics: Float[Tensor, "*#batch 4 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch 3"]:
    """Calculate the 3D positions that correspond to the specified 2D points on the
    epipolar lines defined by the origins and directions. The extrinsics and intrinsics
    are for the images the 2D points lie on.
    """

    xy_origins, xy_directions = get_world_rays(xy, extrinsics, intrinsics)
    return intersect_rays(origins, directions, xy_origins, xy_directions)


def get_depth(
    origins: Float[Tensor, "*#batch 3"],
    directions: Float[Tensor, "*#batch 3"],
    xy: Float[Tensor, "*#batch 2"],
    extrinsics: Float[Tensor, "*#batch 4 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, " *batch"]:
    """Calculate the depths that correspond to the specified 2D points on the epipolar
    lines defined by the origins and directions. The extrinsics and intrinsics are for
    the images the 2D points lie on.
    """
    xyz = lift_to_3d(origins, directions, xy, extrinsics, intrinsics)
    return (xyz - origins).norm(dim=-1)


```

# prompts/papers/depthsplat/src/geometry/projection.py

``` py
from math import prod

import torch
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Bool, Float, Int64
from torch import Tensor


def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(
    vectors: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Convert batched vectors (xyz) to (xyz0)."""
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


def transform_rigid(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    transformation: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Apply a rigid-body transformation to points or vectors."""
    return einsum(transformation, homogeneous_coordinates, "... i j, ... j -> ... i")


def transform_cam2world(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Transform points from 3D camera coordinates to 3D world coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics)


def transform_world2cam(
    homogeneous_coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim dim"],
) -> Float[Tensor, "*batch dim"]:
    """Transform points from 3D world coordinates to 3D camera coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics.inverse())


def project_camera_space(
    points: Float[Tensor, "*#batch dim"],
    intrinsics: Float[Tensor, "*#batch dim dim"],
    epsilon: float = torch.finfo(torch.float32).eps,
    infinity: float = 1e8,
) -> Float[Tensor, "*batch dim-1"]:
    points = points / (points[..., -1:] + epsilon)
    points = points.nan_to_num(posinf=infinity, neginf=-infinity)
    points = einsum(intrinsics, points, "... i j, ... j -> ... i")
    return points[..., :-1]


def project(
    points: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
    intrinsics: Float[Tensor, "*#batch dim dim"],
    epsilon: float = torch.finfo(torch.float32).eps,
) -> tuple[
    Float[Tensor, "*batch dim-1"],  # xy coordinates
    Bool[Tensor, " *batch"],  # whether points are in front of the camera
]:
    points = homogenize_points(points)
    points = transform_world2cam(points, extrinsics)[..., :-1]
    in_front_of_camera = points[..., -1] >= 0
    return project_camera_space(points, intrinsics, epsilon=epsilon), in_front_of_camera


def unproject(
    coordinates: Float[Tensor, "*#batch dim"],
    z: Float[Tensor, "*#batch"],
    intrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
) -> Float[Tensor, "*batch dim+1"]:
    """Unproject 2D camera coordinates with the given Z values."""

    # Apply the inverse intrinsics to the coordinates.
    coordinates = homogenize_points(coordinates)
    ray_directions = einsum(
        intrinsics.inverse(), coordinates, "... i j, ... j -> ... i"
    )

    # Apply the supplied depth values.
    return ray_directions * z[..., None]


def get_world_rays(
    coordinates: Float[Tensor, "*#batch dim"],
    extrinsics: Float[Tensor, "*#batch dim+2 dim+2"],
    intrinsics: Float[Tensor, "*#batch dim+1 dim+1"],
) -> tuple[
    Float[Tensor, "*batch dim+1"],  # origins
    Float[Tensor, "*batch dim+1"],  # directions
]:
    # Get camera-space ray directions.
    directions = unproject(
        coordinates,
        torch.ones_like(coordinates[..., 0]),
        intrinsics,
    )
    directions = directions / directions[..., -1:]

    # Transform ray directions to world coordinates.
    directions = homogenize_vectors(directions)
    directions = transform_cam2world(directions, extrinsics)[..., :-1]

    # Tile the ray origins to have the same shape as the ray directions.
    origins = extrinsics[..., :-1, -1].broadcast_to(directions.shape)

    return origins, directions


def sample_image_grid(
    shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
) -> tuple[
    Float[Tensor, "*shape dim"],  # float coordinates (xy indexing)
    Int64[Tensor, "*shape dim"],  # integer indices (ij indexing)
]:
    """Get normalized (range 0 to 1) coordinates and integer indices for an image."""

    # Each entry is a pixel-wise integer coordinate. In the 2D case, each entry is a
    # (row, col) coordinate.
    indices = [torch.arange(length, device=device) for length in shape]
    stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)

    # Each entry is a floating-point coordinate in the range (0, 1). In the 2D case,
    # each entry is an (x, y) coordinate.
    coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
    coordinates = reversed(coordinates)
    coordinates = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)

    return coordinates, stacked_indices


def sample_training_rays(
    image: Float[Tensor, "batch view channel ..."],
    intrinsics: Float[Tensor, "batch view dim dim"],
    extrinsics: Float[Tensor, "batch view dim+1 dim+1"],
    num_rays: int,
) -> tuple[
    Float[Tensor, "batch ray dim"],  # origins
    Float[Tensor, "batch ray dim"],  # directions
    Float[Tensor, "batch ray 3"],  # sampled color
]:
    device = extrinsics.device
    b, v, _, *grid_shape = image.shape

    # Generate all possible target rays.
    xy, _ = sample_image_grid(tuple(grid_shape), device)
    origins, directions = get_world_rays(
        rearrange(xy, "... d -> ... () () d"),
        extrinsics,
        intrinsics,
    )
    origins = rearrange(origins, "... b v xy -> b (v ...) xy", b=b, v=v)
    directions = rearrange(directions, "... b v xy -> b (v ...) xy", b=b, v=v)
    pixels = rearrange(image, "b v c ... -> b (v ...) c")

    # Sample random rays.
    num_possible_rays = v * prod(grid_shape)
    ray_indices = torch.randint(num_possible_rays, (b, num_rays), device=device)
    batch_indices = repeat(torch.arange(b, device=device), "b -> b n", n=num_rays)

    return (
        origins[batch_indices, ray_indices],
        directions[batch_indices, ray_indices],
        pixels[batch_indices, ray_indices],
    )


def intersect_rays(
    origins_x: Float[Tensor, "*#batch 3"],
    directions_x: Float[Tensor, "*#batch 3"],
    origins_y: Float[Tensor, "*#batch 3"],
    directions_y: Float[Tensor, "*#batch 3"],
    eps: float = 1e-5,
    inf: float = 1e10,
) -> Float[Tensor, "*batch 3"]:
    """Compute the least-squares intersection of rays. Uses the math from here:
    https://math.stackexchange.com/a/1762491/286022
    """

    # Broadcast the rays so their shapes match.
    shape = torch.broadcast_shapes(
        origins_x.shape,
        directions_x.shape,
        origins_y.shape,
        directions_y.shape,
    )
    origins_x = origins_x.broadcast_to(shape)
    directions_x = directions_x.broadcast_to(shape)
    origins_y = origins_y.broadcast_to(shape)
    directions_y = directions_y.broadcast_to(shape)

    # Detect and remove batch elements where the directions are parallel.
    parallel = einsum(directions_x, directions_y, "... xyz, ... xyz -> ...") > 1 - eps
    origins_x = origins_x[~parallel]
    directions_x = directions_x[~parallel]
    origins_y = origins_y[~parallel]
    directions_y = directions_y[~parallel]

    # Stack the rays into (2, *shape).
    origins = torch.stack([origins_x, origins_y], dim=0)
    directions = torch.stack([directions_x, directions_y], dim=0)
    dtype = origins.dtype
    device = origins.device

    # Compute n_i * n_i^T - eye(3) from the equation.
    n = einsum(directions, directions, "r b i, r b j -> r b i j")
    n = n - torch.eye(3, dtype=dtype, device=device).broadcast_to((2, 1, 3, 3))

    # Compute the left-hand side of the equation.
    lhs = reduce(n, "r b i j -> b i j", "sum")

    # Compute the right-hand side of the equation.
    rhs = einsum(n, origins, "r b i j, r b j -> r b i")
    rhs = reduce(rhs, "r b i -> b i", "sum")

    # Left-matrix-multiply both sides by the pseudo-inverse of lhs to find p.
    result = torch.linalg.lstsq(lhs, rhs).solution

    # Handle the case of parallel lines by setting depth to infinity.
    result_all = torch.ones(shape, dtype=dtype, device=device) * inf
    result_all[~parallel] = result
    return result_all


def get_fov(intrinsics: Float[Tensor, "batch 3 3"]) -> Float[Tensor, "batch 2"]:
    intrinsics_inv = intrinsics.inverse()

    def process_vector(vector):
        vector = torch.tensor(vector, dtype=torch.float32, device=intrinsics.device)
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)


```

# prompts/papers/depthsplat/src/global_cfg.py

``` py
from typing import Optional

from omegaconf import DictConfig

cfg: Optional[DictConfig] = None


def get_cfg() -> DictConfig:
    global cfg
    return cfg


def set_cfg(new_cfg: DictConfig) -> None:
    global cfg
    cfg = new_cfg


def get_seed() -> int:
    return cfg.seed


```

# prompts/papers/depthsplat/src/loss/__init__.py

``` py
from .loss import Loss
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper

LOSSES = {
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
}

LossCfgWrapper = LossLpipsCfgWrapper | LossMseCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]


```

# prompts/papers/depthsplat/src/loss/loss.py

``` py
from abc import ABC, abstractmethod
from dataclasses import fields
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians

T_cfg = TypeVar("T_cfg")
T_wrapper = TypeVar("T_wrapper")


class Loss(nn.Module, ABC, Generic[T_cfg, T_wrapper]):
    cfg: T_cfg
    name: str

    def __init__(self, cfg: T_wrapper) -> None:
        super().__init__()

        # Extract the configuration from the wrapper.
        (field,) = fields(type(cfg))
        self.cfg = getattr(cfg, field.name)
        self.name = field.name

    @abstractmethod
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        pass


```

# prompts/papers/depthsplat/src/loss/loss_lpips.py

``` py
from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from lpips import LPIPS
from torch import Tensor

from ..dataset.types import BatchedExample
from ..misc.nn_module_tools import convert_to_buffer
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossLpipsCfg:
    weight: float
    apply_after_step: int


@dataclass
class LossLpipsCfgWrapper:
    lpips: LossLpipsCfg


class LossLpips(Loss[LossLpipsCfg, LossLpipsCfgWrapper]):
    lpips: LPIPS

    def __init__(self, cfg: LossLpipsCfgWrapper) -> None:
        super().__init__(cfg)

        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        valid_depth_mask: Tensor | None
    ) -> Float[Tensor, ""]:
        image = batch["target"]["image"]

        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0, dtype=torch.float32, device=image.device)
        
        if valid_depth_mask is not None and valid_depth_mask.max() > 0.5:
            prediction.color[valid_depth_mask] = 0
            image[valid_depth_mask] = 0

        loss = self.lpips.forward(
            rearrange(prediction.color, "b v c h w -> (b v) c h w"),
            rearrange(image, "b v c h w -> (b v) c h w"),
            normalize=True,
        )
        return self.cfg.weight * loss.mean()


```

# prompts/papers/depthsplat/src/loss/loss_mse.py

``` py
from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
        l1_loss: bool,
        clamp_large_error: float,
        valid_depth_mask: Tensor | None
    ) -> Float[Tensor, ""]:
        delta = prediction.color - batch["target"]["image"]

        if valid_depth_mask is not None and valid_depth_mask.max() > 0.5 and valid_depth_mask.min() < 0.5:
            delta = delta[~valid_depth_mask]

        if clamp_large_error > 0:
            valid_mask = (delta ** 2) < clamp_large_error
            delta = delta[valid_mask]

        if l1_loss:
            return self.cfg.weight * (delta.abs()).mean()
        return self.cfg.weight * (delta**2).mean()


```

# prompts/papers/depthsplat/src/main.py

``` py
import os
from pathlib import Path
import warnings
import copy

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers.wandb import WandbLogger

from pytorch_lightning.plugins.environments import LightningEnvironment


# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.misc.resume_ckpt import find_latest_ckpt
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    if cfg_dict["mode"] == "train" and cfg_dict["train"]["eval_model_every_n_val"] > 0:
        eval_cfg_dict = copy.deepcopy(cfg_dict)
        dataset_dir = str(cfg_dict["dataset"]["roots"]).lower()
        if "re10k" in dataset_dir:
            eval_path = "assets/evaluation_index_re10k.json"
        elif "dl3dv" in dataset_dir:
            if cfg_dict["dataset"]["view_sampler"]["num_context_views"] == 6:
                eval_path = "assets/dl3dv_start_0_distance_50_ctx_6v_tgt_8v.json"
            else:
                raise ValueError("unsupported number of views for dl3dv")
        else:
            raise Exception("Fail to load eval index path")
        eval_cfg_dict["dataset"]["view_sampler"] = {
            "name": "evaluation",
            "index_path": eval_path,
            "num_context_views": cfg_dict["dataset"]["view_sampler"]["num_context_views"],
        }
        eval_cfg = load_typed_root_config(eval_cfg_dict)
    else:
        eval_cfg = None

    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory.
    if cfg_dict.output_dir is None:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
    else:  # for resuming
        output_dir = Path(cfg_dict.output_dir)
        os.makedirs(output_dir, exist_ok=True)
    print(cyan(f"Saving outputs to {output_dir}."))

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled" and cfg.mode == "train":
        wandb_extra_kwargs = {}
        if cfg_dict.wandb.id is not None:
            wandb_extra_kwargs.update({'id': cfg_dict.wandb.id,
                                       'resume': "must"})
        logger = WandbLogger(
            entity=cfg_dict.wandb.entity,
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=os.path.basename(cfg_dict.output_dir),
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            **wandb_extra_kwargs,
        )
        callbacks.append(LearningRateMonitor("step", True))

        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    # Set up checkpointing.
    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            monitor="info/global_step",
            mode="max",
        )
    )
    for cb in callbacks:
        cb.CHECKPOINT_EQUALS_CHAR = '_'

    # Prepare the checkpoint for loading.
    if cfg.checkpointing.resume:
        if not os.path.exists(output_dir / 'checkpoints'):
            checkpoint_path = None
        else:
            checkpoint_path = find_latest_ckpt(output_dir / 'checkpoints')
            print(f'resume from {checkpoint_path}')
    else:
        checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        logger=logger,
        devices=torch.cuda.device_count(),
        strategy='ddp' if torch.cuda.device_count() > 1 else "auto",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=cfg.mode == "test",
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        num_nodes=cfg.trainer.num_nodes,
        plugins=LightningEnvironment() if cfg.use_plugins else None,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)

    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder, cfg.dataset),
        get_losses(cfg.loss),
        step_tracker,
        eval_data_cfg=(
            None if eval_cfg is None else eval_cfg.dataset
        ),
    )
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    if cfg.mode == "train":
        print("train:", len(data_module.train_dataloader()))
        print("val:", len(data_module.val_dataloader()))
        print("test:", len(data_module.test_dataloader()))

    strict_load = not cfg.checkpointing.no_strict_load

    if cfg.mode == "train":
        # only load monodepth
        if cfg.checkpointing.pretrained_monodepth is not None:
            strict_load = False
            pretrained_model = torch.load(cfg.checkpointing.pretrained_monodepth, map_location='cpu')
            if 'state_dict' in pretrained_model:
                pretrained_model = pretrained_model['state_dict']

            model_wrapper.encoder.depth_predictor.load_state_dict(pretrained_model, strict=strict_load)
            print(
                cyan(
                    f"Loaded pretrained monodepth: {cfg.checkpointing.pretrained_monodepth}"
                )
            )

        # load pretrained mvdepth
        if cfg.checkpointing.pretrained_mvdepth is not None:
            pretrained_model = torch.load(cfg.checkpointing.pretrained_mvdepth, map_location='cpu')['model']

            model_wrapper.encoder.depth_predictor.load_state_dict(pretrained_model, strict=False)
            print(
                cyan(
                    f"Loaded pretrained mvdepth: {cfg.checkpointing.pretrained_mvdepth}"
                )
            )
        
        # load full model
        if cfg.checkpointing.pretrained_model is not None:
            pretrained_model = torch.load(cfg.checkpointing.pretrained_model, map_location='cpu')
            if 'state_dict' in pretrained_model:
                pretrained_model = pretrained_model['state_dict']

            model_wrapper.load_state_dict(pretrained_model, strict=strict_load)
            print(
                cyan(
                    f"Loaded pretrained weights: {cfg.checkpointing.pretrained_model}"
                )
            )

        # load pretrained depth
        if cfg.checkpointing.pretrained_depth is not None:
            pretrained_model = torch.load(cfg.checkpointing.pretrained_depth, map_location='cpu')['model']

            strict_load = True
            model_wrapper.encoder.depth_predictor.load_state_dict(pretrained_model, strict=strict_load)
            print(
                cyan(
                    f"Loaded pretrained depth: {cfg.checkpointing.pretrained_depth}"
                )
            )
            
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        # load full model
        if cfg.checkpointing.pretrained_model is not None:
            pretrained_model = torch.load(cfg.checkpointing.pretrained_model, map_location='cpu')
            if 'state_dict' in pretrained_model:
                pretrained_model = pretrained_model['state_dict']

            model_wrapper.load_state_dict(pretrained_model, strict=strict_load)
            print(
                cyan(
                    f"Loaded pretrained weights: {cfg.checkpointing.pretrained_model}"
                )
            )

        # load pretrained depth model only
        if cfg.checkpointing.pretrained_depth is not None:
            pretrained_model = torch.load(cfg.checkpointing.pretrained_depth, map_location='cpu')['model']

            strict_load = True
            model_wrapper.encoder.depth_predictor.load_state_dict(pretrained_model, strict=strict_load)
            print(
                cyan(
                    f"Loaded pretrained depth: {cfg.checkpointing.pretrained_depth}"
                )
            )
            
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision('high')

    train()


```

# prompts/papers/depthsplat/src/misc/LocalLogger.py

``` py
import os
import torch
import numpy as np
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only

LOG_PATH = Path("outputs/local")


class LocalLogger(Logger):
    def __init__(self) -> None:
        super().__init__()
        self.experiment = None
        os.system(f"rm -r {LOG_PATH}")

    @property
    def name(self):
        return "LocalLogger"

    @property
    def version(self):
        return 0

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        pass

    @rank_zero_only
    def log_image(
        self,
        key: str,
        images: list[Any],
        step: Optional[int] = None,
        **kwargs,
    ):
        # The function signature is the same as the wandb logger's, but the step is
        # actually required.
        assert step is not None
        for index, image in enumerate(images):
            path = LOG_PATH / f"{key}/{index:0>2}_{step:0>6}.png"
            path.parent.mkdir(exist_ok=True, parents=True)
            if isinstance(image, torch.Tensor):
                Image.fromarray(image.permute(1, 2, 0).numpy().astype(np.uint8)).save(path)
            else:
                Image.fromarray(image).save(path)


```

# prompts/papers/depthsplat/src/misc/benchmarker.py

``` py
import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from time import time

import numpy as np
import torch


class Benchmarker:
    def __init__(self):
        self.execution_times = defaultdict(list)

    @contextmanager
    def time(self, tag: str, num_calls: int = 1):
        try:
            start_time = time()
            yield
        finally:
            end_time = time()
            for _ in range(num_calls):
                self.execution_times[tag].append((end_time - start_time) / num_calls)

    def dump(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(dict(self.execution_times), f)

    def dump_memory(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(torch.cuda.memory_stats()["allocated_bytes.all.peak"], f)

    def summarize(self) -> None:
        for tag, times in self.execution_times.items():
            print(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")

    def clear_history(self) -> None:
        self.execution_times = defaultdict(list)


```

# prompts/papers/depthsplat/src/misc/collation.py

``` py
from typing import Callable, Dict, Union

from torch import Tensor

Tree = Union[Dict[str, "Tree"], Tensor]


def collate(trees: list[Tree], merge_fn: Callable[[list[Tensor]], Tensor]) -> Tree:
    """Merge nested dictionaries of tensors."""
    if isinstance(trees[0], Tensor):
        return merge_fn(trees)
    else:
        return {
            key: collate([tree[key] for tree in trees], merge_fn) for key in trees[0]
        }


```

# prompts/papers/depthsplat/src/misc/discrete_probability_distribution.py

``` py
import torch
from einops import reduce
from jaxtyping import Float, Int64
from torch import Tensor


def sample_discrete_distribution(
    pdf: Float[Tensor, "*batch bucket"],
    num_samples: int,
    eps: float = torch.finfo(torch.float32).eps,
) -> tuple[
    Int64[Tensor, "*batch sample"],  # index
    Float[Tensor, "*batch sample"],  # probability density
]:
    *batch, bucket = pdf.shape
    normalized_pdf = pdf / (eps + reduce(pdf, "... bucket -> ... ()", "sum"))
    cdf = normalized_pdf.cumsum(dim=-1)
    samples = torch.rand((*batch, num_samples), device=pdf.device)
    index = torch.searchsorted(cdf, samples, right=True).clip(max=bucket - 1)
    return index, normalized_pdf.gather(dim=-1, index=index)


def gather_discrete_topk(
    pdf: Float[Tensor, "*batch bucket"],
    num_samples: int,
    eps: float = torch.finfo(torch.float32).eps,
) -> tuple[
    Int64[Tensor, "*batch sample"],  # index
    Float[Tensor, "*batch sample"],  # probability density
]:
    normalized_pdf = pdf / (eps + reduce(pdf, "... bucket -> ... ()", "sum"))
    index = pdf.topk(k=num_samples, dim=-1).indices
    return index, normalized_pdf.gather(dim=-1, index=index)


```

# prompts/papers/depthsplat/src/misc/heterogeneous_pairings.py

``` py
import torch
from einops import repeat
from jaxtyping import Int
from torch import Tensor

Index = Int[Tensor, "n n-1"]


def generate_heterogeneous_index(
    n: int,
    device: torch.device = torch.device("cpu"),
) -> tuple[Index, Index]:
    """Generate indices for all pairs except self-pairs."""
    arange = torch.arange(n, device=device)

    # Generate an index that represents the item itself.
    index_self = repeat(arange, "h -> h w", w=n - 1)

    # Generate an index that represents the other items.
    index_other = repeat(arange, "w -> h w", h=n).clone()
    index_other += torch.ones((n, n), device=device, dtype=torch.int64).triu()
    index_other = index_other[:, :-1]

    return index_self, index_other


def generate_heterogeneous_index_transpose(
    n: int,
    device: torch.device = torch.device("cpu"),
) -> tuple[Index, Index]:
    """Generate an index that can be used to "transpose" the heterogeneous index.
    Applying the index a second time inverts the "transpose."
    """
    arange = torch.arange(n, device=device)
    ones = torch.ones((n, n), device=device, dtype=torch.int64)

    index_self = repeat(arange, "w -> h w", h=n).clone()
    index_self = index_self + ones.triu()

    index_other = repeat(arange, "h -> h w", w=n)
    index_other = index_other - (1 - ones.triu())

    return index_self[:, :-1], index_other[:, :-1]


```

# prompts/papers/depthsplat/src/misc/image_io.py

``` py
import io
from pathlib import Path
from typing import Union
import skvideo.io

import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]


def fig_to_image(
    fig: Figure,
    dpi: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "3 height width"]:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="raw", dpi=dpi)
    buffer.seek(0)
    data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    h = int(fig.bbox.bounds[3])
    w = int(fig.bbox.bounds[2])
    data = rearrange(data, "(h w c) -> c h w", h=h, w=w, c=4)
    buffer.close()
    return (torch.tensor(data, device=device, dtype=torch.float32) / 255)[:3]


def prep_image(image: FloatImage) -> UInt8[np.ndarray, "height width channel"]:
    # Handle batched images.
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")

    # Handle single-channel images.
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    # Ensure that there are 3 or 4 channels.
    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()


def save_image(
    image: FloatImage,
    path: Union[Path, str],
) -> None:
    """Save an image. Assumed to be in range 0-1."""

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Save the image.
    Image.fromarray(prep_image(image)).save(path)


def load_image(
    path: Union[Path, str],
) -> Float[Tensor, "3 height width"]:
    return tf.ToTensor()(Image.open(path))[:3]


def save_video(
    images: list[FloatImage],
    path: Union[Path, str],
    fps: None | int = None
) -> None:
    """Save an image. Assumed to be in range 0-1."""

    # Create the parent directory if it doesn't already exist.
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)

    # Save the image.
    # Image.fromarray(prep_image(image)).save(path)
    frames = []
    for image in images:
        frames.append(prep_image(image))

    outputdict = {'-pix_fmt': 'yuv420p', '-crf': '23',
                  '-vf': f'setpts=1.*PTS'}
                  
    if fps is not None:
        outputdict.update({'-r': str(fps)})

    writer = skvideo.io.FFmpegWriter(path,
                                     outputdict=outputdict)
    for frame in frames:
        writer.writeFrame(frame)
    writer.close()


```

# prompts/papers/depthsplat/src/misc/nn_module_tools.py

``` py
from torch import nn


def convert_to_buffer(module: nn.Module, persistent: bool = True):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)


```

# prompts/papers/depthsplat/src/misc/render_utils.py

``` py
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/render_utils.py

import numpy as np
import os
import enum
import types
from typing import List, Mapping, Optional, Text, Tuple, Union
import copy
from PIL import Image
# import mediapy as media
from matplotlib import cm
from tqdm import tqdm

import torch


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def pad_poses(p: np.ndarray) -> np.ndarray:
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p: np.ndarray) -> np.ndarray:
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def recenter_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Recenter poses around the origin."""
    cam2world = average_pose(poses)
    transform = np.linalg.inv(pad_poses(cam2world))
    poses = transform @ pad_poses(poses)
    return unpad_poses(poses), transform


def average_pose(poses: np.ndarray) -> np.ndarray:
    """New pose using average position, z-axis, and up vector of input poses."""
    position = poses[:, :3, 3].mean(0)
    z_axis = poses[:, :3, 2].mean(0)
    up = poses[:, :3, 1].mean(0)
    cam2world = viewmatrix(z_axis, up, position)
    return cam2world


def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def transform_poses_pca(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Transforms poses so principal components lie on XYZ axes.

    Args:
      poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

    Returns:
      A tuple (poses, transform), with the transformed poses and the applied
      camera_to_world transforms.
    """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    return poses_recentered, transform
    # points = np.random.rand(3,100)
    # points_h = np.concatenate((points,np.ones_like(points[:1])), axis=0)
    # (poses_recentered @ points_h)[0]
    # (transform @ pad_poses(poses) @ points_h)[0,:3]
    # import pdb; pdb.set_trace()

    # # Just make sure it's it in the [-1, 1]^3 cube
    # scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
    # poses_recentered[:, :3, 3] *= scale_factor
    # transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    # return poses_recentered, transform


def generate_ellipse_path(poses: np.ndarray,
                          n_frames: int = 120,
                          const_speed: bool = True,
                          z_variation: float = 0.,
                          z_phase: float = 0.) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses)
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0], center[1], 0])

    # Calculate scaling for ellipse axes based on input camera positions.
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack([
            low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
            low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
            z_variation * (z_low[2] + (z_high - z_low)[2] *
                           (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
        ], -1)

    theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)

    # if const_speed:

    # # Resample theta angles so that the velocity is closer to constant.
    # lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
    # theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
    # positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([viewmatrix(p - center, up, p) for p in positions])


def generate_path(viewpoint_cameras, n_frames=480):
    c2ws = np.array([np.linalg.inv(np.asarray(
        (cam.world_view_transform.T).cpu().numpy())) for cam in viewpoint_cameras])
    pose = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
    pose_recenter, colmap_to_world_transform = transform_poses_pca(pose)

    # generate new poses
    new_poses = generate_ellipse_path(poses=pose_recenter, n_frames=n_frames)
    # warp back to orignal scale
    new_poses = np.linalg.inv(colmap_to_world_transform) @ pad_poses(new_poses)

    traj = []
    for c2w in new_poses:
        c2w = c2w @ np.diag([1, -1, -1, 1])
        cam = copy.deepcopy(viewpoint_cameras[0])
        cam.image_height = int(cam.image_height / 2) * 2
        cam.image_width = int(cam.image_width / 2) * 2
        cam.world_view_transform = torch.from_numpy(
            np.linalg.inv(c2w).T).float().cuda()
        cam.full_proj_transform = (cam.world_view_transform.unsqueeze(
            0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
        cam.camera_center = cam.world_view_transform.inverse()[3, :3]
        traj.append(cam)

    return traj


def generate_video_render_path(c2ws, n_frames=480):
    # c2ws = np.array([np.linalg.inv(np.asarray(
    #     (cam.world_view_transform.T).cpu().numpy())) for cam in viewpoint_cameras])
    # c2ws: [V, 4, 4]
    pose = c2ws[:, :3, :] @ np.diag([1, -1, -1, 1])
    pose_recenter, colmap_to_world_transform = transform_poses_pca(pose)

    # generate new poses
    new_poses = generate_ellipse_path(poses=pose_recenter, n_frames=n_frames)
    # warp back to orignal scale
    new_poses = np.linalg.inv(colmap_to_world_transform) @ pad_poses(new_poses)

    traj = []
    for c2w in new_poses:
        c2w = c2w @ np.diag([1, -1, -1, 1])
        # cam = copy.deepcopy(viewpoint_cameras[0])
        # cam.image_height = int(cam.image_height / 2) * 2
        # cam.image_width = int(cam.image_width / 2) * 2
        # cam.world_view_transform = torch.from_numpy(
        #     np.linalg.inv(c2w).T).float().cuda()
        # cam.full_proj_transform = (cam.world_view_transform.unsqueeze(
        #     0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
        # cam.camera_center = cam.world_view_transform.inverse()[3, :3]
        # traj.append(cam)
        traj.append(c2w)

    return traj


def load_img(pth: str) -> np.ndarray:
    """Load an image and cast to float32."""
    with open(pth, 'rb') as f:
        image = np.array(Image.open(f), dtype=np.float32)
    return image


def create_videos(base_dir, input_dir, out_name, num_frames=480):
    """Creates videos out of the images saved to disk."""
    # Last two parts of checkpoint path are experiment name and scene name.
    video_prefix = f'{out_name}'

    zpad = max(5, len(str(num_frames - 1)))
    def idx_to_str(idx): return str(idx).zfill(zpad)

    os.makedirs(base_dir, exist_ok=True)
    render_dist_curve_fn = np.log

    # Load one example frame to get image shape and depth range.
    depth_file = os.path.join(input_dir, 'vis', f'depth_{idx_to_str(0)}.tiff')
    depth_frame = load_img(depth_file)
    shape = depth_frame.shape
    p = 3
    distance_limits = np.percentile(depth_frame.flatten(), [p, 100 - p])
    lo, hi = [render_dist_curve_fn(x) for x in distance_limits]
    print(f'Video shape is {shape[:2]}')

    video_kwargs = {
        'shape': shape[:2],
        'codec': 'h264',
        'fps': 60,
        'crf': 18,
    }

    for k in ['depth', 'normal', 'color']:
        video_file = os.path.join(base_dir, f'{video_prefix}_{k}.mp4')
        input_format = 'gray' if k == 'alpha' else 'rgb'

        file_ext = 'png' if k in ['color', 'normal'] else 'tiff'
        idx = 0

        if k == 'color':
            file0 = os.path.join(input_dir, 'renders',
                                 f'{idx_to_str(0)}.{file_ext}')
        else:
            file0 = os.path.join(
                input_dir, 'vis', f'{k}_{idx_to_str(0)}.{file_ext}')

        if not os.path.exists(file0):
            print(f'Images missing for tag {k}')
            continue
        print(f'Making video {video_file}...')
        with media.VideoWriter(
                video_file, **video_kwargs, input_format=input_format) as writer:
            for idx in tqdm(range(num_frames)):
                # img_file = os.path.join(input_dir, f'{k}_{idx_to_str(idx)}.{file_ext}')
                if k == 'color':
                    img_file = os.path.join(
                        input_dir, 'renders', f'{idx_to_str(idx)}.{file_ext}')
                else:
                    img_file = os.path.join(
                        input_dir, 'vis', f'{k}_{idx_to_str(idx)}.{file_ext}')

                if not os.path.exists(img_file):
                    ValueError(f'Image file {img_file} does not exist.')
                img = load_img(img_file)
                if k in ['color', 'normal']:
                    img = img / 255.
                elif k.startswith('depth'):
                    img = render_dist_curve_fn(img)
                    img = np.clip((img - np.minimum(lo, hi)) /
                                  np.abs(hi - lo), 0, 1)
                    img = cm.get_cmap('turbo')(img)[..., :3]

                frame = (np.clip(np.nan_to_num(img), 0., 1.)
                         * 255.).astype(np.uint8)
                writer.add_image(frame)
                idx += 1


def save_img_u8(img, pth):
    """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
    with open(pth, 'wb') as f:
        Image.fromarray(
            (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)).save(
                f, 'PNG')


def save_img_f32(depthmap, pth):
    """Save an image (probably a depthmap) to disk as a float32 TIFF."""
    with open(pth, 'wb') as f:
        Image.fromarray(np.nan_to_num(depthmap).astype(
            np.float32)).save(f, 'TIFF')


```

# prompts/papers/depthsplat/src/misc/resume_ckpt.py

``` py
import os


# Function to extract the step number from the filename
def extract_step(file_name):
    step_str = file_name.split("-")[1].split("_")[1].replace(".ckpt", "")
    return int(step_str)


def find_latest_ckpt(ckpt_dir):
    # List all files in the directory that end with .ckpt
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]

    # Check if there are any .ckpt files in the directory
    if not ckpt_files:
        raise ValueError(f"No .ckpt files found in {ckpt_dir}.")
    else:
        # Find the file with the maximum step
        latest_ckpt_file = max(ckpt_files, key=extract_step)

        return ckpt_dir / latest_ckpt_file


```

# prompts/papers/depthsplat/src/misc/sh_rotation.py

``` py
from math import isqrt

import torch
from e3nn.o3 import matrix_to_angles, wigner_D
from einops import einsum
from jaxtyping import Float
from torch import Tensor


def rotate_sh(
    sh_coefficients: Float[Tensor, "*#batch n"],
    rotations: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch n"]:
    device = sh_coefficients.device
    dtype = sh_coefficients.dtype

    *_, n = sh_coefficients.shape
    alpha, beta, gamma = matrix_to_angles(rotations)
    result = []
    for degree in range(isqrt(n)):
        with torch.device(device):
            sh_rotations = wigner_D(degree, alpha, beta, gamma).type(dtype)
        sh_rotated = einsum(
            sh_rotations,
            sh_coefficients[..., degree**2 : (degree + 1) ** 2],
            "... i j, ... j -> ... i",
        )
        result.append(sh_rotated)

    return torch.cat(result, dim=-1)


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt
    from e3nn.o3 import spherical_harmonics
    from matplotlib import cm
    from scipy.spatial.transform.rotation import Rotation as R

    device = torch.device("cuda")

    # Generate random spherical harmonics coefficients.
    degree = 4
    coefficients = torch.rand((degree + 1) ** 2, dtype=torch.float32, device=device)

    def plot_sh(sh_coefficients, path: Path) -> None:
        phi = torch.linspace(0, torch.pi, 100, device=device)
        theta = torch.linspace(0, 2 * torch.pi, 100, device=device)
        phi, theta = torch.meshgrid(phi, theta, indexing="xy")
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
        xyz = torch.stack([x, y, z], dim=-1)
        sh = spherical_harmonics(list(range(degree + 1)), xyz, True)
        result = einsum(sh, sh_coefficients, "... n, n -> ...")
        result = (result - result.min()) / (result.max() - result.min())

        # Set the aspect ratio to 1 so our sphere looks spherical
        fig = plt.figure(figsize=plt.figaspect(1.0))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            x.cpu().numpy(),
            y.cpu().numpy(),
            z.cpu().numpy(),
            rstride=1,
            cstride=1,
            facecolors=cm.seismic(result.cpu().numpy()),
        )
        # Turn off the axis planes
        ax.set_axis_off()
        path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(path)

    for i, angle in enumerate(torch.linspace(0, 2 * torch.pi, 30)):
        rotation = torch.tensor(
            R.from_euler("x", angle.item()).as_matrix(), device=device
        )
        plot_sh(rotate_sh(coefficients, rotation), Path(f"sh_rotation/{i:0>3}.png"))

    print("Done!")


```

# prompts/papers/depthsplat/src/misc/stablize_camera.py

``` py
"""
https://github.com/google/dynibar/blob/main/ibrnet/data_loaders/llff_data_utils.py
"""

import numpy as np
import cv2


def render_stabilization_path(poses, k_size=45):
    """Rendering stablizaed camera path."""

    # hwf = poses[0, :, 4:5]
    num_frames = poses.shape[0]
    output_poses = []

    input_poses = []

    for i in range(num_frames):
        input_poses.append(
            np.concatenate(
                [poses[i, :3, 0:1], poses[i, :3, 1:2], poses[i, :3, 3:4]], axis=-1
            )
        )

    input_poses = np.array(input_poses)

    gaussian_kernel = cv2.getGaussianKernel(ksize=k_size, sigma=-1)
    output_r1 = cv2.filter2D(input_poses[:, :, 0], -1, gaussian_kernel)
    output_r2 = cv2.filter2D(input_poses[:, :, 1], -1, gaussian_kernel)

    output_r1 = output_r1 / np.linalg.norm(output_r1, axis=-1, keepdims=True)
    output_r2 = output_r2 / np.linalg.norm(output_r2, axis=-1, keepdims=True)

    output_t = cv2.filter2D(input_poses[:, :, 2], -1, gaussian_kernel)

    for i in range(num_frames):
        output_r3 = np.cross(output_r1[i], output_r2[i])

        render_pose = np.concatenate(
            [
                output_r1[i, :, None],
                output_r2[i, :, None],
                output_r3[:, None],
                output_t[i, :, None],
            ],
            axis=-1,
        )

        output_poses.append(render_pose[:3, :])

    return output_poses


```

# prompts/papers/depthsplat/src/misc/step_tracker.py

``` py
from multiprocessing import RLock

import torch
from jaxtyping import Int64
from torch import Tensor
from torch.multiprocessing import Manager


class StepTracker:
    lock: RLock
    step: Int64[Tensor, ""]

    def __init__(self):
        self.lock = Manager().RLock()
        self.step = torch.tensor(0, dtype=torch.int64).share_memory_()

    def set_step(self, step: int) -> None:
        with self.lock:
            self.step.fill_(step)

    def get_step(self) -> int:
        with self.lock:
            return self.step.item()


```

# prompts/papers/depthsplat/src/misc/wandb_tools.py

``` py
from pathlib import Path

import wandb


def version_to_int(artifact) -> int:
    """Convert versions of the form vX to X. For example, v12 to 12."""
    return int(artifact.version[1:])


def download_checkpoint(
    run_id: str,
    download_dir: Path,
    version: str | None,
) -> Path:
    api = wandb.Api()
    run = api.run(run_id)

    # Find the latest saved model checkpoint.
    chosen = None
    for artifact in run.logged_artifacts():
        if artifact.type != "model" or artifact.state != "COMMITTED":
            continue

        # If no version is specified, use the latest.
        if version is None:
            if chosen is None or version_to_int(artifact) > version_to_int(chosen):
                chosen = artifact

        # If a specific verison is specified, look for it.
        elif version == artifact.version:
            chosen = artifact
            break

    # Download the checkpoint.
    download_dir.mkdir(exist_ok=True, parents=True)
    root = download_dir / run_id
    chosen.download(root=root)
    return root / "model.ckpt"


def update_checkpoint_path(path: str | None, wandb_cfg: dict) -> Path | None:
    if path is None:
        return None

    if not str(path).startswith("wandb://"):
        return Path(path)

    run_id, *version = path[len("wandb://") :].split(":")
    if len(version) == 0:
        version = None
    elif len(version) == 1:
        version = version[0]
    else:
        raise ValueError("Invalid version specifier!")

    project = wandb_cfg["project"]
    return download_checkpoint(
        f"{project}/{run_id}",
        Path("checkpoints"),
        version,
    )


```

# prompts/papers/depthsplat/src/model/decoder/__init__.py

``` py
from ...dataset import DatasetCfg
from .decoder import Decoder
from .decoder_splatting_cuda import DecoderSplattingCUDA, DecoderSplattingCUDACfg

DECODERS = {
    "splatting_cuda": DecoderSplattingCUDA,
}

DecoderCfg = DecoderSplattingCUDACfg


def get_decoder(decoder_cfg: DecoderCfg, dataset_cfg: DatasetCfg) -> Decoder:
    return DECODERS[decoder_cfg.name](decoder_cfg, dataset_cfg)


```

# prompts/papers/depthsplat/src/model/decoder/cuda_splatting.py

``` py
from math import isqrt
from typing import Literal

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...geometry.projection import get_fov, homogenize_points


def get_projection_matrix(
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    fov_x: Float[Tensor, " batch"],
    fov_y: Float[Tensor, " batch"],
) -> Float[Tensor, "batch 4 4"]:
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result


def render_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    scale_invariant: bool = True,
    use_sh: bool = True,
) -> Float[Tensor, "batch 3 height width"]:
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    # Make sure everything is in a range where numerical issues don't appear.
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
        gaussian_means = gaussian_means * scale[:, None, None]
        near = near * scale
        far = far * scale

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    b, _, _ = extrinsics.shape
    h, w = image_shape

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        image, radii = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )
        all_images.append(image)
        all_radii.append(radii)
    return torch.stack(all_images)


def render_cuda_orthographic(
    extrinsics: Float[Tensor, "batch 4 4"],
    width: Float[Tensor, " batch"],
    height: Float[Tensor, " batch"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    fov_degrees: float = 0.1,
    use_sh: bool = True,
    dump: dict | None = None,
) -> Float[Tensor, "batch 3 height width"]:
    b, _, _ = extrinsics.shape
    h, w = image_shape
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    # Create fake "orthographic" projection by moving the camera back and picking a
    # small field of view.
    fov_x = torch.tensor(fov_degrees, device=extrinsics.device).deg2rad()
    tan_fov_x = (0.5 * fov_x).tan()
    distance_to_near = (0.5 * width) / tan_fov_x
    tan_fov_y = 0.5 * height / distance_to_near
    fov_y = (2 * tan_fov_y).atan()
    near = near + distance_to_near
    far = far + distance_to_near
    move_back = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    move_back[2, 3] = -distance_to_near
    extrinsics = extrinsics @ move_back

    # Escape hatch for visualization/figures.
    if dump is not None:
        dump["extrinsics"] = extrinsics
        dump["fov_x"] = fov_x
        dump["fov_y"] = fov_y
        dump["near"] = near
        dump["far"] = far

    projection_matrix = get_projection_matrix(
        near, far, repeat(fov_x, "-> b", b=b), fov_y
    )
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x,
            tanfovy=tan_fov_y,
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        image, radii = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )
        all_images.append(image)
        all_radii.append(radii)
    return torch.stack(all_images)


DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]


def render_depth_cuda(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    scale_invariant: bool = True,
    mode: DepthRenderingMode = "depth",
) -> Float[Tensor, "batch height width"]:
    # Specify colors according to Gaussian depths.
    camera_space_gaussians = einsum(
        extrinsics.inverse(), homogenize_points(gaussian_means), "b i j, b g j -> b g i"
    )
    fake_color = camera_space_gaussians[..., 2]

    if mode == "disparity":
        fake_color = 1 / fake_color
    elif mode == "log":
        fake_color = fake_color.minimum(near[:, None]).maximum(far[:, None]).log()

    # Render using depth as color.
    b, _ = fake_color.shape
    result = render_cuda(
        extrinsics,
        intrinsics,
        near,
        far,
        image_shape,
        torch.zeros((b, 3), dtype=fake_color.dtype, device=fake_color.device),
        gaussian_means,
        gaussian_covariances,
        repeat(fake_color, "b g -> b g c ()", c=3),
        gaussian_opacities,
        scale_invariant=scale_invariant,
        use_sh=False,
    )
    return result.mean(dim=1)


```

# prompts/papers/depthsplat/src/model/decoder/decoder.py

``` py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from ...dataset import DatasetCfg
from ..types import Gaussians

DepthRenderingMode = Literal[
    "depth",
    "log",
    "disparity",
    "relative_disparity",
]


@dataclass
class DecoderOutput:
    color: Float[Tensor, "batch view 3 height width"]
    depth: Float[Tensor, "batch view height width"] | None


T = TypeVar("T")


class Decoder(nn.Module, ABC, Generic[T]):
    cfg: T
    dataset_cfg: DatasetCfg

    def __init__(self, cfg: T, dataset_cfg: DatasetCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg

    @abstractmethod
    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
    ) -> DecoderOutput:
        pass


```

# prompts/papers/depthsplat/src/model/decoder/decoder_splatting_cuda.py

``` py
from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...dataset import DatasetCfg
from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda, render_depth_cuda
from .decoder import Decoder, DecoderOutput


@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
        dataset_cfg: DatasetCfg,
    ) -> None:
        super().__init__(cfg, dataset_cfg)
        self.register_buffer(
            "background_color",
            torch.tensor(dataset_cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
    ) -> DecoderOutput:
        b, v, _, _ = extrinsics.shape
        color = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
        )
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)

        return DecoderOutput(
            color,
            None
            if depth_mode is None
            else self.render_depth(
                gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode
            ),
        )

    def render_depth(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        mode: DepthRenderingMode = "depth",
    ) -> Float[Tensor, "batch view height width"]:
        b, v, _, _ = extrinsics.shape
        result = render_depth_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            mode=mode,
        )
        return rearrange(result, "(b v) h w -> b v h w", b=b, v=v)


```

# prompts/papers/depthsplat/src/model/encoder/__init__.py

``` py
from typing import Optional

from .encoder import Encoder
from .encoder_depthsplat import EncoderDepthSplat, EncoderDepthSplatCfg
from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_depthsplat import EncoderVisualizerDepthSplat

ENCODERS = {
    "depthsplat": (EncoderDepthSplat, EncoderVisualizerDepthSplat),
}

EncoderCfg = EncoderDepthSplatCfg


def get_encoder(cfg: EncoderCfg) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer


```

# prompts/papers/depthsplat/src/model/encoder/common/gaussian_adapter.py

``` py
from dataclasses import dataclass

import torch
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn
import torch.nn.functional as F

from ....geometry.projection import get_world_rays
from ....misc.sh_rotation import rotate_sh
from .gaussians import build_covariance


@dataclass
class Gaussians:
    means: Float[Tensor, "*batch 3"]
    covariances: Float[Tensor, "*batch 3 3"]
    scales: Float[Tensor, "*batch 3"]
    rotations: Float[Tensor, "*batch 4"]
    harmonics: Float[Tensor, "*batch 3 _"]
    opacities: Float[Tensor, " *batch"]


@dataclass
class GaussianAdapterCfg:
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int


class GaussianAdapter(nn.Module):
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def forward(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"] | None,
        coordinates: Float[Tensor, "*#batch 2"],
        depths: Float[Tensor, "*#batch"] | None,
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        image_shape: tuple[int, int],
        eps: float = 1e-8,
        point_cloud: Float[Tensor, "*#batch 3"] | None = None,
        input_images: Tensor | None = None,
    ) -> Gaussians:
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)

        scales = torch.clamp(F.softplus(scales - 4.),
            min=self.cfg.gaussian_scale_min,
            max=self.cfg.gaussian_scale_max,
            )

        assert input_images is not None

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        # [2, 2, 65536, 1, 1, 3, 25]
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        if input_images is not None:
            # [B, V, H*W, 1, 1, 3]
            imgs = rearrange(input_images, "b v c h w -> b v (h w) () () c")
            # init sh with input images
            sh[..., 0] = sh[..., 0] + RGB2SH(imgs)

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)

        # Compute Gaussian means.
        origins, directions = get_world_rays(coordinates, extrinsics, intrinsics)
        means = origins + directions * depths[..., None]

        return Gaussians(
            means=means,
            covariances=covariances,
            harmonics=rotate_sh(sh, c2w_rotations[..., None, :, :]),
            opacities=opacities,
            # NOTE: These aren't yet rotated into world space, but they're only used for
            # exporting Gaussians to ply files. This needs to be fixed...
            scales=scales,
            rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
        )

    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:
        xy_multipliers = multiplier * einsum(
            intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh


def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


```

# prompts/papers/depthsplat/src/model/encoder/common/gaussians.py

``` py
import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor


# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(
    quaternions: Float[Tensor, "*batch 4"],
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 3 3"]:
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)


def build_covariance(
    scale: Float[Tensor, "*#batch 3"],
    rotation_xyzw: Float[Tensor, "*#batch 4"],
) -> Float[Tensor, "*batch 3 3"]:
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )


```

# prompts/papers/depthsplat/src/model/encoder/common/sampler.py

``` py
from jaxtyping import Float, Int64, Shaped
from torch import Tensor, nn

from ....misc.discrete_probability_distribution import (
    gather_discrete_topk,
    sample_discrete_distribution,
)


class Sampler(nn.Module):
    def forward(
        self,
        probabilities: Float[Tensor, "*batch bucket"],
        num_samples: int,
        deterministic: bool,
    ) -> tuple[
        Int64[Tensor, "*batch 1"],  # index
        Float[Tensor, "*batch 1"],  # probability density
    ]:
        return (
            gather_discrete_topk(probabilities, num_samples)
            if deterministic
            else sample_discrete_distribution(probabilities, num_samples)
        )

    def gather(
        self,
        index: Int64[Tensor, "*batch sample"],
        target: Shaped[Tensor, "..."],  # *batch bucket *shape
    ) -> Shaped[Tensor, "..."]:  # *batch sample *shape
        """Gather from the target according to the specified index. Handle the
        broadcasting needed for the gather to work. See the comments for the actual
        expected input/output shapes since jaxtyping doesn't support multiple variadic
        lengths in annotations.
        """
        bucket_dim = index.ndim - 1
        while len(index.shape) < len(target.shape):
            index = index[..., None]
        broadcasted_index_shape = list(target.shape)
        broadcasted_index_shape[bucket_dim] = index.shape[bucket_dim]
        index = index.broadcast_to(broadcasted_index_shape)
        return target.gather(dim=bucket_dim, index=index)


```

# prompts/papers/depthsplat/src/model/encoder/encoder.py

``` py
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from torch import nn

from ...dataset.types import BatchedViews, DataShim
from ..types import Gaussians

T = TypeVar("T")


class Encoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        context: BatchedViews,
        deterministic: bool,
    ) -> Gaussians:
        pass

    def get_data_shim(self) -> DataShim:
        """The default shim doesn't modify the batch."""
        return lambda x: x


```

# prompts/papers/depthsplat/src/model/encoder/encoder_depthsplat.py

``` py
from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .encoder import Encoder
from .visualization.encoder_visualizer_depthsplat_cfg import EncoderVisualizerDepthSplatCfg

import torchvision.transforms as T
import torch.nn.functional as F

from .unimatch.mv_unimatch import MultiViewUniMatch
from .unimatch.dpt_head import DPTHead


@dataclass
class EncoderDepthSplatCfg:
    name: Literal["depthsplat"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerDepthSplatCfg
    gaussian_adapter: GaussianAdapterCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]

    # mv_unimatch
    num_scales: int
    upsample_factor: int
    lowest_feature_resolution: int
    depth_unet_channels: int
    grid_sample_disable_cudnn: bool

    # depthsplat color branch
    large_gaussian_head: bool
    color_large_unet: bool
    init_sh_input_img: bool
    feature_upsampler_channels: int
    gaussian_regressor_channels: int

    # loss config
    supervise_intermediate_depth: bool
    return_depth: bool

    # only depth
    train_depth_only: bool

    # monodepth config
    monodepth_vit_type: str

    # multi-view matching
    local_mv_match: int


class EncoderDepthSplat(Encoder[EncoderDepthSplatCfg]):
    def __init__(self, cfg: EncoderDepthSplatCfg) -> None:
        super().__init__(cfg)

        self.depth_predictor = MultiViewUniMatch(
            num_scales=cfg.num_scales,
            upsample_factor=cfg.upsample_factor,
            lowest_feature_resolution=cfg.lowest_feature_resolution,
            vit_type=cfg.monodepth_vit_type,
            unet_channels=cfg.depth_unet_channels,
            grid_sample_disable_cudnn=cfg.grid_sample_disable_cudnn,
        )

        if self.cfg.train_depth_only:
            return

        # upsample features to the original resolution
        model_configs = {
            'vits': {'in_channels': 384, 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'in_channels': 768, 'features': 96, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'in_channels': 1024, 'features': 128, 'out_channels': [128, 256, 512, 1024]},
        }

        self.feature_upsampler = DPTHead(**model_configs[cfg.monodepth_vit_type],
                                        downsample_factor=cfg.upsample_factor,
                                        return_feature=True,
                                        num_scales=cfg.num_scales,
                                        )
        feature_upsampler_channels = model_configs[cfg.monodepth_vit_type]["features"]
        
        # gaussians adapter
        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        # concat(img, depth, match_prob, features)
        in_channels = 3 + 1 + 1 + feature_upsampler_channels
        channels = self.cfg.gaussian_regressor_channels

        # conv regressor
        modules = [
                    nn.Conv2d(in_channels, channels, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, 3, 1, 1),
                ]

        self.gaussian_regressor = nn.Sequential(*modules)

        # predict gaussian parameters: scale, q, sh, offset, opacity
        num_gaussian_parameters = self.gaussian_adapter.d_in + 2 + 1

        # concat(img, features, regressor_out, match_prob)
        in_channels = 3 + feature_upsampler_channels + channels + 1
        self.gaussian_head = nn.Sequential(
                nn.Conv2d(in_channels, num_gaussian_parameters,
                          3, 1, 1, padding_mode='replicate'),
                nn.GELU(),
                nn.Conv2d(num_gaussian_parameters,
                          num_gaussian_parameters, 3, 1, 1, padding_mode='replicate')
            )

        if self.cfg.init_sh_input_img:
            nn.init.zeros_(self.gaussian_head[-1].weight[10:])
            nn.init.zeros_(self.gaussian_head[-1].bias[10:])

        # init scale
        # first 3: opacity, offset_xy
        nn.init.zeros_(self.gaussian_head[-1].weight[3:6])
        nn.init.zeros_(self.gaussian_head[-1].bias[3:6])

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        scene_names: Optional[list] = None,
    ):
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        if v > 3:
            with torch.no_grad():
                xyzs = context["extrinsics"][:, :, :3, -1].detach()
                cameras_dist_matrix = torch.cdist(xyzs, xyzs, p=2)
                cameras_dist_index = torch.argsort(cameras_dist_matrix)

                cameras_dist_index = cameras_dist_index[:, :, :(self.cfg.local_mv_match + 1)]
        else:
            cameras_dist_index = None

        # depth prediction
        results_dict = self.depth_predictor(
            context["image"],
            attn_splits_list=[2],
            min_depth=1. / context["far"],
            max_depth=1. / context["near"],
            intrinsics=context["intrinsics"],
            extrinsics=context["extrinsics"],
            nn_matrix=cameras_dist_index,
        )

        # list of [B, V, H, W], with all the intermediate depths
        depth_preds = results_dict['depth_preds']

        # [B, V, H, W]
        depth = depth_preds[-1]

        if self.cfg.train_depth_only:
            # convert format
            # [B, V, H*W, 1, 1]
            depths = rearrange(depth, "b v h w -> b v (h w) () ()")

            if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
                # supervise all the intermediate depth predictions
                num_depths = len(depth_preds)

                # [B, V, H*W, 1, 1]
                intermediate_depths = torch.cat(
                    depth_preds[:(num_depths - 1)], dim=0)
                intermediate_depths = rearrange(
                    intermediate_depths, "b v h w -> b v (h w) () ()")

                # concat in the batch dim
                depths = torch.cat((intermediate_depths, depths), dim=0)

                b *= num_depths

            # return depth prediction for supervision
            depths = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).squeeze(-1).squeeze(-1)
            # print(depths.shape)  # [B, V, H, W]

            return {
                "gaussians": None,
                "depths": depths
            }

        # features [BV, C, H, W]
        features = self.feature_upsampler(results_dict["features_mono_intermediate"],
                                          cnn_features=results_dict["features_cnn_all_scales"][::-1],
                                          mv_features=results_dict["features_mv"][
                                          0] if self.cfg.num_scales == 1 else results_dict["features_mv"][::-1]
                                          )

        # match prob from softmax
        # [BV, D, H, W] in feature resolution
        match_prob = results_dict['match_probs'][-1]
        match_prob = torch.max(match_prob, dim=1, keepdim=True)[
            0]  # [BV, 1, H, W]
        match_prob = F.interpolate(
            match_prob, size=depth.shape[-2:], mode='nearest')

        # unet input
        concat = torch.cat((
            rearrange(context["image"], "b v c h w -> (b v) c h w"),
            rearrange(depth, "b v h w -> (b v) () h w"),
            match_prob,
            features,
        ), dim=1)

        out = self.gaussian_regressor(concat)

        concat = [out,
                    rearrange(context["image"],
                            "b v c h w -> (b v) c h w"),
                    features,
                    match_prob]

        out = torch.cat(concat, dim=1)

        gaussians = self.gaussian_head(out)  # [BV, C, H, W]

        gaussians = rearrange(gaussians, "(b v) c h w -> b v c h w", b=b, v=v)

        depths = rearrange(depth, "b v h w -> b v (h w) () ()")

        # [B, V, H*W, 1, 1]
        densities = rearrange(
            match_prob, "(b v) c h w -> b v (c h w) () ()", b=b, v=v)
        # [B, V, H*W, 84]
        raw_gaussians = rearrange(
            gaussians, "b v c h w -> b v (h w) c")

        if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:

            # supervise all the intermediate depth predictions
            num_depths = len(depth_preds)

            # [B, V, H*W, 1, 1]
            intermediate_depths = torch.cat(
                depth_preds[:(num_depths - 1)], dim=0)
            
            intermediate_depths = rearrange(
                intermediate_depths, "b v h w -> b v (h w) () ()")

            # concat in the batch dim
            depths = torch.cat((intermediate_depths, depths), dim=0)

            # shared color head
            densities = torch.cat([densities] * num_depths, dim=0)
            raw_gaussians = torch.cat(
                [raw_gaussians] * num_depths, dim=0)

            b *= num_depths

        # [B, V, H*W, 1, 1]
        opacities = raw_gaussians[..., :1].sigmoid().unsqueeze(-1)
        raw_gaussians = raw_gaussians[..., 1:]
        
        # Convert the features and depths into Gaussians.
        xy_ray, _ = sample_image_grid((h, w), device)
        xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
        gaussians = rearrange(
            raw_gaussians,
            "... (srf c) -> ... srf c",
            srf=self.cfg.num_surfaces,
        )
        offset_xy = gaussians[..., :2].sigmoid()
        pixel_size = 1 / \
            torch.tensor((w, h), dtype=torch.float32, device=device)
        xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size

        sh_input_images = context["image"]

        if self.cfg.supervise_intermediate_depth and len(depth_preds) > 1:
            context_extrinsics = torch.cat(
                [context["extrinsics"]] * len(depth_preds), dim=0)
            context_intrinsics = torch.cat(
                [context["intrinsics"]] * len(depth_preds), dim=0)

            gaussians = self.gaussian_adapter.forward(
                rearrange(context_extrinsics, "b v i j -> b v () () () i j"),
                rearrange(context_intrinsics, "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                opacities,
                rearrange(
                    gaussians[..., 2:],
                    "b v r srf c -> b v r srf () c",
                ),
                (h, w),
                input_images=sh_input_images.repeat(
                    len(depth_preds), 1, 1, 1, 1) if self.cfg.init_sh_input_img else None,
            )

        else:
            gaussians = self.gaussian_adapter.forward(
                rearrange(context["extrinsics"],
                          "b v i j -> b v () () () i j"),
                rearrange(context["intrinsics"],
                          "b v i j -> b v () () () i j"),
                rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                depths,
                opacities,
                rearrange(
                    gaussians[..., 2:],
                    "b v r srf c -> b v r srf () c",
                ),
                (h, w),
                input_images=sh_input_images if self.cfg.init_sh_input_img else None,
            )

        # Dump visualizations if needed.
        if visualization_dump is not None:
            visualization_dump["depth"] = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            )
            visualization_dump["scales"] = rearrange(
                gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
            )
            visualization_dump["rotations"] = rearrange(
                gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            )

        gaussians = Gaussians(
            rearrange(
                gaussians.means,
                "b v r srf spp xyz -> b (v r srf spp) xyz",
            ),
            rearrange(
                gaussians.covariances,
                "b v r srf spp i j -> b (v r srf spp) i j",
            ),
            rearrange(
                gaussians.harmonics,
                "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
            ),
            rearrange(
                gaussians.opacities,
                "b v r srf spp -> b (v r srf spp)",
            ),
        )

        if self.cfg.return_depth:
            # return depth prediction for supervision
            depths = rearrange(
                depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
            ).squeeze(-1).squeeze(-1)
            # print(depths.shape)  # [B, V, H, W]

            return {
                "gaussians": gaussians,
                "depths": depths
            }

        return gaussians

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size
                * self.cfg.downscale_factor,
            )

            return batch

        return data_shim

    @property
    def sampler(self):
        return None


```

# prompts/papers/depthsplat/src/model/encoder/unimatch/backbone.py

``` py
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        planes,
        norm_layer=nn.InstanceNorm2d,
        stride=1,
        dilation=1,
    ):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
            stride=stride,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.relu = nn.ReLU(inplace=True)

        self.norm1 = norm_layer(planes)
        self.norm2 = norm_layer(planes)
        if not stride == 1 or in_planes != planes:
            self.norm3 = norm_layer(planes)

        if stride == 1 and in_planes == planes:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3
            )

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class CNNEncoder(nn.Module):
    def __init__(
        self,
        output_dim=128,
        norm_layer=nn.InstanceNorm2d,
        num_output_scales=1,
        return_quarter=False,  # return 1/4 resolution feature
        lowest_scale=8,  # lowest resolution, 1/8 or 1/4
        return_all_scales=False,
        **kwargs,
    ):
        super(CNNEncoder, self).__init__()
        self.num_scales = num_output_scales
        self.return_quarter = return_quarter
        self.lowest_scale = lowest_scale
        self.return_all_scales = return_all_scales

        feature_dims = [64, 96, 128]

        self.conv1 = nn.Conv2d(
            3, feature_dims[0], kernel_size=7, stride=2, padding=3, bias=False
        )  # 1/2
        self.norm1 = norm_layer(feature_dims[0])
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = feature_dims[0]
        self.layer1 = self._make_layer(
            feature_dims[0], stride=1, norm_layer=norm_layer
        )  # 1/2

        if self.lowest_scale == 4:
            stride = 1
        else:
            stride = 2
        self.layer2 = self._make_layer(
            feature_dims[1], stride=stride, norm_layer=norm_layer
        )  # 1/2 or 1/4

        # lowest resolution 1/4 or 1/8
        self.layer3 = self._make_layer(
            feature_dims[2],
            stride=2,
            norm_layer=norm_layer,
        )  # 1/4 or 1/8

        self.conv2 = nn.Conv2d(feature_dims[2], output_dim, 1, 1, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d):
        layer1 = ResidualBlock(
            self.in_planes, dim, norm_layer=norm_layer, stride=stride, dilation=dilation
        )
        layer2 = ResidualBlock(
            dim, dim, norm_layer=norm_layer, stride=1, dilation=dilation
        )

        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        output_all_scales = []
        output = []
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)  # 1/2

        if self.return_all_scales:
            output_all_scales.append(x)

        if self.num_scales >= 3:
            output.append(x)

        x = self.layer2(x)  # 1/2 or 1/4
        if self.return_quarter:
            output.append(x)

        if self.return_all_scales:
            output_all_scales.append(x)

        if self.num_scales >= 2:
            output.append(x)

        x = self.layer3(x)  # 1/4 or 1/8
        x = self.conv2(x)

        if self.return_all_scales:
            output_all_scales.append(x)

        if self.return_all_scales:
            return output_all_scales

        if self.return_quarter:
            output.append(x)
            return output

        if self.num_scales >= 1:
            output.append(x)
            return output

        out = [x]

        return out


```

# prompts/papers/depthsplat/src/model/encoder/unimatch/dpt_head.py

``` py
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3],
            out_shape4,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )

    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead(nn.Module):
    def __init__(
        self,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
        use_clstoken=False,
        concat_cnn_features=True,
        concat_mv_features=True,
        cnn_feature_channels=[64, 96, 128],
        concat_features=True,
        downsample_factor=8,
        return_feature=False,
        num_scales=1,
    ):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken

        self.concat_cnn_features = concat_cnn_features
        self.concat_mv_features = concat_mv_features
        self.concat_features = concat_features
        self.downsample_factor = downsample_factor
        self.return_feature = return_feature
        self.num_scales = num_scales

        if self.concat_features:
            if self.downsample_factor == 4 and num_scales == 2:
                depth_channel = 0 if self.return_feature else 1
                self.concat_projects = nn.ModuleList(
                    [
                        nn.Conv2d(
                            cnn_feature_channels[0] + out_channels[0],
                            out_channels[0],
                            1,
                        ),
                        nn.Conv2d(
                            cnn_feature_channels[1]
                            + out_channels[1]
                            + 64
                            + depth_channel,
                            out_channels[1],
                            1,
                        ),  # 1/4 concat(cnn, mono, mv, depth)
                        nn.Conv2d(
                            cnn_feature_channels[2] + out_channels[2] + 128,
                            out_channels[2],
                            1,
                        ),  # 1/8 concat(cnn, mono, mv)
                    ]
                )
            elif self.downsample_factor == 2 and num_scales == 2:
                depth_channel = 0 if self.return_feature else 1
                self.concat_projects = nn.ModuleList(
                    [
                        nn.Conv2d(
                            cnn_feature_channels[0]
                            + cnn_feature_channels[1]
                            + out_channels[0]
                            + 64
                            + depth_channel,
                            out_channels[0],
                            1,
                        ),  # 1/2
                        nn.Conv2d(
                            cnn_feature_channels[2] + out_channels[1] + 128,
                            out_channels[1],
                            1,
                        ),  # 1/4 concat(cnn, mono, mv, depth)
                        nn.Conv2d(out_channels[2], out_channels[2], 1),  # 1/8 mono
                    ]
                )
            elif self.downsample_factor == 4 and num_scales == 1:
                depth_channel = 0 if self.return_feature else 1
                self.concat_projects = nn.ModuleList(
                    [
                        nn.Conv2d(
                            cnn_feature_channels[0]
                            + cnn_feature_channels[1]
                            + out_channels[0],
                            out_channels[0],
                            1,
                        ),
                        nn.Conv2d(
                            cnn_feature_channels[2]
                            + out_channels[1]
                            + 128
                            + depth_channel,
                            out_channels[1],
                            1,
                        ),
                        nn.Conv2d(out_channels[2], out_channels[2], 1),  # 1/8 mono
                    ]
                )
            else:
                depth_channel = 0 if self.return_feature else 1
                self.concat_projects = nn.ModuleList(
                    [
                        nn.Conv2d(
                            cnn_feature_channels[0] + out_channels[0],
                            out_channels[0],
                            1,
                        ),
                        nn.Conv2d(
                            cnn_feature_channels[1] + out_channels[1],
                            out_channels[1],
                            1,
                        ),
                        nn.Conv2d(
                            cnn_feature_channels[2]
                            + out_channels[2]
                            + 128
                            + depth_channel,
                            out_channels[2],
                            1,
                        ),  # 1/8 concat(cnn, mono, mv, depth)
                    ]
                )
        else:
            if self.concat_cnn_features:
                self.cnn_projects = nn.ModuleList(
                    [
                        nn.Conv2d(cnn_feature_channels[i], out_channels[i], 1)
                        for i in range(len(cnn_feature_channels))
                    ]
                )

            if self.concat_mv_features:
                self.mv_projects = nn.Conv2d(128, out_channels[2], 1)

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for out_channel in out_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0],
                    out_channels=out_channels[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1],
                    out_channels=out_channels[1],
                    kernel_size=2,
                    stride=2,
                    padding=0,
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3],
                    out_channels=out_channels[3],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * in_channels, in_channels), nn.GELU())
                )

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # not used
        del self.scratch.refinenet4.resConfUnit1

        head_features_1 = features
        head_features_2 = 16

        if not self.return_feature:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(
                    head_features_1,
                    head_features_1 // 2,
                    3,
                    1,
                    1,
                    padding_mode="replicate",
                ),
                nn.GELU(),
                nn.Conv2d(
                    head_features_1 // 2,
                    head_features_2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode="replicate",
                ),
                nn.GELU(),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            )

            # init delta depth as zero
            nn.init.zeros_(self.scratch.output_conv[-1].weight)
            nn.init.zeros_(self.scratch.output_conv[-1].bias)

    def forward(
        self,
        out_features,
        downsample_factor=8,
        cnn_features=None,
        mv_features=None,
        depth=None,
    ):
        out = []
        for i, x in enumerate(out_features):
            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        # 1/2, 1/4, 1/8, 1/16
        layer_1, layer_2, layer_3, layer_4 = out

        if self.concat_features:
            if not self.return_feature:
                assert depth is not None

            if self.downsample_factor == 4 and self.num_scales == 1:
                concat1 = torch.cat((cnn_features[0], cnn_features[1], layer_1), dim=1)
            elif self.downsample_factor == 2 and self.num_scales == 2:
                if self.return_feature:
                    concat1 = torch.cat(
                        (cnn_features[0], cnn_features[1], mv_features[0], layer_1),
                        dim=1,
                    )
                else:
                    concat1 = torch.cat(
                        (
                            cnn_features[0],
                            cnn_features[1],
                            mv_features[0],
                            depth,
                            layer_1,
                        ),
                        dim=1,
                    )
            else:
                concat1 = torch.cat((cnn_features[0], layer_1), dim=1)
            layer_1 = self.concat_projects[0](concat1)  # 1/2

            if self.downsample_factor == 4 and self.num_scales == 2:
                assert isinstance(mv_features, list)
                if self.return_feature:
                    concat2 = torch.cat(
                        (cnn_features[1], layer_2, mv_features[0]), dim=1
                    )
                else:
                    concat2 = torch.cat(
                        (cnn_features[1], layer_2, mv_features[0], depth), dim=1
                    )
                layer_2 = self.concat_projects[1](concat2)  # 1/4

                concat3 = torch.cat((cnn_features[2], layer_3, mv_features[1]), dim=1)
                layer_3 = self.concat_projects[2](concat3)  # 1/8
            elif self.downsample_factor == 2 and self.num_scales == 2:
                assert isinstance(mv_features, list)
                concat2 = torch.cat((cnn_features[2], layer_2, mv_features[1]), dim=1)
                layer_2 = self.concat_projects[1](concat2)  # 1/4

                concat3 = layer_3
                layer_3 = self.concat_projects[2](concat3)  # 1/8
            elif self.downsample_factor == 4 and self.num_scales == 1:
                if self.return_feature:
                    concat2 = torch.cat((cnn_features[2], layer_2, mv_features), dim=1)
                else:
                    concat2 = torch.cat(
                        (cnn_features[2], layer_2, mv_features, depth), dim=1
                    )
                layer_2 = self.concat_projects[1](concat2)  # 1/4

                concat3 = layer_3
                layer_3 = self.concat_projects[2](concat3)  # 1/8
            else:
                concat2 = torch.cat((cnn_features[1], layer_2), dim=1)
                layer_2 = self.concat_projects[1](concat2)  # 1/4

                if self.return_feature:
                    concat3 = torch.cat((cnn_features[2], layer_3, mv_features), dim=1)
                else:
                    concat3 = torch.cat(
                        (cnn_features[2], layer_3, mv_features, depth), dim=1
                    )
                layer_3 = self.concat_projects[2](concat3)  # 1/8
        else:
            if self.concat_cnn_features:
                assert cnn_features is not None
                assert len(cnn_features) == 3  # 1/2, 1/4, 1/8
                cnn_features = [
                    self.cnn_projects[i](f) for i, f in enumerate(cnn_features)
                ]

                layer_1 = layer_1 + cnn_features[0]  # 1/2
                layer_2 = layer_2 + cnn_features[1]  # 1/4
                layer_3 = layer_3 + cnn_features[2]  # 1/8

            if self.concat_mv_features:
                # 1/8
                mv_features = self.mv_projects(mv_features)

                layer_3 = layer_3 + mv_features  # 1/8

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])  # 1/8
        path_3 = self.scratch.refinenet3(
            path_4, layer_3_rn, size=layer_2_rn.shape[2:]
        )  # 1/4
        path_2 = self.scratch.refinenet2(
            path_3, layer_2_rn, size=layer_1_rn.shape[2:]
        )  # 1/2
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)  # 1

        if self.return_feature:
            return path_1

        out = self.scratch.output_conv(path_1)

        return out


if __name__ == "__main__":
    device = torch.device("cuda")
    c = 384
    model = DPTHead(
        in_channels=c,
        concat_cnn_features=True,
        concat_mv_features=True,
    ).to(device)
    print(model)

    h, w = 16, 32

    x = torch.randn(2, c, h, w).to(device)

    out_features = [x] * 4

    cnn_features = [
        torch.randn(2, 64, h * 4, w * 4).to(device),
        torch.randn(2, 96, h * 2, w * 2).to(device),
        torch.randn(2, 128, h, w).to(device),
    ]

    mv_features = torch.randn(2, 128, h, w).to(device)

    out = model(out_features, h, w, cnn_features=cnn_features, mv_features=mv_features)

    print(out.shape)


```

# prompts/papers/depthsplat/src/model/encoder/unimatch/feature_upsampler.py

``` py
import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class ResizeConvFeatureUpsampler(nn.Module):
    """
    https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, num_scales=1,
                 lowest_feature_resolution=8,
                 out_channels=128,
                 vit_type='vits',
                 no_mono_feature=False,
                 gaussian_downsample=None,
                 monodepth_backbone=False,
                 ):
        super(ResizeConvFeatureUpsampler, self).__init__()

        self.num_scales = num_scales
        self.monodepth_backbone = monodepth_backbone

        self.upsampler = nn.ModuleList()

        vit_feature_channel_dict = {
            'vits': 384,
            'vitb': 768,
            'vitl': 1024
        }

        vit_feature_channel = vit_feature_channel_dict[vit_type]

        if monodepth_backbone:
            vit_feature_channel = 384

        out_channels = out_channels // num_scales

        for i in range(num_scales):
            cnn_feature_channels = 128 - (32 * i)
            mv_transformer_feature_channels = 128 // (2 ** i)
            if no_mono_feature:
                mono_feature_channels = 0
            else:
                mono_feature_channels = vit_feature_channel // (2 ** i)

            in_channels = cnn_feature_channels + \
                mv_transformer_feature_channels + mono_feature_channels

            if monodepth_backbone:
                in_channels = 384

            curr_upsample_factor = lowest_feature_resolution // (2 ** i)

            num_upsample = int(math.log(curr_upsample_factor, 2))

            modules = []
            if num_upsample == 1:
                curr_in_channels = out_channels * 2
            else:
                curr_in_channels = out_channels * 2 * (num_upsample - 1)
            modules.append(nn.Conv2d(in_channels, curr_in_channels, 1))
            for i in range(num_upsample):
                modules.append(nn.Upsample(scale_factor=2, mode='nearest'))

                if i == num_upsample - 1:
                    modules.append(nn.Conv2d(curr_in_channels,
                                             out_channels, 3, 1, 1, padding_mode='replicate'))
                else:
                    modules.append(nn.Conv2d(curr_in_channels,
                                             curr_in_channels // 2, 3, 1, 1, padding_mode='replicate'))
                    curr_in_channels = curr_in_channels // 2
                    modules.append(nn.GELU())

            if gaussian_downsample is not None:
                if gaussian_downsample == 2:
                    del modules[-3:]
                elif gaussian_downsample == 4:
                    del modules[-6:]
                else:
                    raise NotImplementedError

            self.upsampler.append(nn.Sequential(*modules))

    def forward(self, features_list_cnn, features_list_mv, features_list_mono=None):
        out = []

        for i in range(self.num_scales):
            if self.monodepth_backbone:
                concat = features_list_cnn[i]
            elif features_list_mono is None:
                concat = torch.cat(
                (features_list_cnn[i], features_list_mv[i]), dim=1)
            else:
                concat = torch.cat(
                    (features_list_cnn[i], features_list_mv[i], features_list_mono[i]), dim=1)
            concat = self.upsampler[i](concat)

            out.append(concat)

        out = torch.cat(out, dim=1)

        return out


def _test():
    device = torch.device('cuda:0')
    
    model = ResizeConvFeatureUpsampler(num_scales=2,
                                       lowest_feature_resolution=4,
                                       ).to(device)
    print(model)

    b, h, w = 2, 32, 64
    features_list_cnn = [torch.randn(b, 128, h, w).to(device)]
    features_list_mv = [torch.randn(b, 128, h, w).to(device)]
    features_list_mono = [torch.randn(b, 384, h, w).to(device)]

    # scale 2
    features_list_cnn.append(torch.randn(b, 96, h * 2, w * 2).to(device))
    features_list_mv.append(torch.randn(b, 64, h * 2, w * 2).to(device))
    features_list_mono.append(torch.randn(b, 192, h * 2, w * 2).to(device))

    out = model(features_list_cnn,
                features_list_mv, features_list_mono)

    print(out.shape)


if __name__ == '__main__':
    _test()


```

# prompts/papers/depthsplat/src/model/encoder/unimatch/ldm_unet/attention.py

``` py
from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=False):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        # self.checkpoint = checkpoint

    def forward(self, x, context=None):
        # return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

        return _forward(x, context)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in

```

# prompts/papers/depthsplat/src/model/encoder/unimatch/ldm_unet/cross_attention.py

``` py
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import warnings


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        # warnings.warn("xFormers is available (Attention)")
    else:
        # warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    # warnings.warn("xFormers is not available (Attention)")


class CrossAttention(nn.Module):
    def __init__(
        self, 
        in_dim1,
        in_dim2,
        dim=128,
        out_dim=None,
        num_heads=4,
        qkv_bias=False,
        proj_bias=False,
    ):
        super().__init__()

        assert XFORMERS_AVAILABLE

        if out_dim is None:
            out_dim = in_dim1

        self.num_heads = num_heads
        self.dim = dim
        self.q = nn.Linear(in_dim1, dim, bias=qkv_bias)
        self.kv = nn.Linear(in_dim2, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, out_dim, bias=proj_bias)

    def forward(self, x, y):
        c = self.dim
        b, n1, c1 = x.shape
        n2, c2 = y.shape[1:]

        q = self.q(x).reshape(b, n1, self.num_heads, c // self.num_heads)
        kv = self.kv(y).reshape(b, n2, 2, self.num_heads, c // self.num_heads)
        k, v = unbind(kv, 2)

        x = memory_efficient_attention(q, k, v)
        x = x.reshape(b, n1, c)
        
        x = self.proj(x)

        return x
        

class UNetCrossAttentionBlock(nn.Module):
    def __init__(self,
        in_dim1,
        in_dim2,
        dim=128,
        out_dim=None,
        num_heads=4,
        qkv_bias=False,
        proj_bias=False,
        with_ffn=False,
        concat_cross_attn=False,
        concat_output=False,
        no_cross_attn=False,
        with_norm=False,
        concat_conv3x3=False,
        ):
        super().__init__()

        out_dim = out_dim or in_dim1

        self.no_cross_attn = no_cross_attn
        self.with_norm = with_norm

        if no_cross_attn:
            if concat_conv3x3:
                self.proj = nn.Conv2d(in_dim1 + in_dim2, out_dim, 3, 1, 1)
            else:
                self.proj = nn.Conv2d(in_dim1 + in_dim2, out_dim, 1)
        else:
            self.with_ffn = with_ffn
            self.concat_cross_attn = concat_cross_attn
            self.concat_output = concat_output
            
            self.cross_attn = CrossAttention(
                in_dim1=in_dim1,
                in_dim2=in_dim2,
                dim=dim,
                out_dim=out_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
            )

            if with_norm:
                self.norm1 = nn.LayerNorm(out_dim)
            else:
                self.norm1 = nn.Identity()

            if with_ffn:
                in_channels = out_dim + in_dim1 if concat_cross_attn else in_dim1
                ffn_dim_expansion = 4
                self.mlp = nn.Sequential(
                    nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                    nn.GELU(),
                    nn.Linear(in_channels * ffn_dim_expansion, in_dim1, bias=False),
                )

                if with_norm:
                    self.norm2 = nn.LayerNorm(in_dim1)
                else:
                    self.norm2 = nn.Identity()

            if self.concat_output:
                self.out = nn.Linear(out_dim + in_dim1, in_dim1)

    def forward(self, x, y):
        # x: [B, C, H, W]
        # y: [B, N, C] or [B, C, H, W]

        if self.no_cross_attn:
            assert x.dim() == 4 and y.dim() == 4
            if y.shape[2:] != x.shape[2:]:
                y = F.interpolate(y, x.shape[2:], mode='bilinear', align_corners=True)
            return self.proj(torch.cat((x, y), dim=1))

        identity = x

        b, c, h, w = x.size()
        x = x.view(b, c, -1).permute(0, 2, 1)

        cross_attn = self.norm1(self.cross_attn(x, y))

        if self.with_ffn:
            if self.concat_cross_attn:
                concat = torch.cat((x, cross_attn), dim=-1)
            else:
                concat = x + cross_attn

            cross_attn = self.norm2(self.mlp(concat))

        if self.concat_output:
            return self.out(torch.cat((x, cross_attn), dim=-1))

        # reshape back
        cross_attn = cross_attn.view(b, h, w, c).permute(0, 3, 1, 2)  # [B, C, H, W]

        return identity + cross_attn



```

# prompts/papers/depthsplat/src/model/encoder/unimatch/ldm_unet/unet.py

``` py
from abc import abstractmethod
from functools import partial
import math
from typing import Iterable
from einops import rearrange

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch

from .util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from .attention import SpatialTransformer

from .cross_attention import UNetCrossAttentionBlock


# dummy replace
def convert_module_to_f16(x):
    pass

def convert_module_to_f32(x):
    pass


## go
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1,
    downsample_3ddim=False,  # downsample all 3d dims instead of only spatial dims
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=padding)

        self.downsample_3ddim = downsample_3ddim

    def forward(self, x, y=None):
        assert x.shape[1] == self.channels
        if self.dims == 3 and not self.downsample_3ddim:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class TransposedUpsample(nn.Module):
    'Learned 2x upsampling without padding'
    def __init__(self, channels, out_channels=None, ks=5):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.up = nn.ConvTranspose2d(self.channels,self.out_channels,kernel_size=ks,stride=2)

    def forward(self,x):
        return self.up(x)


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None,padding=1, 
    downsample_3ddim=False,  # downsample all 3d dims instead of only spatial dims
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)

        if downsample_3ddim:
            assert dims == 3
            stride = 2

        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x, y=None):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        postnorm=False,
        channels_per_group=None,
        kernel_size=3,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        if postnorm:
            self.in_layers = nn.Sequential(
                conv_nd(dims, channels, self.out_channels, kernel_size, padding=(kernel_size - 1) // 2),
                normalization(self.out_channels, channels_per_group=channels_per_group),
                nn.SiLU(),
            )
        else:
            self.in_layers = nn.Sequential(
                normalization(channels, channels_per_group=channels_per_group),
                nn.SiLU(),
                conv_nd(dims, channels, self.out_channels, kernel_size, padding=(kernel_size - 1) // 2),
            )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # self.emb_layers = nn.Sequential(
        #     nn.SiLU(),
        #     linear(
        #         emb_channels,
        #         2 * self.out_channels if use_scale_shift_norm else self.out_channels,
        #     ),
        # )

        if postnorm:
            self.out_layers = nn.Sequential(
                conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=(kernel_size - 1) // 2),
                zero_module(
                    normalization(self.out_channels, channels_per_group=channels_per_group),
                ),
                nn.SiLU(),
        )
        else:
            self.out_layers = nn.Sequential(
                normalization(self.out_channels, channels_per_group=channels_per_group),
                nn.SiLU(),
                nn.Dropout(p=dropout),
                zero_module(
                    conv_nd(dims, self.out_channels, self.out_channels, kernel_size, padding=(kernel_size - 1) // 2)
                ),
            )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, kernel_size, padding=(kernel_size - 1) // 2
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb=None):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        # emb_out = self.emb_layers(emb).type(h.dtype)
        # while len(emb_out.shape) < len(h.shape):
        #     emb_out = emb_out[..., None]
        # if self.use_scale_shift_norm:
        #     out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        #     scale, shift = th.chunk(emb_out, 2, dim=1)
        #     h = out_norm(h) * (1 + scale) + shift
        #     h = out_rest(h)
        # else:
        #     h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
        postnorm=False,
        channels_per_group=None,
        num_frames=2,
        use_cross_view_self_attn=False,
    ):
        super().__init__()

        # NOTE: current attention layer doesn't have positional encoding (TODO)
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(
                self.num_heads,
                n_frames=num_frames,
                use_cross_view_self_attn=use_cross_view_self_attn,
            )

        if postnorm:
            self.proj_out = conv_nd(1, channels, channels, 1)
            self.norm = zero_module(normalization(channels, channels_per_group=channels_per_group))
        else:
            self.norm = normalization(channels, channels_per_group=channels_per_group)
            self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

        self.postnorm = postnorm

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)

        if self.postnorm:
            qkv = self.qkv(x)
            h = self.attention(qkv)
            h = self.proj_out(h)
            h = self.norm(h)
        else:
            qkv = self.qkv(self.norm(x))
            h = self.attention(qkv)
            h = self.proj_out(h)

        return (x + h).reshape(b, c, *spatial)


class CrossAttentionBlock(nn.Module):
    """
    Corss attention conditioning
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        condition_channels,
        num_heads=8,
        proj_channels=512,
        num_views=3,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
        channels_per_group=None,
        with_norm=False,
        tanh_gating=False,  # following Flamingo
        ffn_after_cross_attn=False,  # following Flamingo
    ):
        super().__init__()

        self.channels = channels
        self.num_head = num_heads
        self.num_views = num_views
        self.proj_channels = proj_channels
        self.with_norm = with_norm
        self.tanh_gating = tanh_gating
        self.ffn_after_cross_attn = ffn_after_cross_attn

        self.q_proj = nn.Linear(channels, proj_channels)
        self.k_proj = nn.Linear(condition_channels, proj_channels)
        self.v_proj = nn.Linear(condition_channels, proj_channels)

        # TODO: whether need norm layer
        # self.norm = normalization(proj_channels, channels_per_group=channels_per_group)

        if self.tanh_gating:
            self.out_proj = conv_nd(3, proj_channels, channels, 1)
            self.attn_gate = nn.Parameter(torch.tensor([0.]))
        else:
            if self.with_norm:
                self.out_proj = conv_nd(3, proj_channels, channels, 1)
                self.norm = zero_module(normalization(channels, channels_per_group=channels_per_group))
            else:
                self.out_proj = zero_module(conv_nd(3, proj_channels, channels, 1))

        if self.ffn_after_cross_attn:
            self.ffn_gate = nn.Parameter(torch.tensor([0.]))
            self.ffn = nn.Sequential(nn.Conv3d(channels, channels * 4, 1, 1, 0),
            normalization(channels * 4, channels_per_group=channels_per_group),
            nn.GELU(),
            nn.Conv3d(channels * 4, channels, 1, 1, 0)
            )

    def forward(self, x, y=None):
        # return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!
        #return pt_checkpoint(self._forward, x)  # pytorch
        return self._forward(x, y)

    def _forward(self, x, y=None):
        # x: [B, C, D, H, W], feature
        # y: [B, H*W, D, C]
        print(x.shape, y.shape)
        assert x.dim() == 5 and y.dim() == 4
        # NOTE: the resolutions of feature x and color y are different

        b, c1, d, h, w = x.size()
        lx = h * w
        ly = y.size(1)
        c2 = y.size(-1)

        identity = x

        x = x.permute(0, 2, 3, 4, 1).reshape(b * d, h * w, c1)  # [B*D, H*W, C1]
        y = y.permute(0, 2, 1, 3).reshape(b * d, ly, c2)  # [B*D, H*W, C2]
        
        c = self.proj_channels

        q = self.q_proj(x)  # [B*D, H*W, C]
        k = self.k_proj(y)  # [B*D, H*W, C]
        v = self.v_proj(y)

        if self.num_head > 1:
            assert c % self.num_head == 0
            q = q.view(b * d, lx, self.num_head, c // self.num_head)  # [B*D, H*W, N, C]
            k = k.view(b * d, ly, self.num_head, c // self.num_head)  # [B*D, H*W, N, C]
            v = v.view(b * d, ly, self.num_head, c // self.num_head)  # [B*D, H*W, N, C]

            scores = torch.matmul(q.permute(0, 2, 1, 3), k.permute(0, 2, 3, 1)) / ((c // self.num_head) ** 0.5)  # [B*D, N, H*W, H*W]
            prob = torch.softmax(scores, dim=-1)
            out = torch.matmul(prob, v.permute(0, 2, 1, 3))  # [B*D, H*W, N, C]
            out = out.view(b * d, lx, -1)  # [B*D, H*W, C]

        else:
            scores = torch.matmul(q, k.permute(0, 2, 1)) / (c ** 0.5)  # [B*D, H*W, H*W]
            prob = torch.softmax(scores, dim=-1)

            out = torch.matmul(prob, v)  # [B*D, H*W, C]

        out = out.view(b, d, h, w, c).permute(0, 4, 1, 2, 3)  # [B, C, D, H, W]

        # out = self.norm(out)

        if self.tanh_gating:
            # print('tanh', self.attn_gate.tanh())
            out = self.attn_gate.tanh() * self.out_proj(out)
        else:
            if self.with_norm:
                out = self.out_proj(out)
                out = self.norm(out)
            else:
                out = self.out_proj(out)

        out = identity + out

        if self.ffn_after_cross_attn:
            out = out + self.ffn_gate.tanh() * self.ffn(out)

        return out


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads, n_frames=2, use_cross_view_self_attn=False):
        super().__init__()
        self.n_heads = n_heads
        self.n_frames = n_frames
        self.use_cross_view_self_attn = use_cross_view_self_attn

    def forward(self, qkv, num_views=None):
        """
        Apply QKV attention.
        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """

        # move the view dim into T for cross views attention
        # (b v) ...
        if self.use_cross_view_self_attn:
            n_views = self.n_frames if num_views is None else num_views
            qkv = rearrange(qkv, "(b v) n t -> b n (v t)", v=n_views)

        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v).reshape(bs, -1, length)

        # move view dim back to batch dim in original '(b v)' order
        if self.use_cross_view_self_attn:
            a = rearrange(a, "b n (v t) -> (b v) n t", v=n_views)

        return a

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        middle_block_attn=False,  # use attn in middle block
        middle_block_no_identity=False,  # some previous models are trained without the identity layer
        postnorm=False,  # default prenorm doesn't converge
        attn_prenorm=False,  # try postnorm for resblock and prenorm for attn
        downsample_3ddim=False,  # downsample all 3d dims instead of only spatial dims
        zero_final_layer=False,  # init zero final output layer
        channels_per_group=None,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        cross_attn_condition=False,  
        tanh_gating=False,
        ffn_after_cross_attn=False,
        cross_attn_with_norm=False,
        condition_channels=384,
        condition_num_views=3,
        no_self_attn=False,
        conv_kernel_size=3,
        concat_condition=False,
        concat_conv3x3=False,
        num_frames=2,
        use_cross_view_self_attn=False,
        downsample_factor=None,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        self.middle_block_attn = middle_block_attn
        
        self.middle_block_no_identity = middle_block_no_identity

        # output lower resolution gaussians
        self.downsample_factor = downsample_factor

        time_embed_dim = model_channels * 4
        # self.time_embed = nn.Sequential(
        #     linear(model_channels, time_embed_dim),
        #     nn.SiLU(),
        #     linear(time_embed_dim, time_embed_dim),
        # )

        # if self.num_classes is not None:
        #     self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        self.cross_attn_condition = cross_attn_condition

        self.input_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        postnorm=postnorm,
                        channels_per_group=channels_per_group,
                        kernel_size=conv_kernel_size,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

                    if not no_self_attn:  # only cross attn, without self attn
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                                postnorm=False if attn_prenorm else postnorm,
                                channels_per_group=channels_per_group,
                                num_frames=num_frames,
                                use_cross_view_self_attn=use_cross_view_self_attn,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                            )
                        )

                    if cross_attn_condition:
                        layers.append(
                            UNetCrossAttentionBlock(ch,
                            condition_channels,
                            dim=256,
                            no_cross_attn=concat_condition,
                            with_norm=cross_attn_with_norm,
                            concat_conv3x3=concat_conv3x3,
                            )
                        )

                self.input_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    nn.Sequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            postnorm=postnorm,
                            channels_per_group=channels_per_group,
                            kernel_size=conv_kernel_size,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch,
                            downsample_3ddim=downsample_3ddim,
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        if self.middle_block_attn:
            self.middle_block = nn.Sequential(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    postnorm=postnorm,
                    channels_per_group=channels_per_group,
                    kernel_size=conv_kernel_size,
                ),
                # original has attention block in the middle
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=dim_head,
                    use_new_attention_order=use_new_attention_order,
                    postnorm=False if attn_prenorm else postnorm,
                    channels_per_group=channels_per_group,
                    num_frames=num_frames,
                    use_cross_view_self_attn=use_cross_view_self_attn,
                ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                            ),
                # cross attention condition
                UNetCrossAttentionBlock(ch,
                condition_channels,
                dim=256,
                no_cross_attn=concat_condition,
                with_norm=cross_attn_with_norm,
                concat_conv3x3=concat_conv3x3,
                ) if cross_attn_condition else nn.Identity(),
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    postnorm=postnorm,
                    channels_per_group=channels_per_group,
                    kernel_size=conv_kernel_size,
                ),
            )
        else:
            if self.middle_block_no_identity:
                self.middle_block = nn.Sequential(
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    postnorm=postnorm,
                    channels_per_group=channels_per_group,
                    kernel_size=conv_kernel_size,
                ),
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    postnorm=postnorm,
                    channels_per_group=channels_per_group,
                    kernel_size=conv_kernel_size,
                ),
                )
            else:
                self.middle_block = nn.Sequential(
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        postnorm=postnorm,
                        channels_per_group=channels_per_group,
                        kernel_size=conv_kernel_size,
                    ),
                    UNetCrossAttentionBlock(ch,
                    condition_channels,
                    dim=256,
                    no_cross_attn=concat_condition,
                    with_norm=cross_attn_with_norm,
                    concat_conv3x3=concat_conv3x3,
                    ) if cross_attn_condition else nn.Identity(),
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        postnorm=postnorm,
                        channels_per_group=channels_per_group,
                        kernel_size=conv_kernel_size,
                    ),
                )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        postnorm=postnorm,
                        channels_per_group=channels_per_group,
                        kernel_size=conv_kernel_size,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

                    if not no_self_attn:  # only cross attn, without self attn
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                                postnorm=False if attn_prenorm else postnorm,
                                channels_per_group=channels_per_group,
                                num_frames=num_frames,
                                use_cross_view_self_attn=use_cross_view_self_attn,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim
                            )
                        )

                    if cross_attn_condition:
                        layers.append(
                            UNetCrossAttentionBlock(ch,
                            condition_channels,
                            dim=256,
                            no_cross_attn=concat_condition,
                            with_norm=cross_attn_with_norm,
                            concat_conv3x3=concat_conv3x3,
                            )
                        )
                    
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            postnorm=postnorm,
                            channels_per_group=channels_per_group,
                            kernel_size=conv_kernel_size,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch,
                        downsample_3ddim=downsample_3ddim,
                        )
                    )
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))
                self._feature_size += ch

        if self.downsample_factor is not None:
            if self.downsample_factor == 2:
                del self.output_blocks[-3:]
            elif self.downsample_factor == 4:
                del self.output_blocks[-5:]
            else:
                raise NotImplementedError

        if postnorm:
            self.out = nn.Sequential(
                conv_nd(dims, model_channels, out_channels, 3, padding=1),
                normalization(out_channels, channels_per_group=channels_per_group) if not zero_final_layer else zero_module(normalization(out_channels, channels_per_group=channels_per_group)),
                nn.SiLU(),
            )
        else:
            if self.downsample_factor is not None:
                in_channels = self.model_channels * self.channel_mult[self.downsample_factor // 2]
                model_channels = model_channels * self.channel_mult[self.downsample_factor // 2]
                out_channels = model_channels
            else:
                in_channels = ch
            self.out = nn.Sequential(
                normalization(in_channels, channels_per_group=channels_per_group),
                nn.SiLU(),
                zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
            )

        # print(self.out)

        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch, channels_per_group=channels_per_group),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, num_views=None, timesteps=None, context=None, y=None,**kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        # t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        # emb = self.time_embed(t_emb)
        emb = None

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            # h = module(h, emb, context)
            # if i == 0:  # conv layer

            if self.cross_attn_condition:
                for submodule in module:
                    if 'UNetCrossAttentionBlock' == submodule.__class__.__name__:
                        h = submodule(h, context)
                    else:
                        h = submodule(h)
            else:
                h = module(h)
            # else:
            #     print(module)
            #     h = module(h, context)
            hs.append(h)
        # h = self.middle_block(h, emb, context)
        # h = self.middle_block(h)

        for module in self.middle_block:
            if 'UNetCrossAttentionBlock' == module.__class__.__name__:
                h = module(h, context)
            else:
                h = module(h)

        # print(len(self.output_blocks))

        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            # h = module(h, emb, context)
            if self.cross_attn_condition:
                for submodule in module:
                    if 'UNetCrossAttentionBlock' == submodule.__class__.__name__:
                        h = submodule(h, context)
                    else:
                        h = submodule(h)
            else:
                h = module(h)
            # h = module(h, context)
            # print(h.shape)
        h = h.type(x.dtype)
        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


class StackUNet(nn.Module):
    def __init__(self,
                 in_channels,
                 model_channels,
                 out_channels,
                 num_res_blocks=1,
                 attention_resolutions=[],
                 channel_mult=[1, 1, 1, 1],
                 num_head_channels=32,
                 dims=3,
                 postnorm=True,
                 attn_prenorm=False,
                 middle_block_attn=False,
                 num_stacks=1,
                 zero_final_layer=False,
                 resblock_updown=False,
                 channels_per_group=None,
                 cross_attn_condition=False,  
                 cross_attn_with_norm=False,
                condition_channels=128,
                tanh_gating=False,
                ffn_after_cross_attn=False,
                condition_num_views=3,
                no_self_attn=False,
                middle_block_no_identity=False,
                conv_kernel_size=3,
                 ):

        super().__init__()

        self.num_stacks = num_stacks

        self.stacks = nn.ModuleList()

        in_channels = in_channels

        for i in range(num_stacks):
            self.stacks.append(UNetModel(image_size=None,
                            in_channels=in_channels,
                            model_channels=model_channels,
                            out_channels=out_channels,
                            num_res_blocks=num_res_blocks,
                            attention_resolutions=attention_resolutions,
                            channel_mult=channel_mult,
                            num_head_channels=num_head_channels,
                            dims=dims,
                            middle_block_attn=middle_block_attn,
                            middle_block_no_identity=middle_block_no_identity,
                            postnorm=postnorm,
                            attn_prenorm=attn_prenorm,
                            zero_final_layer=zero_final_layer and i == 0,
                            resblock_updown=resblock_updown,
                            channels_per_group=channels_per_group,
                            cross_attn_condition=cross_attn_condition,  
                            tanh_gating=tanh_gating,
                            ffn_after_cross_attn=ffn_after_cross_attn,
                            cross_attn_with_norm=cross_attn_with_norm,
                            condition_channels=condition_channels,
                            condition_num_views=condition_num_views,
                            no_self_attn=no_self_attn,
                            conv_kernel_size=conv_kernel_size,
                            )
            )

            in_channels = out_channels

        self.convs = nn.ModuleList()

        for i in range(num_stacks - 1):
            self.convs.append(zero_module(conv_nd(
                dims, out_channels, in_channels, 3, padding=1
            )))


    def forward(self, x, context=None):
        x = self.stacks[0](x, context=context)
        for i in range(self.num_stacks - 1):
            residual = self.convs[i](self.stacks[i + 1](x, context=context))
            x = x + residual

        return x





```

# prompts/papers/depthsplat/src/model/encoder/unimatch/ldm_unet/util.py

``` py
# adopted from
# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# and
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# and
# https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py
#
# thanks!


import os
import math
import torch
import torch.nn as nn
import numpy as np
from einops import repeat

# from ldm.util import instantiate_from_config


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels, channels_per_group=None):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    # return GroupNorm32(32, channels)  # original

    if channels_per_group is not None:
        return GroupNorm(channels // channels_per_group, channels)
        # return GroupNorm4(4, channels)
        # if channels % channels_per_group == 0:
        #     # adjust group number according to the channels
        #     return GroupNorm8(channels // channels_per_group, channels)
        # else:
        #     return GroupNorm4(4, channels)

    if channels % 8 != 0:
        return GroupNorm4(4, channels)

    return GroupNorm8(8, channels)


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class GroupNorm8(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

class GroupNorm4(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


# class HybridConditioner(nn.Module):

#     def __init__(self, c_concat_config, c_crossattn_config):
#         super().__init__()
#         self.concat_conditioner = instantiate_from_config(c_concat_config)
#         self.crossattn_conditioner = instantiate_from_config(c_crossattn_config)

#     def forward(self, c_concat, c_crossattn):
#         c_concat = self.concat_conditioner(c_concat)
#         c_crossattn = self.crossattn_conditioner(c_crossattn)
#         return {'c_concat': [c_concat], 'c_crossattn': [c_crossattn]}


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

```

# prompts/papers/depthsplat/src/model/encoder/unimatch/matching.py

``` py
import torch
import torch.nn.functional as F


def coords_grid(b, h, w, homogeneous=False, device=None):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w))  # [H, W]

    stacks = [x, y]

    if homogeneous:
        ones = torch.ones_like(x)  # [H, W]
        stacks.append(ones)

    grid = torch.stack(stacks, dim=0).float()  # [2, H, W] or [3, H, W]

    grid = grid[None].repeat(b, 1, 1, 1)  # [B, 2, H, W] or [B, 3, H, W]

    if device is not None:
        grid = grid.to(device)

    return grid


def warp_with_pose_depth_candidates(
    feature1,
    intrinsics,
    pose,
    depth,
    clamp_min_depth=1e-3,
    grid_sample_disable_cudnn=False,
):
    """
    feature1: [B, C, H, W]
    intrinsics: [B, 3, 3]
    pose: [B, 4, 4]
    depth: [B, D, H, W]
    """

    assert intrinsics.size(1) == intrinsics.size(2) == 3
    assert pose.size(1) == pose.size(2) == 4
    assert depth.dim() == 4

    b, d, h, w = depth.size()
    c = feature1.size(1)

    with torch.no_grad():
        # pixel coordinates
        grid = coords_grid(
            b, h, w, homogeneous=True, device=depth.device
        )  # [B, 3, H, W]
        # back project to 3D and transform viewpoint
        points = torch.inverse(intrinsics).bmm(grid.view(b, 3, -1))  # [B, 3, H*W]
        points = torch.bmm(pose[:, :3, :3], points).unsqueeze(2).repeat(
            1, 1, d, 1
        ) * depth.view(
            b, 1, d, h * w
        )  # [B, 3, D, H*W]
        points = points + pose[:, :3, -1:].unsqueeze(-1)  # [B, 3, D, H*W]
        # reproject to 2D image plane
        points = torch.bmm(intrinsics, points.view(b, 3, -1)).view(
            b, 3, d, h * w
        )  # [B, 3, D, H*W]
        pixel_coords = points[:, :2] / points[:, -1:].clamp(
            min=clamp_min_depth
        )  # [B, 2, D, H*W]

        # normalize to [-1, 1]
        x_grid = 2 * pixel_coords[:, 0] / (w - 1) - 1
        y_grid = 2 * pixel_coords[:, 1] / (h - 1) - 1

        grid = torch.stack([x_grid, y_grid], dim=-1)  # [B, D, H*W, 2]

    # sample features
    # ref: https://github.com/pytorch/pytorch/issues/88380
    # print(feature1.shape, grid.shape)
    # hardcoded workaround
    if feature1.numel() > 1000000:
        grid_sample_disable_cudnn = True
    with torch.backends.cudnn.flags(enabled=not grid_sample_disable_cudnn):
        warped_feature = F.grid_sample(
            feature1,
            grid.view(b, d * h, w, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).view(
            b, c, d, h, w
        )  # [B, C, D, H, W]

    return warped_feature


```

# prompts/papers/depthsplat/src/model/encoder/unimatch/mv_transformer.py

``` py
import torch
import torch.nn as nn
from einops import repeat

from .utils import split_feature, merge_splits


def single_head_full_attention(q, k, v):
    # q, k, v: [B, L, C]
    assert q.dim() == k.dim() == v.dim() == 3

    scores = torch.matmul(q, k.permute(0, 2, 1)) / (q.size(2) ** 0.5)  # [B, L, L]
    attn = torch.softmax(scores, dim=2)  # [B, L, L]
    out = torch.matmul(attn, v)  # [B, L, C]

    return out


def generate_shift_window_attn_mask(
    input_resolution,
    window_size_h,
    window_size_w,
    shift_size_h,
    shift_size_w,
    device=torch.device("cuda"),
):
    # Ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    # calculate attention mask for SW-MSA
    h, w = input_resolution
    img_mask = torch.zeros((1, h, w, 1)).to(device)  # 1 H W 1
    h_slices = (
        slice(0, -window_size_h),
        slice(-window_size_h, -shift_size_h),
        slice(-shift_size_h, None),
    )
    w_slices = (
        slice(0, -window_size_w),
        slice(-window_size_w, -shift_size_w),
        slice(-shift_size_w, None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = split_feature(
        img_mask, num_splits=input_resolution[-1] // window_size_w, channel_last=True
    )

    mask_windows = mask_windows.view(-1, window_size_h * window_size_w)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )

    return attn_mask


def single_head_split_window_attention(
    q,
    k,
    v,
    num_splits=1,
    with_shift=False,
    h=None,
    w=None,
    attn_mask=None,
):
    # Ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    # q, k, v: [B, L, C] for 2-view
    # for multi-view cross-attention, q: [B, L, C], k, v: [B, N-1, L, C]

    # multi(>2)-view corss-attention
    if not (q.dim() == k.dim() == v.dim() == 3):
        assert k.dim() == v.dim() == 4
        assert h is not None and w is not None
        assert q.size(1) == h * w

        m = k.size(1)  # m + 1 is num of views

        b, _, c = q.size()

        b_new = b * num_splits * num_splits

        window_size_h = h // num_splits
        window_size_w = w // num_splits

        q = q.view(b, h, w, c)  # [B, H, W, C]
        k = k.view(b, m, h, w, c)  # [B, N-1, H, W, C]
        v = v.view(b, m, h, w, c)

        scale_factor = c**0.5

        if with_shift:
            assert attn_mask is not None  # compute once
            shift_size_h = window_size_h // 2
            shift_size_w = window_size_w // 2

            q = torch.roll(q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
            k = torch.roll(k, shifts=(-shift_size_h, -shift_size_w), dims=(2, 3))
            v = torch.roll(v, shifts=(-shift_size_h, -shift_size_w), dims=(2, 3))

        q = split_feature(
            q, num_splits=num_splits, channel_last=True
        )  # [B*K*K, H/K, W/K, C]
        k = split_feature(
            k.permute(0, 2, 3, 4, 1).reshape(b, h, w, -1),
            num_splits=num_splits,
            channel_last=True,
        )  # [B*K*K, H/K, W/K, C*(N-1)]
        v = split_feature(
            v.permute(0, 2, 3, 4, 1).reshape(b, h, w, -1),
            num_splits=num_splits,
            channel_last=True,
        )  # [B*K*K, H/K, W/K, C*(N-1)]

        k = (
            k.view(b_new, h // num_splits, w // num_splits, c, m)
            .permute(0, 3, 1, 2, 4)
            .reshape(b_new, c, -1)
        )  # [B*K*K, C, H/K*W/K*(N-1)]
        v = (
            v.view(b_new, h // num_splits, w // num_splits, c, m)
            .permute(0, 1, 2, 4, 3)
            .reshape(b_new, -1, c)
        )  # [B*K*K, H/K*W/K*(N-1), C]

        scores = (
            torch.matmul(q.view(b_new, -1, c), k) / scale_factor
        )  # [B*K*K, H/K*W/K, H/K*W/K*(N-1)]

        if with_shift:
            scores += attn_mask.repeat(b, 1, m)

        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)  # [B*K*K, H/K*W/K, C]

        out = merge_splits(
            out.view(b_new, h // num_splits, w // num_splits, c),
            num_splits=num_splits,
            channel_last=True,
        )  # [B, H, W, C]

        # shift back
        if with_shift:
            out = torch.roll(out, shifts=(shift_size_h, shift_size_w), dims=(1, 2))

        out = out.view(b, -1, c)
    else:
        # 2-view self-attention or cross-attention
        assert q.dim() == k.dim() == v.dim() == 3

        assert h is not None and w is not None
        assert q.size(1) == h * w

        b, _, c = q.size()

        b_new = b * num_splits * num_splits

        window_size_h = h // num_splits
        window_size_w = w // num_splits

        q = q.view(b, h, w, c)  # [B, H, W, C]
        k = k.view(b, h, w, c)
        v = v.view(b, h, w, c)

        scale_factor = c**0.5

        if with_shift:
            assert attn_mask is not None  # compute once
            shift_size_h = window_size_h // 2
            shift_size_w = window_size_w // 2

            q = torch.roll(q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
            k = torch.roll(k, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
            v = torch.roll(v, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))

        q = split_feature(
            q, num_splits=num_splits, channel_last=True
        )  # [B*K*K, H/K, W/K, C]
        k = split_feature(k, num_splits=num_splits, channel_last=True)
        v = split_feature(v, num_splits=num_splits, channel_last=True)

        scores = (
            torch.matmul(q.view(b_new, -1, c), k.view(b_new, -1, c).permute(0, 2, 1))
            / scale_factor
        )  # [B*K*K, H/K*W/K, H/K*W/K]

        if with_shift:
            scores += attn_mask.repeat(b, 1, 1)

        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, v.view(b_new, -1, c))  # [B*K*K, H/K*W/K, C]

        out = merge_splits(
            out.view(b_new, h // num_splits, w // num_splits, c),
            num_splits=num_splits,
            channel_last=True,
        )  # [B, H, W, C]

        # shift back
        if with_shift:
            out = torch.roll(out, shifts=(shift_size_h, shift_size_w), dims=(1, 2))

        out = out.view(b, -1, c)

    return out


def multi_head_split_window_attention(
    q,
    k,
    v,
    num_splits=1,
    with_shift=False,
    h=None,
    w=None,
    attn_mask=None,
    num_head=1,
):
    """Multi-head scaled dot-product attention
    Args:
        q: [N, L, D]
        k: [N, S, D]
        v: [N, S, D]
    Returns:
        out: (N, L, D)
    """

    assert h is not None and w is not None
    assert q.size(1) == h * w

    b, _, c = q.size()

    b_new = b * num_splits * num_splits

    window_size_h = h // num_splits
    window_size_w = w // num_splits

    q = q.view(b, h, w, c)  # [B, H, W, C]
    k = k.view(b, h, w, c)
    v = v.view(b, h, w, c)

    assert c % num_head == 0

    scale_factor = (c // num_head) ** 0.5

    if with_shift:
        assert attn_mask is not None  # compute once
        shift_size_h = window_size_h // 2
        shift_size_w = window_size_w // 2

        q = torch.roll(q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        k = torch.roll(k, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        v = torch.roll(v, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))

    q = split_feature(q, num_splits=num_splits)  # [B*K*K, H/K, W/K, C]
    k = split_feature(k, num_splits=num_splits)
    v = split_feature(v, num_splits=num_splits)

    # multi-head attn
    q = q.view(b_new, -1, num_head, c // num_head).permute(0, 2, 1, 3)  # [B, N, H*W, C]
    k = k.view(b_new, -1, num_head, c // num_head).permute(0, 2, 3, 1)  # [B, N, C, H*W]
    scores = torch.matmul(q, k) / scale_factor  # [B*K*K, N, H/K*W/K, H/K*W/K]

    if with_shift:
        scores += attn_mask.unsqueeze(1).repeat(b, num_head, 1, 1)

    attn = torch.softmax(scores, dim=-1)  # [B*K*K, N, H/K*W/K, H/K*W/K]

    out = torch.matmul(
        attn, v.view(b_new, -1, num_head, c // num_head).permute(0, 2, 1, 3)
    )  # [B*K*K, N, H/K*W/K, C]

    out = merge_splits(
        out.permute(0, 2, 1, 3).reshape(b_new, h // num_splits, w // num_splits, c),
        num_splits=num_splits,
    )  # [B, H, W, C]

    # shift back
    if with_shift:
        out = torch.roll(out, shifts=(shift_size_h, shift_size_w), dims=(1, 2))

    out = out.view(b, -1, c)

    return out


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=1,
        attention_type="swin",
        no_ffn=False,
        ffn_dim_expansion=4,
        with_shift=False,
        add_per_view_attn=False,
        **kwargs,
    ):
        super(TransformerLayer, self).__init__()

        self.dim = d_model
        self.nhead = nhead
        self.attention_type = attention_type
        self.no_ffn = no_ffn
        self.add_per_view_attn = add_per_view_attn

        self.with_shift = with_shift

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.merge = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)

        # no ffn after self-attn, with ffn after cross-attn
        if not self.no_ffn:
            in_channels = d_model * 2
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                nn.GELU(),
                nn.Linear(in_channels * ffn_dim_expansion, d_model, bias=False),
            )

            self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        source,
        target,
        height=None,
        width=None,
        shifted_window_attn_mask=None,
        attn_num_splits=None,
        **kwargs,
    ):
        if "attn_type" in kwargs:
            attn_type = kwargs["attn_type"]
        else:
            attn_type = self.attention_type

        # source, target: [B, L, C] for 2-view
        # for multi-view cross-attention, source: [B, L, C], target: [B, N-1, L, C]
        query, key, value = source, target, target

        # single-head attention
        query = self.q_proj(query)  # [B, L, C]
        key = self.k_proj(key)  # [B, L, C] or [B, N-1, L, C]
        value = self.v_proj(value)  # [B, L, C] or [B, N-1, L, C]

        if attn_type == "swin" and attn_num_splits > 1:
            if self.nhead > 1:
                message = multi_head_split_window_attention(
                    query,
                    key,
                    value,
                    num_splits=attn_num_splits,
                    with_shift=self.with_shift,
                    h=height,
                    w=width,
                    attn_mask=shifted_window_attn_mask,
                    num_head=self.nhead,
                )
            else:
                if self.add_per_view_attn:
                    assert query.dim() == 3 and key.dim() == 4 and value.dim() == 4
                    b, l, c = query.size()
                    query = query.unsqueeze(1).repeat(
                        1, key.size(1), 1, 1
                    )  # [B, N-1, L, C]
                    query = query.view(-1, l, c)  # [B*(N-1), L, C]
                    key = key.view(-1, l, c)
                    value = value.view(-1, l, c)
                    message = single_head_split_window_attention(
                        query,
                        key,
                        value,
                        num_splits=attn_num_splits,
                        with_shift=self.with_shift,
                        h=height,
                        w=width,
                        attn_mask=shifted_window_attn_mask,
                    )
                    # [B, L, C]  # add per view attn
                    message = message.view(b, -1, l, c).sum(1)
                else:
                    message = single_head_split_window_attention(
                        query,
                        key,
                        value,
                        num_splits=attn_num_splits,
                        with_shift=self.with_shift,
                        h=height,
                        w=width,
                        attn_mask=shifted_window_attn_mask,
                    )
        else:
            message = single_head_full_attention(query, key, value)  # [B, L, C]

        message = self.merge(message)  # [B, L, C]
        message = self.norm1(message)

        if not self.no_ffn:
            message = self.mlp(torch.cat([source, message], dim=-1))
            message = self.norm2(message)

        return source + message


class TransformerBlock(nn.Module):
    """self attention + cross attention + FFN"""

    def __init__(
        self,
        d_model=256,
        nhead=1,
        attention_type="swin",
        ffn_dim_expansion=4,
        with_shift=False,
        add_per_view_attn=False,
        no_cross_attn=False,
        **kwargs,
    ):
        super(TransformerBlock, self).__init__()

        self.no_cross_attn = no_cross_attn

        if no_cross_attn:
            self.self_attn = TransformerLayer(
                d_model=d_model,
                nhead=nhead,
                attention_type=attention_type,
                ffn_dim_expansion=ffn_dim_expansion,
                with_shift=with_shift,
                add_per_view_attn=add_per_view_attn,
            )
        else:
            self.self_attn = TransformerLayer(
                d_model=d_model,
                nhead=nhead,
                attention_type=attention_type,
                no_ffn=True,
                ffn_dim_expansion=ffn_dim_expansion,
                with_shift=with_shift,
            )

            self.cross_attn_ffn = TransformerLayer(
                d_model=d_model,
                nhead=nhead,
                attention_type=attention_type,
                ffn_dim_expansion=ffn_dim_expansion,
                with_shift=with_shift,
                add_per_view_attn=add_per_view_attn,
            )

    def forward(
        self,
        source,
        target,
        height=None,
        width=None,
        shifted_window_attn_mask=None,
        attn_num_splits=None,
        **kwargs,
    ):
        # source, target: [B, L, C]
        # self attention
        source = self.self_attn(
            source,
            source,
            height=height,
            width=width,
            shifted_window_attn_mask=shifted_window_attn_mask,
            attn_num_splits=attn_num_splits,
            **kwargs,
        )

        if self.no_cross_attn:
            return source

        # cross attention and ffn
        source = self.cross_attn_ffn(
            source,
            target,
            height=height,
            width=width,
            shifted_window_attn_mask=shifted_window_attn_mask,
            attn_num_splits=attn_num_splits,
            **kwargs,
        )

        return source


def batch_features(features, nn_matrix=None):
    # construct inputs to multi-view transformer in batch
    # features: list of [B, C, H, W] or [B, H*W, C]

    # query, key and value for transformer
    q = []
    kv = []

    num_views = len(features)
    if nn_matrix is not None:
        # (b v c h w) or (b v hw c)
        features_tensor = torch.stack(features, dim=1)

    for i in range(num_views):
        x = features.copy()
        q.append(x.pop(i))  # [B, C, H, W] or [B, H*W, C]

        # [B, N-1, C, H, W] or [B, N-1, H*W, C]
        if nn_matrix is not None:
            # select views based on the provided nn matrix
            if features_tensor.dim() == 5:
                c, h, w = features_tensor.shape[-3:]
                index = repeat(nn_matrix[:, i, 1:], "b v -> b v c h w", c=c, h=h, w=w)
            elif features_tensor.dim() == 4:
                hw, c = features_tensor.shape[-2:]
                index = repeat(nn_matrix[:, i, 1:], "b v -> b v hw c", hw=hw, c=c)

            kv_x = torch.gather(features_tensor, dim=1, index=index)
        else:
            kv_x = torch.stack(x, dim=1)
        kv.append(kv_x)

    q = torch.cat(q, dim=0)  # [N*B, C, H, W] or [N*B, H*W, C]
    kv = torch.cat(kv, dim=0)  # [N*B, N-1, C, H, W] or [N*B, N-1, H*W, C]

    return q, kv


class MultiViewFeatureTransformer(nn.Module):
    def __init__(
        self,
        num_layers=6,
        d_model=128,
        nhead=1,
        attention_type="swin",
        ffn_dim_expansion=4,
        add_per_view_attn=False,
        no_cross_attn=False,
        **kwargs,
    ):
        super(MultiViewFeatureTransformer, self).__init__()

        self.attention_type = attention_type

        self.d_model = d_model
        self.nhead = nhead

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    nhead=nhead,
                    attention_type=attention_type,
                    ffn_dim_expansion=ffn_dim_expansion,
                    with_shift=(
                        True if attention_type == "swin" and i % 2 == 1 else False
                    ),
                    add_per_view_attn=add_per_view_attn,
                    no_cross_attn=no_cross_attn,
                )
                for i in range(num_layers)
            ]
        )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # zero init layers beyond 6
        if num_layers > 6:
            for i in range(6, num_layers):
                self.layers[i].self_attn.norm1.weight.data.zero_()
                self.layers[i].self_attn.norm1.bias.data.zero_()
                self.layers[i].cross_attn_ffn.norm2.weight.data.zero_()
                self.layers[i].cross_attn_ffn.norm2.bias.data.zero_()

    def forward(
        self,
        multi_view_features,
        attn_num_splits=None,
        **kwargs,
    ):
        nn_matrix = kwargs.pop("nn_matrix", None)

        # multi_view_features: list of [B, C, H, W]
        b, c, h, w = multi_view_features[0].shape
        assert self.d_model == c

        num_views = len(multi_view_features)

        if self.attention_type == "swin" and attn_num_splits > 1:
            # global and refine use different number of splits
            window_size_h = h // attn_num_splits
            window_size_w = w // attn_num_splits

            # compute attn mask once
            shifted_window_attn_mask = generate_shift_window_attn_mask(
                input_resolution=(h, w),
                window_size_h=window_size_h,
                window_size_w=window_size_w,
                shift_size_h=window_size_h // 2,
                shift_size_w=window_size_w // 2,
                device=multi_view_features[0].device,
            )  # [K*K, H/K*W/K, H/K*W/K]
        else:
            shifted_window_attn_mask = None

        # [N*B, C, H, W], [N*B, N-1, C, H, W]
        concat0, concat1 = batch_features(multi_view_features, nn_matrix=nn_matrix)
        concat0 = concat0.reshape(num_views * b, c, -1).permute(
            0, 2, 1
        )  # [N*B, H*W, C]
        c1_v = num_views - 1 if nn_matrix is None else nn_matrix.shape[-1] - 1
        concat1 = concat1.reshape(num_views * b, c1_v, c, -1).permute(
            0, 1, 3, 2
        )  # [N*B, N-1, H*W, C]

        for i, layer in enumerate(self.layers):
            concat0 = layer(
                concat0,
                concat1,
                height=h,
                width=w,
                shifted_window_attn_mask=shifted_window_attn_mask,
                attn_num_splits=attn_num_splits,
            )

            if i < len(self.layers) - 1:
                # list of features
                features = list(concat0.chunk(chunks=num_views, dim=0))
                # [N*B, H*W, C], [N*B, N-1, H*W, C]
                concat0, concat1 = batch_features(features, nn_matrix=nn_matrix)

        features = concat0.chunk(chunks=num_views, dim=0)
        features = [
            f.view(b, h, w, c).permute(0, 3, 1, 2).contiguous() for f in features
        ]

        return features


def batch_features_camera_parameters(
    features,
    intrinsics,
    extrinsics,
    nn_matrix=None,
    no_batch=False,
):
    # construct inputs for warping with plane-sweep stereo
    # features: list of [B, C, H, W]
    # intrinsics: list of [B, 3, 3]
    # extrinsics: list of [B, 4, 4]

    assert (
        features[0].dim() == 4 and intrinsics[0].dim() == 3 and extrinsics[0].dim() == 3
    )
    assert intrinsics[0].size(-1) == intrinsics[0].size(-2) == 3
    assert extrinsics[0].size(-1) == extrinsics[0].size(-2) == 4

    # query, key and value for transformer
    q = []
    q_intrinsics = []
    q_extrinsics = []
    kv = []
    kv_intrinsics = []
    kv_extrinsics = []

    num_views = len(features)
    if nn_matrix is not None:
        features_tensor = torch.stack(features, dim=1)  # [B, V, C, H, W]
        intrinsics_tensor = torch.stack(intrinsics, dim=1)  # [B, V, 3, 3]
        extrinsics_tensor = torch.stack(extrinsics, dim=1)  # [B, V, 4, 4]

        num_selected_views = nn_matrix.size(-1) - 1
    else:
        num_selected_views = num_views - 1

    for i in range(num_views):
        # features
        x = features.copy()
        q.append(x.pop(i))  # [B, C, H, W]

        # camera
        y = intrinsics.copy()
        q_intrinsics.append(y.pop(i))
        z = extrinsics.copy()
        q_extrinsics.append(z.pop(i))

        # [B, V-1, C, H, W]
        if nn_matrix is not None:
            # select views based on the provided nn matrix
            if features_tensor.dim() == 5:
                c, h, w = features_tensor.shape[-3:]
                index = repeat(nn_matrix[:, i, 1:], "b v -> b v c h w", c=c, h=h, w=w)
            elif features_tensor.dim() == 4:
                hw, c = features_tensor.shape[-2:]
                index = repeat(nn_matrix[:, i, 1:], "b v -> b v hw c", hw=hw, c=c)

            kv_x = torch.gather(features_tensor, dim=1, index=index)

            # select intrinsics and extrinsics
            index = repeat(nn_matrix[:, i, 1:], "b v -> b v 3 3")
            kv_y_intrinsics = torch.gather(intrinsics_tensor, dim=1, index=index)

            index = repeat(nn_matrix[:, i, 1:], "b v -> b v 4 4")
            kv_z_extrinsics = torch.gather(extrinsics_tensor, dim=1, index=index)

        else:
            kv_x = torch.stack(x, dim=1)
            kv_y_intrinsics = torch.stack(y, dim=1)
            kv_z_extrinsics = torch.stack(z, dim=1)

        kv.append(kv_x)
        kv_intrinsics.append(kv_y_intrinsics)
        kv_extrinsics.append(kv_z_extrinsics)

    if no_batch:
        # list of [B, C, H, W]
        return q, q_intrinsics, q_extrinsics, kv, kv_intrinsics, kv_extrinsics

    c, h, w = q[0].shape[1:]

    q = torch.stack(q, dim=1).view(-1, c, h, w)  # [BV, C, H, W]
    q_intrinsics = torch.stack(q_intrinsics, dim=1).view(-1, 3, 3)  # [BV, 3, 3]
    q_extrinsics = torch.stack(q_extrinsics, dim=1).view(-1, 4, 4)  # [BV, 4, 4]
    kv = torch.stack(kv, dim=1).view(
        -1, num_selected_views, c, h, w
    )  # [BV, V-1, C, H, W]
    kv_intrinsics = torch.stack(kv_intrinsics, dim=1).view(
        -1, num_selected_views, 3, 3
    )  # [BV, V-1, 3, 3]
    kv_extrinsics = torch.stack(kv_extrinsics, dim=1).view(
        -1, num_selected_views, 4, 4
    )  # [BV, V-1, 4, 4]

    return q, q_intrinsics, q_extrinsics, kv, kv_intrinsics, kv_extrinsics


```

# prompts/papers/depthsplat/src/model/encoder/unimatch/mv_unimatch.py

``` py
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import CNNEncoder
from .vit_fpn import ViTFeaturePyramid
from .mv_transformer import (
    MultiViewFeatureTransformer,
    batch_features_camera_parameters,
)
from .matching import warp_with_pose_depth_candidates
from .utils import mv_feature_add_position
from .dpt_head import DPTHead
from .ldm_unet.unet import UNetModel, AttentionBlock
from einops import rearrange


class MultiViewUniMatch(nn.Module):
    def __init__(
        self,
        num_scales=1,
        feature_channels=128,
        upsample_factor=8,
        lowest_feature_resolution=8,
        num_head=1,
        ffn_dim_expansion=4,
        num_transformer_layers=6,
        num_depth_candidates=128,
        vit_type="vits",
        unet_channels=128,
        unet_channel_mult=[1, 1, 1],
        unet_num_res_blocks=1,
        unet_attn_resolutions=[4],
        grid_sample_disable_cudnn=False,
        **kwargs,
    ):
        super(MultiViewUniMatch, self).__init__()

        # CNN
        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.lowest_feature_resolution = lowest_feature_resolution
        self.upsample_factor = upsample_factor

        # monocular backbones: final
        self.vit_type = vit_type

        # cost volume
        self.num_depth_candidates = num_depth_candidates

        # upsampler
        vit_feature_channel_dict = {"vits": 384, "vitb": 768, "vitl": 1024}

        vit_feature_channel = vit_feature_channel_dict[vit_type]

        # CNN
        self.backbone = CNNEncoder(
            output_dim=feature_channels,
            num_output_scales=num_scales,
            downsample_factor=upsample_factor,
            lowest_scale=lowest_feature_resolution,
            return_all_scales=True,
        )

        # Transformer
        self.transformer = MultiViewFeatureTransformer(
            num_layers=num_transformer_layers,
            d_model=feature_channels,
            nhead=num_head,
            ffn_dim_expansion=ffn_dim_expansion,
        )

        if self.num_scales > 1:
            # generate multi-scale features
            self.mv_pyramid = ViTFeaturePyramid(
                in_channels=128, scale_factors=[2**i for i in range(self.num_scales)]
            )

        # monodepth
        encoder = vit_type  # can also be 'vitb' or 'vitl'
        self.pretrained = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_{:}14".format(encoder)
        )

        del self.pretrained.mask_token  # unused

        if self.num_scales > 1:
            # generate multi-scale features
            self.mono_pyramid = ViTFeaturePyramid(
                in_channels=vit_feature_channel,
                scale_factors=[2**i for i in range(self.num_scales)],
            )

        # UNet regressor
        self.regressor = nn.ModuleList()
        self.regressor_residual = nn.ModuleList()
        self.depth_head = nn.ModuleList()

        for i in range(self.num_scales):
            curr_depth_candidates = num_depth_candidates // (4**i)
            cnn_feature_channels = 128 - (32 * i)
            mv_transformer_feature_channels = 128 // (2**i)

            mono_feature_channels = vit_feature_channel // (2**i)

            # concat(cost volume, cnn feature, mv feature, mono feature)
            in_channels = (
                curr_depth_candidates
                + cnn_feature_channels
                + mv_transformer_feature_channels
                + mono_feature_channels
            )

            # unet channels
            channels = unet_channels // (2**i)

            # unet channel mult & unet_attn_resolutions
            if i > 0:
                unet_channel_mult = unet_channel_mult + [1]
                unet_attn_resolutions = [x * 2 for x in unet_attn_resolutions]

            # unet
            modules = [
                nn.Conv2d(in_channels, channels, 3, 1, 1),
                nn.GroupNorm(8, channels),
                nn.GELU(),
            ]

            modules.append(
                UNetModel(
                    image_size=None,
                    in_channels=channels,
                    model_channels=channels,
                    out_channels=channels,
                    num_res_blocks=unet_num_res_blocks,
                    attention_resolutions=unet_attn_resolutions,
                    channel_mult=unet_channel_mult,
                    num_head_channels=32,
                    dims=2,
                    postnorm=False,
                    num_frames=2,
                    use_cross_view_self_attn=True,
                )
            )

            modules.append(nn.Conv2d(channels, channels, 3, 1, 1))

            self.regressor.append(nn.Sequential(*modules))

            # regressor residual
            self.regressor_residual.append(nn.Conv2d(in_channels, channels, 1))

            # depth head
            self.depth_head.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels, channels * 2, 3, 1, 1, padding_mode="replicate"
                    ),
                    nn.GELU(),
                    nn.Conv2d(
                        channels * 2,
                        curr_depth_candidates,
                        3,
                        1,
                        1,
                        padding_mode="replicate",
                    ),
                )
            )

        # upsampler
        # concat(lowres_depth, cnn feature, mv feature, mono feature)
        in_channels = (
            1
            + cnn_feature_channels
            + mv_transformer_feature_channels
            + mono_feature_channels
        )

        model_configs = {
            "vits": {
                "in_channels": 384,
                "features": 32,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "in_channels": 768,
                "features": 48,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "in_channels": 1024,
                "features": 64,
                "out_channels": [128, 256, 512, 1024],
            },
        }

        self.upsampler = DPTHead(
            **model_configs[vit_type],
            downsample_factor=upsample_factor,
            num_scales=num_scales,
        )

        self.grid_sample_disable_cudnn = grid_sample_disable_cudnn

    def normalize_images(self, images):
        """Normalize image to match the pretrained UniMatch model.
        images: (B, V, C, H, W)
        """
        shape = [*[1] * (images.dim() - 3), 3, 1, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(*shape).to(images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(*shape).to(images.device)

        return (images - mean) / std

    def extract_feature(self, images):
        # images: [B, V, C, H, W]
        b, v = images.shape[:2]
        concat = rearrange(images, "b v c h w -> (b v) c h w")
        # list of [BV, C, H, W], resolution from high to low
        features = self.backbone(concat)
        # reverse: resolution from low to high
        features = features[::-1]

        return features

    def forward(
        self,
        images,
        attn_splits_list=None,
        intrinsics=None,
        min_depth=1.0 / 0.5,  # inverse depth range
        max_depth=1.0 / 100,
        num_depth_candidates=128,
        extrinsics=None,
        nn_matrix=None,
        **kwargs,
    ):

        results_dict = {}
        depth_preds = []
        match_probs = []

        # first normalize images
        images = self.normalize_images(images)
        b, v, _, ori_h, ori_w = images.shape

        # update the num_views in unet attention, useful for random input views
        set_num_views(self.regressor, num_views=v)

        # NOTE: in this codebase, intrinsics are normalized by image width and height
        # in unimatch's codebase: https://github.com/autonomousvision/unimatch, no normalization
        intrinsics = intrinsics.clone()
        intrinsics[:, :, 0] *= ori_w
        intrinsics[:, :, 1] *= ori_h

        # max_depth, min_depth: [B, V] -> [BV]
        max_depth = max_depth.view(-1)
        min_depth = min_depth.view(-1)

        # list of features, resolution low to high
        # list of [BV, C, H, W]
        features_list_cnn = self.extract_feature(images)
        features_list_cnn_all_scales = features_list_cnn
        features_list_cnn = features_list_cnn[: self.num_scales]
        results_dict.update({"features_cnn_all_scales": features_list_cnn_all_scales})
        results_dict.update({"features_cnn": features_list_cnn})

        # mv transformer features
        # add position to features
        attn_splits = attn_splits_list[0]

        # [BV, C, H, W]
        features_cnn_pos = mv_feature_add_position(
            features_list_cnn[0], attn_splits, self.feature_channels
        )

        # list of [B, C, H, W]
        features_list = list(
            torch.unbind(
                rearrange(features_cnn_pos, "(b v) c h w -> b v c h w", b=b, v=v), dim=1
            )
        )
        features_list_mv = self.transformer(
            features_list,
            attn_num_splits=attn_splits,
            nn_matrix=nn_matrix,
        )

        features_mv = rearrange(
            torch.stack(features_list_mv, dim=1), "b v c h w -> (b v) c h w"
        )  # [BV, C, H, W]

        if self.num_scales > 1:
            # multi-scale mv features: resolution from low to high
            # list of [BV, C, H, W]
            features_list_mv = self.mv_pyramid(features_mv)
        else:
            features_list_mv = [features_mv]

        results_dict.update({"features_mv": features_list_mv})

        # mono feature
        ori_h, ori_w = images.shape[-2:]
        resize_h, resize_w = ori_h // 14 * 14, ori_w // 14 * 14
        concat = rearrange(images, "b v c h w -> (b v) c h w")
        concat = F.interpolate(
            concat, (resize_h, resize_w), mode="bilinear", align_corners=True
        )

        # get intermediate features
        intermediate_layer_idx = {
            "vits": [2, 5, 8, 11],
            "vitb": [2, 5, 8, 11],
            "vitl": [4, 11, 17, 23],
        }

        mono_intermediate_features = list(
            self.pretrained.get_intermediate_layers(
                concat, intermediate_layer_idx[self.vit_type], return_class_token=False
            )
        )

        for i in range(len(mono_intermediate_features)):
            curr_features = (
                mono_intermediate_features[i]
                .reshape(concat.shape[0], resize_h // 14, resize_w // 14, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            # resize to 1/8 resolution
            curr_features = F.interpolate(
                curr_features,
                (ori_h // 8, ori_w // 8),
                mode="bilinear",
                align_corners=True,
            )
            mono_intermediate_features[i] = curr_features

        results_dict.update({"features_mono_intermediate": mono_intermediate_features})

        # last mono feature
        mono_features = mono_intermediate_features[-1]

        if self.lowest_feature_resolution == 4:
            mono_features = F.interpolate(
                mono_features, scale_factor=2, mode="bilinear", align_corners=True
            )

        if self.num_scales > 1:
            # multi-scale mono features, resolution from low to high
            # list of [BV, C, H, W]
            features_list_mono = self.mono_pyramid(mono_features)
        else:
            features_list_mono = [mono_features]

        results_dict.update({"features_mono": features_list_mono})

        depth = None

        for scale_idx in range(self.num_scales):
            downsample_factor = self.upsample_factor * (
                2 ** (self.num_scales - 1 - scale_idx)
            )

            # scale intrinsics
            intrinsics_curr = intrinsics.clone()  # [B, V, 3, 3]
            intrinsics_curr[:, :, :2] = intrinsics_curr[:, :, :2] / downsample_factor

            # build cost volume
            features_mv = features_list_mv[scale_idx]  # [BV, C, H, W]

            # list of [B, C, H, W]
            features_mv_curr = list(
                torch.unbind(
                    rearrange(features_mv, "(b v) c h w -> b v c h w", b=b, v=v), dim=1
                )
            )

            intrinsics_curr = list(
                torch.unbind(intrinsics_curr, dim=1)
            )  # list of [B, 3, 3]
            extrinsics_curr = list(torch.unbind(extrinsics, dim=1))  # list of [B, 4, 4]

            # ref: [BV, C, H, W], [BV, 3, 3], [BV, 4, 4]
            # tgt: [BV, V-1, C, H, W], [BV, V-1, 3, 3], [BV, V-1, 4, 4]
            (
                ref_features,
                ref_intrinsics,
                ref_extrinsics,
                tgt_features,
                tgt_intrinsics,
                tgt_extrinsics,
            ) = batch_features_camera_parameters(
                features_mv_curr,
                intrinsics_curr,
                extrinsics_curr,
                nn_matrix=nn_matrix,
            )

            b_new, _, c, h, w = tgt_features.size()

            # relative pose
            # extrinsics: c2w
            pose_curr = torch.matmul(
                tgt_extrinsics.inverse(), ref_extrinsics.unsqueeze(1)
            )  # [BV, V-1, 4, 4]

            if scale_idx > 0:
                # 2x upsample depth
                assert depth is not None
                depth = F.interpolate(
                    depth, scale_factor=2, mode="bilinear", align_corners=True
                ).detach()

            num_depth_candidates = self.num_depth_candidates // (4**scale_idx)

            # generate depth candidates
            if scale_idx == 0:
                # min_depth, max_depth: [BV]
                depth_interval = (max_depth - min_depth) / (
                    self.num_depth_candidates - 1
                )  # [BV]

                linear_space = (
                    torch.linspace(0, 1, num_depth_candidates)
                    .type_as(features_list_cnn[0])
                    .view(1, num_depth_candidates, 1, 1)
                )  # [1, D, 1, 1]

                depth_candidates = min_depth.view(-1, 1, 1, 1) + linear_space * (
                    max_depth - min_depth
                ).view(
                    -1, 1, 1, 1
                )  # [BV, D, 1, 1]
            else:
                # half interval each scale
                depth_interval = (
                    (max_depth - min_depth)
                    / (self.num_depth_candidates - 1)
                    / (2**scale_idx)
                )  # [BV]
                # [BV, 1, 1, 1]
                depth_interval = depth_interval.view(-1, 1, 1, 1)

                # [BV, 1, H, W]
                depth_range_min = (
                    depth - depth_interval * (num_depth_candidates // 2)
                ).clamp(min=min_depth.view(-1, 1, 1, 1))
                depth_range_max = (
                    depth + depth_interval * (num_depth_candidates // 2 - 1)
                ).clamp(max=max_depth.view(-1, 1, 1, 1))

                linear_space = (
                    torch.linspace(0, 1, num_depth_candidates)
                    .type_as(features_list_cnn[0])
                    .view(1, num_depth_candidates, 1, 1)
                )  # [1, D, 1, 1]
                depth_candidates = depth_range_min + linear_space * (
                    depth_range_max - depth_range_min
                )  # [BV, D, H, W]

            if scale_idx == 0:
                # [BV*(V-1), D, H, W]
                depth_candidates_curr = (
                    depth_candidates.unsqueeze(1)
                    .repeat(1, tgt_features.size(1), 1, h, w)
                    .view(-1, num_depth_candidates, h, w)
                )
            else:
                depth_candidates_curr = (
                    depth_candidates.unsqueeze(1)
                    .repeat(1, tgt_features.size(1), 1, 1, 1)
                    .view(-1, num_depth_candidates, h, w)
                )

            intrinsics_input = torch.stack(intrinsics_curr, dim=1).view(
                -1, 3, 3
            )  # [BV, 3, 3]
            intrinsics_input = intrinsics_input.unsqueeze(1).repeat(
                1, tgt_features.size(1), 1, 1
            )  # [BV, V-1, 3, 3]

            warped_tgt_features = warp_with_pose_depth_candidates(
                rearrange(tgt_features, "b v ... -> (b v) ..."),
                rearrange(intrinsics_input, "b v ... -> (b v) ..."),
                rearrange(pose_curr, "b v ... -> (b v) ..."),
                1.0 / depth_candidates_curr,  # convert inverse depth to depth
                grid_sample_disable_cudnn=self.grid_sample_disable_cudnn,
            )  # [BV*(V-1), C, D, H, W]

            # ref: [BV, C, H, W]
            # warped: [BV*(V-1), C, D, H, W] -> [BV, V-1, C, D, H, W]
            warped_tgt_features = rearrange(
                warped_tgt_features,
                "(b v) ... -> b v ...",
                b=b_new,
                v=tgt_features.size(1),
            )
            # [BV, V-1, D, H, W] -> [BV, D, H, W]
            # average cross other views
            cost_volume = (
                (ref_features.unsqueeze(-3).unsqueeze(1) * warped_tgt_features).sum(2)
                / (c**0.5)
            ).mean(1)

            # regressor
            features_cnn = features_list_cnn[scale_idx]  # [BV, C, H, W]

            features_mono = features_list_mono[scale_idx]  # [BV, C, H, W]

            concat = torch.cat(
                (cost_volume, features_cnn, features_mv, features_mono), dim=1
            )

            out = self.regressor[scale_idx](concat) + self.regressor_residual[
                scale_idx
            ](concat)

            # depth pred
            match_prob = F.softmax(
                self.depth_head[scale_idx](out), dim=1
            )  # [BV, D, H, W]
            match_probs.append(match_prob)

            if scale_idx == 0:
                # [BV, D, H, W]
                depth_candidates = depth_candidates.repeat(1, 1, h, w)
            depth = (match_prob * depth_candidates).sum(
                dim=1, keepdim=True
            )  # [BV, 1, H, W]

            # upsample to the original resolution for supervison at training time only
            if self.training and scale_idx < self.num_scales - 1:
                depth_bilinear = F.interpolate(
                    depth,
                    scale_factor=downsample_factor,
                    mode="bilinear",
                    align_corners=True,
                )
                depth_preds.append(depth_bilinear)

            # final output, learned upsampler
            if scale_idx == self.num_scales - 1:
                residual_depth = self.upsampler(
                    mono_intermediate_features,
                    # resolution high to low
                    cnn_features=features_list_cnn_all_scales[::-1],
                    mv_features=(
                        features_mv if self.num_scales == 1 else features_list_mv[::-1]
                    ),
                    depth=depth,
                )

                depth_bilinear = F.interpolate(
                    depth,
                    scale_factor=self.upsample_factor,
                    mode="bilinear",
                    align_corners=True,
                )
                depth = (depth_bilinear + residual_depth).clamp(
                    min=min_depth.view(-1, 1, 1, 1), max=max_depth.view(-1, 1, 1, 1)
                )

                depth_preds.append(depth)

        # convert inverse depth to depth
        for i in range(len(depth_preds)):
            depth_pred = 1.0 / depth_preds[i].squeeze(1)  # [BV, H, W]
            depth_preds[i] = rearrange(
                depth_pred, "(b v) ... -> b v ...", b=b, v=v
            )  # [B, V, H, W]

        results_dict.update({"depth_preds": depth_preds})
        results_dict.update({"match_probs": match_probs})

        return results_dict


def set_num_views(module, num_views):
    if isinstance(module, AttentionBlock):
        module.attention.n_frames = num_views
    elif (
        isinstance(module, nn.ModuleList)
        or isinstance(module, nn.Sequential)
        or isinstance(module, nn.Module)
    ):
        for submodule in module.children():
            set_num_views(submodule, num_views)


```

# prompts/papers/depthsplat/src/model/encoder/unimatch/position.py

``` py
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# https://github.com/facebookresearch/detr/blob/main/models/position_encoding.py

import torch
import torch.nn as nn
import math


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        # x = tensor_list.tensors  # [B, C, H, W]
        # mask = tensor_list.mask  # [B, H, W], input with padding, valid as 0
        b, c, h, w = x.size()
        mask = torch.ones((b, h, w), device=x.device)  # [B, H, W]
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


```

# prompts/papers/depthsplat/src/model/encoder/unimatch/utils.py

``` py
import torch
import torch.nn.functional as F
from .position import PositionEmbeddingSine


def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    assert device is not None

    x, y = torch.meshgrid(
        [
            torch.linspace(w_min, w_max, len_w, device=device),
            torch.linspace(h_min, h_max, len_h, device=device),
        ],
    )
    grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]

    return grid


def normalize_coords(coords, h, w):
    # coords: [B, H, W, 2]
    c = torch.Tensor([(w - 1) / 2.0, (h - 1) / 2.0]).float().to(coords.device)
    return (coords - c) / c  # [-1, 1]


def normalize_img(img0, img1):
    # loaded images are in [0, 255]
    # normalize by ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img1.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img1.device)
    img0 = (img0 / 255.0 - mean) / std
    img1 = (img1 / 255.0 - mean) / std

    return img0, img1


def split_feature(
    feature,
    num_splits=2,
    channel_last=False,
):
    if channel_last:  # [B, H, W, C]
        b, h, w, c = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = (
            feature.view(b, num_splits, h // num_splits, num_splits, w // num_splits, c)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(b_new, h_new, w_new, c)
        )  # [B*K*K, H/K, W/K, C]
    else:  # [B, C, H, W]
        b, c, h, w = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = (
            feature.view(b, c, num_splits, h // num_splits, num_splits, w // num_splits)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(b_new, c, h_new, w_new)
        )  # [B*K*K, C, H/K, W/K]

    return feature


def merge_splits(
    splits,
    num_splits=2,
    channel_last=False,
):
    if channel_last:  # [B*K*K, H/K, W/K, C]
        b, h, w, c = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, h, w, c)
        merge = (
            splits.permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(new_b, num_splits * h, num_splits * w, c)
        )  # [B, H, W, C]
    else:  # [B*K*K, C, H/K, W/K]
        b, c, h, w = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, c, h, w)
        merge = (
            splits.permute(0, 3, 1, 4, 2, 5)
            .contiguous()
            .view(new_b, c, num_splits * h, num_splits * w)
        )  # [B, C, H, W]

    return merge


def generate_shift_window_attn_mask(
    input_resolution,
    window_size_h,
    window_size_w,
    shift_size_h,
    shift_size_w,
    device=torch.device("cuda"),
):
    # ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    # calculate attention mask for SW-MSA
    h, w = input_resolution
    img_mask = torch.zeros((1, h, w, 1)).to(device)  # 1 H W 1
    h_slices = (
        slice(0, -window_size_h),
        slice(-window_size_h, -shift_size_h),
        slice(-shift_size_h, None),
    )
    w_slices = (
        slice(0, -window_size_w),
        slice(-window_size_w, -shift_size_w),
        slice(-shift_size_w, None),
    )
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = split_feature(
        img_mask, num_splits=input_resolution[-1] // window_size_w, channel_last=True
    )

    mask_windows = mask_windows.view(-1, window_size_h * window_size_w)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )

    return attn_mask


def feature_add_position(feature0, feature1, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    if attn_splits > 1:  # add position in splited window
        feature0_splits = split_feature(feature0, num_splits=attn_splits)
        feature1_splits = split_feature(feature1, num_splits=attn_splits)

        position = pos_enc(feature0_splits)

        feature0_splits = feature0_splits + position
        feature1_splits = feature1_splits + position

        feature0 = merge_splits(feature0_splits, num_splits=attn_splits)
        feature1 = merge_splits(feature1_splits, num_splits=attn_splits)
    else:
        position = pos_enc(feature0)

        feature0 = feature0 + position
        feature1 = feature1 + position

    return feature0, feature1


def mv_feature_add_position(features, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

    assert features.dim() == 4  # [B*V, C, H, W]

    if attn_splits > 1:  # add position in splited window
        features_splits = split_feature(features, num_splits=attn_splits)
        position = pos_enc(features_splits)
        features_splits = features_splits + position
        features = merge_splits(features_splits, num_splits=attn_splits)
    else:
        position = pos_enc(features)
        features = features + position

    return features


```

# prompts/papers/depthsplat/src/model/encoder/unimatch/vit_fpn.py

``` py
import torch
import torch.nn as nn
import torch.nn.functional as F


# Ref: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py#L363


class ViTFeaturePyramid(nn.Module):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        in_channels,
        scale_factors,
    ):
        """
        Args:
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
        """
        super(ViTFeaturePyramid, self).__init__()

        self.scale_factors = scale_factors

        out_dim = dim = in_channels
        self.stages = nn.ModuleList()
        for idx, scale in enumerate(scale_factors):
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            if scale != 1.0:
                layers.extend(
                    [
                        nn.GELU(),
                        nn.Conv2d(out_dim, out_dim, 3, 1, 1),
                    ]
                )
            layers = nn.Sequential(*layers)

            self.stages.append(layers)

    def forward(self, x):
        results = []

        for stage in self.stages:
            results.append(stage(x))

        return results


def _test():
    model = ViTFeaturePyramid(
        384,
        scale_factors=[1, 2, 4],
    ).cuda()
    print(model)

    x = torch.randn(2, 384, 64, 96).cuda()

    out = model(x)

    for x in out:
        print(x.shape)


if __name__ == "__main__":
    _test()


```

# prompts/papers/depthsplat/src/model/encoder/visualization/encoder_visualizer.py

``` py
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from jaxtyping import Float
from torch import Tensor

T_cfg = TypeVar("T_cfg")
T_encoder = TypeVar("T_encoder")


class EncoderVisualizer(ABC, Generic[T_cfg, T_encoder]):
    cfg: T_cfg
    encoder: T_encoder

    def __init__(self, cfg: T_cfg, encoder: T_encoder) -> None:
        self.cfg = cfg
        self.encoder = encoder

    @abstractmethod
    def visualize(
        self,
        context: dict,
        global_step: int,
    ) -> dict[str, Float[Tensor, "3 _ _"]]:
        pass


```

# prompts/papers/depthsplat/src/model/encoder/visualization/encoder_visualizer_depthsplat.py

``` py
from pathlib import Path
from random import randrange
from typing import Optional

import numpy as np
import torch
import wandb
from einops import rearrange, reduce, repeat
from jaxtyping import Bool, Float
from torch import Tensor

from ....dataset.types import BatchedViews
from ....misc.heterogeneous_pairings import generate_heterogeneous_index
from ....visualization.annotation import add_label
from ....visualization.color_map import apply_color_map, apply_color_map_to_image
from ....visualization.colors import get_distinct_color
from ....visualization.drawing.lines import draw_lines
from ....visualization.drawing.points import draw_points
from ....visualization.layout import add_border, hcat, vcat
# from ...ply_export import export_ply
from ..encoder_depthsplat import EncoderDepthSplat
# from ..epipolar.epipolar_sampler import EpipolarSampling
from .encoder_visualizer import EncoderVisualizer
from .encoder_visualizer_depthsplat_cfg import EncoderVisualizerDepthSplatCfg


def box(
    image: Float[Tensor, "3 height width"],
) -> Float[Tensor, "3 new_height new_width"]:
    return add_border(add_border(image), 1, 0)


class EncoderVisualizerDepthSplat(
    EncoderVisualizer[EncoderVisualizerDepthSplatCfg, EncoderDepthSplat]
):
    def visualize(
        self,
        context: BatchedViews,
        global_step: int,
    ) -> dict[str, Float[Tensor, "3 _ _"]]:
        # Short-circuit execution when using mvsplat.
        return {}

        visualization_dump = {}

        softmax_weights = []

        def hook(module, input, output):
            softmax_weights.append(output)

        # Register hooks to grab attention.
        handles = [
            layer[0].fn.attend.register_forward_hook(hook)
            for layer in self.encoder.epipolar_transformer.transformer.layers
        ]

        result = self.encoder.forward(
            context,
            global_step,
            visualization_dump=visualization_dump,
            deterministic=True,
        )

        # De-register hooks.
        for handle in handles:
            handle.remove()

        softmax_weights = torch.stack(softmax_weights)

        # Generate high-resolution context images that can be drawn on.
        context_images = context["image"]
        _, _, _, h, w = context_images.shape
        length = min(h, w)
        min_resolution = self.cfg.min_resolution
        scale_multiplier = (min_resolution + length - 1) // length
        if scale_multiplier > 1:
            context_images = repeat(
                context_images,
                "b v c h w -> b v c (h rh) (w rw)",
                rh=scale_multiplier,
                rw=scale_multiplier,
            )

        # This is kind of hacky for now, since we're using it for short experiments.
        if self.cfg.export_ply and wandb.run is not None:
            name = wandb.run._name.split(" ")[0]
            ply_path = Path(f"outputs/gaussians/{name}/{global_step:0>6}.ply")
            export_ply(
                context["extrinsics"][0, 0],
                result.means[0],
                visualization_dump["scales"][0],
                visualization_dump["rotations"][0],
                result.harmonics[0],
                result.opacities[0],
                ply_path,
            )

        return {
            "attention": self.visualize_attention(
                context_images,
                visualization_dump["sampling"],
                softmax_weights,
            ),
            "epipolar_samples": self.visualize_epipolar_samples(
                context_images,
                visualization_dump["sampling"],
            ),
            "epipolar_color_samples": self.visualize_epipolar_color_samples(
                context_images,
                context,
            ),
            "gaussians": self.visualize_gaussians(
                context["image"],
                result.opacities,
                result.covariances,
                result.harmonics[..., 0],  # Just visualize DC component.
            ),
            "overlaps": self.visualize_overlaps(
                context["image"],
                visualization_dump["sampling"],
                visualization_dump.get("is_monocular", None),
            ),
            "depth": self.visualize_depth(
                context,
                visualization_dump["depth"],
            ),
        }

    def visualize_attention(
        self,
        context_images: Float[Tensor, "batch view 3 height width"],
        sampling: None,
        attention: Float[Tensor, "layer bvr head 1 sample"],
    ) -> Float[Tensor, "3 vis_height vis_width"]:
        device = context_images.device

        # Pick a random batch element, view, and other view.
        b, v, ov, r, s, _ = sampling.xy_sample.shape
        rb = randrange(b)
        rv = randrange(v)
        rov = randrange(ov)
        num_samples = self.cfg.num_samples
        rr = np.random.choice(r, num_samples, replace=False)
        rr = torch.tensor(rr, dtype=torch.int64, device=device)

        # Visualize the rays in the ray view.
        ray_view = draw_points(
            context_images[rb, rv],
            sampling.xy_ray[rb, rv, rr],
            0,
            radius=4,
            x_range=(0, 1),
            y_range=(0, 1),
        )
        ray_view = draw_points(
            ray_view,
            sampling.xy_ray[rb, rv, rr],
            [get_distinct_color(i) for i, _ in enumerate(rr)],
            radius=3,
            x_range=(0, 1),
            y_range=(0, 1),
        )

        # Visualize attention in the sample view.
        attention = rearrange(
            attention, "l (b v r) hd () s -> l b v r hd s", b=b, v=v, r=r
        )
        attention = attention[:, rb, rv, rr, :, :]
        num_layers, _, hd, _ = attention.shape

        vis = []
        for il in range(num_layers):
            vis_layer = []
            for ihd in range(hd):
                # Create colors according to attention.
                color = [get_distinct_color(i) for i, _ in enumerate(rr)]
                color = torch.tensor(color, device=attention.device)
                color = rearrange(color, "r c -> r () c")
                attn = rearrange(attention[il, :, ihd], "r s -> r s ()")
                color = rearrange(attn * color, "r s c -> (r s ) c")

                # Draw the alternating bucket lines.
                vis_layer_head = draw_lines(
                    context_images[rb, self.encoder.sampler.index_v[rv, rov]],
                    rearrange(
                        sampling.xy_sample_near[rb, rv, rov, rr], "r s xy -> (r s) xy"
                    ),
                    rearrange(
                        sampling.xy_sample_far[rb, rv, rov, rr], "r s xy -> (r s) xy"
                    ),
                    color,
                    3,
                    cap="butt",
                    x_range=(0, 1),
                    y_range=(0, 1),
                )
                vis_layer.append(vis_layer_head)
            vis.append(add_label(vcat(*vis_layer), f"Layer {il}"))
        vis = add_label(add_border(add_border(hcat(*vis)), 1, 0), "Keys & Values")
        vis = add_border(hcat(add_label(ray_view, "ray_view"), vis, align="top"))
        return vis

    def visualize_depth(
        self,
        context: BatchedViews,
        multi_depth: Float[Tensor, "batch view height width surface spp"],
    ) -> Float[Tensor, "3 vis_width vis_height"]:
        multi_vis = []
        *_, srf, _ = multi_depth.shape
        for i in range(srf):
            depth = multi_depth[..., i, :]
            depth = depth.mean(dim=-1)

            # Compute relative depth and disparity.
            near = rearrange(context["near"], "b v -> b v () ()")
            far = rearrange(context["far"], "b v -> b v () ()")
            relative_depth = (depth - near) / (far - near)
            relative_disparity = 1 - (1 / depth - 1 / far) / (1 / near - 1 / far)

            relative_depth = apply_color_map_to_image(relative_depth, "turbo")
            relative_depth = vcat(*[hcat(*x) for x in relative_depth])
            relative_depth = add_label(relative_depth, "Depth")
            relative_disparity = apply_color_map_to_image(relative_disparity, "turbo")
            relative_disparity = vcat(*[hcat(*x) for x in relative_disparity])
            relative_disparity = add_label(relative_disparity, "Disparity")
            multi_vis.append(add_border(hcat(relative_depth, relative_disparity)))

        return add_border(vcat(*multi_vis))

    def visualize_overlaps(
        self,
        context_images: Float[Tensor, "batch view 3 height width"],
        sampling: None,
        is_monocular: Optional[Bool[Tensor, "batch view height width"]] = None,
    ) -> Float[Tensor, "3 vis_width vis_height"]:
        device = context_images.device
        b, v, _, h, w = context_images.shape
        green = torch.tensor([0.235, 0.706, 0.294], device=device)[..., None, None]
        rb = randrange(b)
        valid = sampling.valid[rb].float()
        ds = self.encoder.cfg.epipolar_transformer.downscale
        valid = repeat(
            valid,
            "v ov (h w) -> v ov c (h rh) (w rw)",
            c=3,
            h=h // ds,
            w=w // ds,
            rh=ds,
            rw=ds,
        )

        if is_monocular is not None:
            is_monocular = is_monocular[rb].float()
            is_monocular = repeat(is_monocular, "v h w -> v c h w", c=3, h=h, w=w)

        # Select context images in grid.
        context_images = context_images[rb]
        index, _ = generate_heterogeneous_index(v)
        valid = valid * (green + context_images[index]) / 2

        vis = vcat(*(hcat(im, hcat(*v)) for im, v in zip(context_images, valid)))
        vis = add_label(vis, "Context Overlaps")

        if is_monocular is not None:
            vis = hcat(vis, add_label(vcat(*is_monocular), "Monocular?"))

        return add_border(vis)

    def visualize_gaussians(
        self,
        context_images: Float[Tensor, "batch view 3 height width"],
        opacities: Float[Tensor, "batch vrspp"],
        covariances: Float[Tensor, "batch vrspp 3 3"],
        colors: Float[Tensor, "batch vrspp 3"],
    ) -> Float[Tensor, "3 vis_height vis_width"]:
        b, v, _, h, w = context_images.shape
        rb = randrange(b)
        context_images = context_images[rb]
        opacities = repeat(
            opacities[rb], "(v h w spp) -> spp v c h w", v=v, c=3, h=h, w=w
        )
        colors = rearrange(colors[rb], "(v h w spp) c -> spp v c h w", v=v, h=h, w=w)

        # Color-map Gaussian covariawnces.
        det = covariances[rb].det()
        det = apply_color_map(det / det.max(), "inferno")
        det = rearrange(det, "(v h w spp) c -> spp v c h w", v=v, h=h, w=w)

        return add_border(
            hcat(
                add_label(box(hcat(*context_images)), "Context"),
                add_label(box(vcat(*[hcat(*x) for x in opacities])), "Opacities"),
                add_label(
                    box(vcat(*[hcat(*x) for x in (colors * opacities)])), "Colors"
                ),
                add_label(box(vcat(*[hcat(*x) for x in colors])), "Colors (Raw)"),
                add_label(box(vcat(*[hcat(*x) for x in det])), "Determinant"),
            )
        )

    def visualize_probabilities(
        self,
        context_images: Float[Tensor, "batch view 3 height width"],
        sampling: None,
        pdf: Float[Tensor, "batch view ray sample"],
    ) -> Float[Tensor, "3 vis_height vis_width"]:
        device = context_images.device

        # Pick a random batch element, view, and other view.
        b, v, ov, r, _, _ = sampling.xy_sample.shape
        rb = randrange(b)
        rv = randrange(v)
        rov = randrange(ov)
        num_samples = self.cfg.num_samples
        rr = np.random.choice(r, num_samples, replace=False)
        rr = torch.tensor(rr, dtype=torch.int64, device=device)
        colors = [get_distinct_color(i) for i, _ in enumerate(rr)]
        colors = torch.tensor(colors, dtype=torch.float32, device=device)

        # Visualize the rays in the ray view.
        ray_view = draw_points(
            context_images[rb, rv],
            sampling.xy_ray[rb, rv, rr],
            0,
            radius=4,
            x_range=(0, 1),
            y_range=(0, 1),
        )
        ray_view = draw_points(
            ray_view,
            sampling.xy_ray[rb, rv, rr],
            colors,
            radius=3,
            x_range=(0, 1),
            y_range=(0, 1),
        )

        # Visualize probabilities in the sample view.
        pdf = pdf[rb, rv, rr]
        pdf = rearrange(pdf, "r s -> r s ()")
        colors = rearrange(colors, "r c -> r () c")
        sample_view = draw_lines(
            context_images[rb, self.encoder.sampler.index_v[rv, rov]],
            rearrange(sampling.xy_sample_near[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            rearrange(sampling.xy_sample_far[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            rearrange(pdf * colors, "r s c -> (r s) c"),
            6,
            cap="butt",
            x_range=(0, 1),
            y_range=(0, 1),
        )

        # Visualize rescaled probabilities in the sample view.
        pdf_magnified = pdf / reduce(pdf, "r s () -> r () ()", "max")
        sample_view_magnified = draw_lines(
            context_images[rb, self.encoder.sampler.index_v[rv, rov]],
            rearrange(sampling.xy_sample_near[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            rearrange(sampling.xy_sample_far[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            rearrange(pdf_magnified * colors, "r s c -> (r s) c"),
            6,
            cap="butt",
            x_range=(0, 1),
            y_range=(0, 1),
        )

        return add_border(
            hcat(
                add_label(ray_view, "Rays"),
                add_label(sample_view, "Samples"),
                add_label(sample_view_magnified, "Samples (Magnified PDF)"),
            )
        )

    def visualize_epipolar_samples(
        self,
        context_images: Float[Tensor, "batch view 3 height width"],
        sampling: None,
    ) -> Float[Tensor, "3 vis_height vis_width"]:
        device = context_images.device

        # Pick a random batch element, view, and other view.
        b, v, ov, r, s, _ = sampling.xy_sample.shape
        rb = randrange(b)
        rv = randrange(v)
        rov = randrange(ov)
        num_samples = self.cfg.num_samples
        rr = np.random.choice(r, num_samples, replace=False)
        rr = torch.tensor(rr, dtype=torch.int64, device=device)

        # Visualize the rays in the ray view.
        ray_view = draw_points(
            context_images[rb, rv],
            sampling.xy_ray[rb, rv, rr],
            0,
            radius=4,
            x_range=(0, 1),
            y_range=(0, 1),
        )
        ray_view = draw_points(
            ray_view,
            sampling.xy_ray[rb, rv, rr],
            [get_distinct_color(i) for i, _ in enumerate(rr)],
            radius=3,
            x_range=(0, 1),
            y_range=(0, 1),
        )

        # Visualize the samples and epipolar lines in the sample view.
        # First, draw the epipolar line in black.
        sample_view = draw_lines(
            context_images[rb, self.encoder.sampler.index_v[rv, rov]],
            sampling.xy_sample_near[rb, rv, rov, rr, 0],
            sampling.xy_sample_far[rb, rv, rov, rr, -1],
            0,
            5,
            cap="butt",
            x_range=(0, 1),
            y_range=(0, 1),
        )

        # Create an alternating line color for the buckets.
        color = repeat(
            torch.tensor([0, 1], device=device),
            "ab -> r (s ab) c",
            r=len(rr),
            s=(s + 1) // 2,
            c=3,
        )
        color = rearrange(color[:, :s], "r s c -> (r s) c")

        # Draw the alternating bucket lines.
        sample_view = draw_lines(
            sample_view,
            rearrange(sampling.xy_sample_near[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            rearrange(sampling.xy_sample_far[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            color,
            3,
            cap="butt",
            x_range=(0, 1),
            y_range=(0, 1),
        )

        # Draw the sample points.
        sample_view = draw_points(
            sample_view,
            rearrange(sampling.xy_sample[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            0,
            radius=4,
            x_range=(0, 1),
            y_range=(0, 1),
        )
        sample_view = draw_points(
            sample_view,
            rearrange(sampling.xy_sample[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            [get_distinct_color(i // s) for i in range(s * len(rr))],
            radius=3,
            x_range=(0, 1),
            y_range=(0, 1),
        )

        return add_border(
            hcat(add_label(ray_view, "Ray View"), add_label(sample_view, "Sample View"))
        )

    def visualize_epipolar_color_samples(
        self,
        context_images: Float[Tensor, "batch view 3 height width"],
        context: BatchedViews,
    ) -> Float[Tensor, "3 vis_height vis_width"]:
        device = context_images.device

        sampling = self.encoder.sampler(
            context["image"],
            context["extrinsics"],
            context["intrinsics"],
            context["near"],
            context["far"],
        )

        # Pick a random batch element, view, and other view.
        b, v, ov, r, s, _ = sampling.xy_sample.shape
        rb = randrange(b)
        rv = randrange(v)
        rov = randrange(ov)
        num_samples = self.cfg.num_samples
        rr = np.random.choice(r, num_samples, replace=False)
        rr = torch.tensor(rr, dtype=torch.int64, device=device)

        # Visualize the rays in the ray view.
        ray_view = draw_points(
            context_images[rb, rv],
            sampling.xy_ray[rb, rv, rr],
            0,
            radius=4,
            x_range=(0, 1),
            y_range=(0, 1),
        )
        ray_view = draw_points(
            ray_view,
            sampling.xy_ray[rb, rv, rr],
            [get_distinct_color(i) for i, _ in enumerate(rr)],
            radius=3,
            x_range=(0, 1),
            y_range=(0, 1),
        )

        # Visualize the samples and in the sample view.
        sample_view = draw_points(
            context_images[rb, self.encoder.sampler.index_v[rv, rov]],
            rearrange(sampling.xy_sample[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            [get_distinct_color(i // s) for i in range(s * len(rr))],
            radius=4,
            x_range=(0, 1),
            y_range=(0, 1),
        )
        sample_view = draw_points(
            sample_view,
            rearrange(sampling.xy_sample[rb, rv, rov, rr], "r s xy -> (r s) xy"),
            rearrange(sampling.features[rb, rv, rov, rr], "r s c -> (r s) c"),
            radius=3,
            x_range=(0, 1),
            y_range=(0, 1),
        )

        return add_border(
            hcat(add_label(ray_view, "Ray View"), add_label(sample_view, "Sample View"))
        )


```

# prompts/papers/depthsplat/src/model/encoder/visualization/encoder_visualizer_depthsplat_cfg.py

``` py
from dataclasses import dataclass

# This is in a separate file to avoid circular imports.


@dataclass
class EncoderVisualizerDepthSplatCfg:
    num_samples: int
    min_resolution: int
    export_ply: bool


```

# prompts/papers/depthsplat/src/model/model_wrapper.py

``` py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import wandb
from einops import pack, rearrange, repeat, einsum
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
import numpy as np
import json
import os
import time
from tqdm import tqdm
import torch.nn.functional as F
import math

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..dataset import DatasetCfg
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode, DecoderOutput
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from src.visualization.vis_depth import viz_depth_tensor
from PIL import Image
from ..misc.stablize_camera import render_stabilization_path
from .ply_export import save_gaussian_ply


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    lr_monodepth: float
    weight_decay: float


@dataclass
class TestCfg:
    output_path: Path
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int
    save_gt_image: bool
    save_input_images: bool
    save_depth: bool
    save_depth_concat_img: bool
    save_depth_npy: bool
    save_gaussian: bool
    render_chunk_size: int | None
    stablize_camera: bool
    stab_camera_kernel: int


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    eval_model_every_n_val: int
    eval_data_length: int
    eval_deterministic: bool
    eval_time_skip_steps: int
    eval_save_model: bool
    l1_loss: bool
    intermediate_loss_weight: float
    no_viz_video: bool
    viz_depth: bool
    forward_depth_only: bool
    train_ignore_large_loss: float
    no_log_projections: bool


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None
    eval_data_cfg: Optional[DatasetCfg | None]

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        eval_data_cfg: Optional[DatasetCfg | None] = None,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker
        self.eval_data_cfg = eval_data_cfg

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {"encoder": 0, "decoder": 0}

    def training_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape

        # Run the model.
        gaussians = self.encoder(
            batch["context"], self.global_step, False, scene_names=batch["scene"]
        )

        if isinstance(gaussians, dict):
            pred_depths = gaussians["depths"]
            gaussians = gaussians["gaussians"]

        supervise_intermediate_depth = False

        if gaussians.means.size(0) != batch["target"]["extrinsics"].size(0):
            supervise_intermediate_depth = True
            assert gaussians.means.size(0) % batch["target"]["extrinsics"].size(0) == 0
            num_depths = gaussians.means.size(0) // batch["target"]["extrinsics"].size(
                0
            )
            # add loss to intermediate depth predictions
            target_extrinsics = torch.cat(
                [batch["target"]["extrinsics"]] * num_depths, dim=0
            )
            target_intrinsics = torch.cat(
                [batch["target"]["intrinsics"]] * num_depths, dim=0
            )
            target_near = torch.cat([batch["target"]["near"]] * num_depths, dim=0)
            target_far = torch.cat([batch["target"]["far"]] * num_depths, dim=0)

            output_all = self.decoder.forward(
                gaussians,
                target_extrinsics,
                target_intrinsics,
                target_near,
                target_far,
                (h, w),
                depth_mode=self.train_cfg.depth_mode,
            )
            # split
            batch_size = batch["target"]["extrinsics"].size(0)
            # order: intermediate depth, final depth
            output_intermediate = DecoderOutput(
                color=output_all.color[:-batch_size],
                depth=(
                    output_all.depth[:-batch_size]
                    if output_all.depth is not None
                    else None
                ),
            )
            output = DecoderOutput(
                color=output_all.color[-batch_size:],
                depth=(
                    output_all.depth[-batch_size:]
                    if output_all.depth is not None
                    else None
                ),
            )

        else:
            output = self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=self.train_cfg.depth_mode,
            )

        target_gt = batch["target"]["image"]

        # Compute metrics.
        psnr_probabilistic = compute_psnr(
            rearrange(target_gt, "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        )
        self.log("train/psnr", psnr_probabilistic.mean())

        # Compute and log loss.
        total_loss = 0

        valid_depth_mask = None

        for loss_fn in self.losses:
            if loss_fn.name == "mse":
                loss = loss_fn.forward(
                    output,
                    batch,
                    gaussians,
                    self.global_step,
                    l1_loss=self.train_cfg.l1_loss,
                    clamp_large_error=self.train_cfg.train_ignore_large_loss,
                    valid_depth_mask=valid_depth_mask,
                )
            else:
                loss = loss_fn.forward(
                    output,
                    batch,
                    gaussians,
                    self.global_step,
                    valid_depth_mask=valid_depth_mask,
                )
            self.log(f"loss/{loss_fn.name}", loss)
            total_loss = total_loss + loss

        # color loss on intermediate output
        if supervise_intermediate_depth:
            for loss_fn in self.losses:
                if output_intermediate.color.size(0) != batch_size:
                    assert output_intermediate.color.size(0) % batch_size == 0
                    num_intermediate = output_intermediate.color.size(0) // batch_size
                    intermediate_loss = 0
                    for i in range(num_intermediate):
                        curr_output = DecoderOutput(
                            color=output_intermediate.color[
                                (batch_size * i) : (batch_size * (i + 1))
                            ],
                            depth=(
                                output_intermediate.depth[
                                    (batch_size * i) : (batch_size * (i + 1))
                                ]
                                if output_intermediate.depth is not None
                                else None
                            ),
                        )
                        curr_loss_weight = self.train_cfg.intermediate_loss_weight ** (
                            num_intermediate - i
                        )

                        if loss_fn.name == "mse":
                            loss = loss_fn.forward(
                                curr_output,
                                batch,
                                gaussians,
                                self.global_step,
                                l1_loss=self.train_cfg.l1_loss,
                                clamp_large_error=self.train_cfg.train_ignore_large_loss,
                                valid_depth_mask=valid_depth_mask,
                            )
                        else:
                            loss = loss_fn.forward(
                                curr_output,
                                batch,
                                gaussians,
                                self.global_step,
                                valid_depth_mask=valid_depth_mask,
                            )

                        intermediate_loss = intermediate_loss + curr_loss_weight * loss

                    self.log(f"loss/{loss_fn.name}_intermediate", intermediate_loss)
                    total_loss = total_loss + intermediate_loss
                else:
                    if loss_fn.name == "mse":
                        loss = loss_fn.forward(
                            output_intermediate,
                            batch,
                            gaussians,
                            self.global_step,
                            l1_loss=self.train_cfg.l1_loss,
                            clamp_large_error=self.train_cfg.train_ignore_large_loss,
                            valid_depth_mask=valid_depth_mask,
                        )
                    else:
                        loss = loss_fn.forward(
                            output_intermediate,
                            batch,
                            gaussians,
                            self.global_step,
                            valid_depth_mask=valid_depth_mask,
                        )
                    self.log(f"loss/{loss_fn.name}_intermediate", loss)
                    total_loss = (
                        total_loss + self.train_cfg.intermediate_loss_weight * loss
                    )

        self.log("loss/total", total_loss)

        if (
            self.global_rank == 0
            and self.global_step % self.train_cfg.print_log_every_n_steps == 0
        ):
            print(
                f"train step {self.global_step}; "
                f"scene = {[x[:20] for x in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}; "
                f"bound = [{batch['context']['near'].detach().cpu().numpy().mean()} "
                f"{batch['context']['far'].detach().cpu().numpy().mean()}]; "
                f"loss = {total_loss:.6f}"
            )
        self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
        self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)

        if self.global_step == 5 and self.global_rank == 0:
            os.system("nvidia-smi")

        return total_loss

    def test_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1

        pred_depths = None

        # save input views for visualization
        if self.test_cfg.save_input_images:
            (scene,) = batch["scene"]
            self.test_cfg.output_path = os.path.join(get_cfg()["output_dir"], "metrics")
            path = Path(get_cfg()["output_dir"])

            input_images = batch["context"]["image"][0]  # [V, 3, H, W]
            index = batch["context"]["index"][0]
            for idx, color in zip(index, input_images):
                save_image(color, path / "images" / scene / f"color/input_{idx:0>6}.png")

        # save depth vis
        if self.test_cfg.save_depth or self.test_cfg.save_gaussian:
            visualization_dump = {}
        else:
            visualization_dump = None

        # Render Gaussians.
        with self.benchmarker.time("encoder"):
            gaussians = self.encoder(
                batch["context"],
                self.global_step,
                deterministic=False,
                visualization_dump=visualization_dump,
            )

            if isinstance(gaussians, dict):
                pred_depths = gaussians["depths"]
                if "depth" in batch["context"]:
                    depth_gt = batch["context"]["depth"]
                gaussians = gaussians["gaussians"]

        # save gaussians
        if self.test_cfg.save_gaussian:
            scene = batch["scene"][0]
            save_path = Path(get_cfg()['output_dir']) / 'gaussians' / (scene + '.ply')
            save_gaussian_ply(gaussians, visualization_dump, batch, save_path)

        if not self.train_cfg.forward_depth_only:
            with self.benchmarker.time("decoder", num_calls=v):

                camera_poses = batch["target"]["extrinsics"]

                if self.test_cfg.stablize_camera:
                    stable_poses = render_stabilization_path(
                        camera_poses[0].detach().cpu().numpy(),
                        k_size=self.test_cfg.stab_camera_kernel,
                    )

                    stable_poses = list(
                        map(
                            lambda x: np.concatenate(
                                (x, np.array([[0.0, 0.0, 0.0, 1.0]])), axis=0
                            ),
                            stable_poses,
                        )
                    )
                    stable_poses = torch.from_numpy(np.stack(stable_poses, axis=0)).to(
                        camera_poses
                    )
                    camera_poses = stable_poses.unsqueeze(0)

                if self.test_cfg.render_chunk_size is not None:
                    chunk_size = self.test_cfg.render_chunk_size
                    num_chunks = math.ceil(camera_poses.shape[1] / chunk_size)

                    output = None
                    for i in range(num_chunks):
                        start = chunk_size * i
                        end = chunk_size * (i + 1)

                        render_intrinsics = batch["target"]["intrinsics"]
                        render_near = batch["target"]["near"]
                        render_far = batch["target"]["far"]

                        curr_output = self.decoder.forward(
                            gaussians,
                            camera_poses[:, start:end],
                            render_intrinsics[:, start:end],
                            render_near[:, start:end],
                            render_far[:, start:end],
                            (h, w),
                            depth_mode=None,
                        )

                        if i == 0:
                            output = curr_output
                        else:
                            # ignore depth
                            output.color = torch.cat(
                                (output.color, curr_output.color), dim=1
                            )

                else:
                    output = self.decoder.forward(
                        gaussians,
                        camera_poses,
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                        depth_mode=None,
                    )

        (scene,) = batch["scene"]
        self.test_cfg.output_path = os.path.join(get_cfg()["output_dir"], "metrics")
        path = Path(get_cfg()["output_dir"])

        # save depth
        if self.test_cfg.save_depth:
            if self.train_cfg.forward_depth_only:
                depth = pred_depths[0].cpu().detach()  # [V, H, W]
            else:
                depth = (
                    visualization_dump["depth"][0, :, :, :, 0, 0].cpu().detach()
                )  # [V, H, W]

            index = batch["context"]["index"][0]

            if self.test_cfg.save_depth_concat_img:
                # concat (img0, img1, depth0, depth1)
                image = batch['context']['image'][0]  # [V, 3, H, W] in [0,1]
                image = rearrange(image, "b c h w -> h (b w) c")  # [H, VW, 3]
                image_concat = (image.detach().cpu().numpy() * 255).astype(np.uint8)  # [H, VW, 3]

                depth_concat = []

            for idx, depth_i in zip(index, depth):
                depth_viz = viz_depth_tensor(
                    1.0 / depth_i, return_numpy=True
                )  # [H, W, 3]

                if self.test_cfg.save_depth_concat_img:
                    depth_concat.append(depth_viz)

                save_path = path / "images" / scene / "depth" / f"{idx:0>6}.png"
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(depth_viz).save(save_path)

                # save depth as npy
                if self.test_cfg.save_depth_npy:
                    depth_npy = depth_i.detach().cpu().numpy()
                    save_path = path / "images" / scene / "depth" / f"{idx:0>6}.npy"
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    np.save(save_path, depth_npy)

            if self.test_cfg.save_depth_concat_img:
                depth_concat = np.concatenate(depth_concat, axis=1)  # [H, VW, 3]
                concat = np.concatenate((image_concat, depth_concat), axis=0)  # [2H, VW, 3]

                save_path = path / "images" / scene / "depth" /  f"img_depth_{scene}.png"
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                Image.fromarray(concat).save(save_path)

            if self.train_cfg.forward_depth_only:
                return

        images_prob = output.color[0]
        rgb_gt = batch["target"]["image"][0]

        # Save images.
        if self.test_cfg.save_image:
            if self.test_cfg.save_gt_image:
                for index, color, gt in zip(
                    batch["target"]["index"][0], images_prob, rgb_gt
                ):
                    save_image(color, path / "images" / scene / f"color/{index:0>6}.png")
                    save_image(gt, path / "images" / scene / f"color/{index:0>6}_gt.png")
            else:
                for index, color in zip(batch["target"]["index"][0], images_prob):
                    save_image(color, path / "images" / scene / f"color/{index:0>6}.png")

        # save video
        if self.test_cfg.save_video:
            frame_str = "_".join([str(x.item()) for x in batch["context"]["index"][0]])
            save_video(
                [a for a in images_prob],
                path / "videos" / f"{scene}_frame_{frame_str}.mp4",
            )

        # compute scores
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += v

            if not self.train_cfg.forward_depth_only:
                rgb = images_prob

                if f"psnr" not in self.test_step_outputs:
                    self.test_step_outputs[f"psnr"] = []
                if f"ssim" not in self.test_step_outputs:
                    self.test_step_outputs[f"ssim"] = []
                if f"lpips" not in self.test_step_outputs:
                    self.test_step_outputs[f"lpips"] = []

                self.test_step_outputs[f"psnr"].append(
                    compute_psnr(rgb_gt, rgb).mean().item()
                )
                self.test_step_outputs[f"ssim"].append(
                    compute_ssim(rgb_gt, rgb).mean().item()
                )
                self.test_step_outputs[f"lpips"].append(
                    compute_lpips(rgb_gt, rgb).mean().item()
                )

    def on_test_end(self) -> None:
        out_dir = Path(self.test_cfg.output_path)
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                avg_scores = sum(metric_scores) / len(metric_scores)
                saved_scores[metric_name] = avg_scores
                print(metric_name, avg_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(
                    f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call"
                )
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / f"scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)
            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(out_dir / "benchmark.json")
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.summarize()

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)

        if self.global_rank == 0:
            print(
                f"validation step {self.global_step}; "
                f"scene = {[a[:20] for a in batch['scene']]}; "
                f"context = {batch['context']['index'].tolist()}"
            )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        gaussians_softmax = self.encoder(
            batch["context"],
            self.global_step,
            deterministic=False,
        )

        pred_depths = None

        if isinstance(gaussians_softmax, dict):
            pred_depths = gaussians_softmax["depths"]
            if "depth" in batch["context"]:
                depth_gt = batch["context"]["depth"]  # [B, V, H, W]
            gaussians_softmax = gaussians_softmax["gaussians"]

        if not self.train_cfg.forward_depth_only:
            output_softmax = self.decoder.forward(
                gaussians_softmax,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
            )
            rgb_softmax = output_softmax.color[0]

            # Compute validation metrics.
            rgb_gt = batch["target"]["image"][0]
            for tag, rgb in zip(("val",), (rgb_softmax,)):
                psnr = compute_psnr(rgb_gt, rgb).mean()
                self.log(f"val/psnr_{tag}", psnr)
                lpips = compute_lpips(rgb_gt, rgb).mean()
                self.log(f"val/lpips_{tag}", lpips)
                ssim = compute_ssim(rgb_gt, rgb).mean()
                self.log(f"val/ssim_{tag}", ssim)

        # viz depth
        if pred_depths is not None:
            # only visualize predicted depth
            pred_depths = pred_depths[0]  # [V, H, W]

            # gaussian downsample
            if pred_depths.shape[1:] != batch["context"]["image"].shape[-2:]:
                pred_depths = F.interpolate(
                    pred_depths.unsqueeze(1),
                    size=batch["context"]["image"].shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(1)

            inverse_depth_pred = 1.0 / pred_depths

            concat = []
            for i in range(inverse_depth_pred.size(0)):
                concat.append(inverse_depth_pred[i])

            concat = torch.cat(concat, dim=1)  # [H, W*N]

            depth_viz = viz_depth_tensor(concat.cpu().detach())  # [3, H, W*N]

            # also concat images
            input_images = batch["context"]["image"][0]  # [N, 3, H, W]
            concat_img = [img for img in input_images]
            concat_img = torch.cat(concat_img, dim=-1) * 255  # [3, H, W*N]

            concat = torch.cat(
                (concat_img.cpu().detach(), depth_viz), dim=1
            )  # [3, H*2, W*N]

            self.logger.log_image(
                "depth",
                [concat],
                step=self.global_step,
                caption=batch["scene"],
            )

        if not self.train_cfg.forward_depth_only:
            # Construct comparison image.
            comparison = hcat(
                add_label(vcat(*batch["context"]["image"][0]), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_softmax), "Target (Prediction)"),
            )
            self.logger.log_image(
                "comparison",
                [prep_image(add_border(comparison))],
                step=self.global_step,
                caption=batch["scene"],
            )

            if not self.train_cfg.no_log_projections:
                # Render projections and construct projection image.
                projections = hcat(
                    *render_projections(
                        gaussians_softmax,
                        256,
                        extra_label="(Prediction)",
                    )[0]
                )
                self.logger.log_image(
                    "projection",
                    [prep_image(add_border(projections))],
                    step=self.global_step,
                )

                # Draw cameras.
                cameras = hcat(*render_cameras(batch, 256))
                self.logger.log_image(
                    "cameras", [prep_image(add_border(cameras))], step=self.global_step
                )

            if self.encoder_visualizer is not None:
                for k, image in self.encoder_visualizer.visualize(
                    batch["context"], self.global_step
                ).items():
                    self.logger.log_image(k, [prep_image(image)], step=self.global_step)

            # Run video validation step.
            if not self.train_cfg.no_viz_video:
                self.render_video_interpolation(batch)
                self.render_video_wobble(batch)
                if self.train_cfg.extended_visualization:
                    self.render_video_interpolation_exaggerated(batch)

    def on_validation_epoch_end(self) -> None:
        """hack to run the full validation"""
        if self.trainer.sanity_checking and self.global_rank == 0:
            print(self.encoder)  # log the model to wandb log files

        if (not self.trainer.sanity_checking) and (self.eval_data_cfg is not None):
            self.eval_cnt = self.eval_cnt + 1
            if self.eval_cnt % self.train_cfg.eval_model_every_n_val == 0:
                # backup current ckpt before running full test sets eval
                if self.train_cfg.eval_save_model:
                    ckpt_saved_path = (
                        self.trainer.checkpoint_callback.format_checkpoint_name(
                            dict(
                                epoch=self.trainer.current_epoch,
                                step=self.trainer.global_step,
                            )
                        )
                    )
                    backup_dir = str(
                        Path(ckpt_saved_path).parent.parent / "checkpoints_backups"
                    )
                    if self.global_rank == 0:
                        os.makedirs(backup_dir, exist_ok=True)
                    ckpt_saved_path = os.path.join(
                        backup_dir, os.path.basename(ckpt_saved_path)
                    )
                    # call save_checkpoint on ALL process as suggested by pytorch_lightning
                    self.trainer.save_checkpoint(
                        ckpt_saved_path,
                        weights_only=True,
                    )
                    if self.global_rank == 0:
                        print(f"backup model to {ckpt_saved_path}.")

                # run full test sets eval on rank=0 device
                self.run_full_test_sets_eval()

    @rank_zero_only
    def run_full_test_sets_eval(self) -> None:
        start_t = time.time()

        pred_depths = None
        depth_gt = None

        full_testsets = self.trainer.datamodule.test_dataloader(
            dataset_cfg=self.eval_data_cfg
        )
        scores_dict = {}

        if not self.train_cfg.forward_depth_only:
            for score_tag in ("psnr", "ssim", "lpips"):
                scores_dict[score_tag] = {}
                for method_tag in ("deterministic", "probabilistic"):
                    scores_dict[score_tag][method_tag] = []

        # evaluate depth
        if self.train_cfg.viz_depth:
            for score_tag in ("abs_rel", "rmse", "a1"):
                scores_dict[score_tag] = {}
                for method_tag in ("deterministic", "probabilistic"):
                    scores_dict[score_tag][method_tag] = []

        self.benchmarker.clear_history()
        time_skip_first_n_steps = min(
            self.train_cfg.eval_time_skip_steps, len(full_testsets)
        )
        time_skip_steps_dict = {"encoder": 0, "decoder": 0}
        for batch_idx, batch in tqdm(
            enumerate(full_testsets),
            total=min(len(full_testsets), self.train_cfg.eval_data_length),
        ):
            if batch_idx >= self.train_cfg.eval_data_length:
                break

            batch = self.data_shim(batch)
            batch = self.transfer_batch_to_device(batch, "cuda", dataloader_idx=0)

            # Render Gaussians.
            b, v, _, h, w = batch["target"]["image"].shape
            assert b == 1
            if batch_idx < time_skip_first_n_steps:
                time_skip_steps_dict["encoder"] += 1
                time_skip_steps_dict["decoder"] += v

            with self.benchmarker.time("encoder"):
                gaussians_probabilistic = self.encoder(
                    batch["context"],
                    self.global_step,
                    deterministic=False,
                )

                if isinstance(gaussians_probabilistic, dict):
                    pred_depths = gaussians_probabilistic["depths"]
                    if "depth" in batch["context"]:
                        depth_gt = batch["context"]["depth"]
                    gaussians_probabilistic = gaussians_probabilistic["gaussians"]

            if not self.train_cfg.forward_depth_only:
                with self.benchmarker.time("decoder", num_calls=v):
                    output_probabilistic = self.decoder.forward(
                        gaussians_probabilistic,
                        batch["target"]["extrinsics"],
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                    )
                rgbs = [output_probabilistic.color[0]]
                tags = ["probabilistic"]

                if self.train_cfg.eval_deterministic:
                    gaussians_deterministic = self.encoder(
                        batch["context"],
                        self.global_step,
                        deterministic=True,
                    )
                    output_deterministic = self.decoder.forward(
                        gaussians_deterministic,
                        batch["target"]["extrinsics"],
                        batch["target"]["intrinsics"],
                        batch["target"]["near"],
                        batch["target"]["far"],
                        (h, w),
                    )
                    rgbs.append(output_deterministic.color[0])
                    tags.append("deterministic")

                # Compute validation metrics.
                rgb_gt = batch["target"]["image"][0]
                for tag, rgb in zip(tags, rgbs):
                    scores_dict["psnr"][tag].append(
                        compute_psnr(rgb_gt, rgb).mean().item()
                    )
                    scores_dict["lpips"][tag].append(
                        compute_lpips(rgb_gt, rgb).mean().item()
                    )
                    scores_dict["ssim"][tag].append(
                        compute_ssim(rgb_gt, rgb).mean().item()
                    )

        # summarise scores and log to logger
        for score_tag, methods in scores_dict.items():
            for method_tag, cur_scores in methods.items():
                if len(cur_scores) > 0:
                    cur_mean = sum(cur_scores) / len(cur_scores)
                    self.log(f"test/{score_tag}", cur_mean)
        # summarise run time
        for tag, times in self.benchmarker.execution_times.items():
            times = times[int(time_skip_steps_dict[tag]) :]
            print(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")
            self.log(f"test/runtime_avg_{tag}", np.mean(times))
        self.benchmarker.clear_history()

        overall_eval_time = time.time() - start_t
        print(f"Eval total time cost: {overall_eval_time:.3f}s")
        self.log("test/runtime_all", overall_eval_time)

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (
                    batch["context"]["extrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["extrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (
                    batch["context"]["intrinsics"][0, 1]
                    if v == 2
                    else batch["target"]["intrinsics"][0, 0]
                ),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob = self.encoder(batch["context"], self.global_step, False)
        # gaussians_det = self.encoder(batch["context"], self.global_step, True)

        if isinstance(gaussians_prob, dict):
            gaussians_prob = gaussians_prob["gaussians"]

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth"
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]

        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Prediction"),
                )
            )
            for image_prob, _ in zip(images_prob, images_prob)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def configure_optimizers(self):
        if self.optimizer_cfg.lr_monodepth > 0:
            pretrained_params = []
            new_params = []

            for name, param in self.named_parameters():
                if "pretrained" in name:
                    pretrained_params.append(param)
                else:
                    new_params.append(param)

            optimizer = torch.optim.AdamW(
                [
                    {
                        "params": pretrained_params,
                        "lr": self.optimizer_cfg.lr_monodepth,
                    },
                    {"params": new_params, "lr": self.optimizer_cfg.lr},
                ],
                weight_decay=self.optimizer_cfg.weight_decay,
            )

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                [self.optimizer_cfg.lr_monodepth, self.optimizer_cfg.lr],
                self.trainer.max_steps + 10,
                pct_start=0.01,
                cycle_momentum=False,
                anneal_strategy="cos",
            )

        else:
            optimizer = optim.AdamW(
                self.parameters(),
                lr=self.optimizer_cfg.lr,
                weight_decay=self.optimizer_cfg.weight_decay,
            )

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                self.optimizer_cfg.lr,
                self.trainer.max_steps + 10,
                pct_start=0.01,
                cycle_momentum=False,
                anneal_strategy="cos",
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


```

# prompts/papers/depthsplat/src/model/ply_export.py

``` py
from pathlib import Path

import numpy as np
import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes


def export_ply(
    extrinsics: Float[Tensor, "4 4"],
    means: Float[Tensor, "gaussian 3"],
    scales: Float[Tensor, "gaussian 3"],
    rotations: Float[Tensor, "gaussian 4"],
    harmonics: Float[Tensor, "gaussian 3 d_sh"],
    opacities: Float[Tensor, " gaussian"],
    path: Path,
):

    view_rotation = extrinsics[:3, :3].inverse()
    # Apply the rotation to the means (Gaussian positions).
    means = einsum(view_rotation, means, "i j, ... j -> ... i")

    # Apply the rotation to the Gaussian rotations.
    rotations = R.from_quat(rotations.detach().cpu().numpy()).as_matrix()
    rotations = view_rotation.detach().cpu().numpy() @ rotations
    rotations = R.from_matrix(rotations).as_quat()
    x, y, z, w = rearrange(rotations, "g xyzw -> xyzw g")
    rotations = np.stack((w, x, y, z), axis=-1)

    # Since our axes are swizzled for the spherical harmonics, we only export the DC band
    harmonics_view_invariant = harmonics[..., 0]

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(0)]
    elements = np.empty(means.shape[0], dtype=dtype_full)
    attributes = (
        means.detach().cpu().numpy(),
        torch.zeros_like(means).detach().cpu().numpy(),
        harmonics_view_invariant.detach().cpu().contiguous().numpy(),
        torch.logit(opacities[..., None]).detach().cpu().numpy(),
        scales.log().detach().cpu().numpy(),
        rotations,
    )
    attributes = np.concatenate(attributes, axis=1)
    elements[:] = list(map(tuple, attributes))
    path.parent.mkdir(exist_ok=True, parents=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)
    

def save_gaussian_ply(gaussians, visualization_dump, example, save_path):

    v, _, h, w = example["context"]["image"].shape[1:]

    # Transform means into camera space.
    means = rearrange(
        gaussians.means, "() (v h w spp) xyz -> h w spp v xyz", v=v, h=h, w=w
    )

    # Create a mask to filter the Gaussians. throw away Gaussians at the
    # borders, since they're generally of lower quality.
    mask = torch.zeros_like(means[..., 0], dtype=torch.bool)
    GAUSSIAN_TRIM = 8
    mask[GAUSSIAN_TRIM:-GAUSSIAN_TRIM, GAUSSIAN_TRIM:-GAUSSIAN_TRIM, :, :] = 1

    def trim(element):
        element = rearrange(
            element, "() (v h w spp) ... -> h w spp v ...", v=v, h=h, w=w
        )
        return element[mask][None]

    # convert the rotations from camera space to world space as required
    cam_rotations = trim(visualization_dump["rotations"])[0]
    c2w_mat = repeat(
        example["context"]["extrinsics"][0, :, :3, :3],
        "v a b -> h w spp v a b",
        h=h,
        w=w,
        spp=1,
    )
    c2w_mat = c2w_mat[mask]  # apply trim

    cam_rotations_np = R.from_quat(
        cam_rotations.detach().cpu().numpy()
    ).as_matrix()
    world_mat = c2w_mat.detach().cpu().numpy() @ cam_rotations_np
    world_rotations = R.from_matrix(world_mat).as_quat()
    world_rotations = torch.from_numpy(world_rotations).to(
        visualization_dump["scales"]
    )

    export_ply(
        example["context"]["extrinsics"][0, 0],
        trim(gaussians.means)[0],
        trim(visualization_dump["scales"])[0],
        world_rotations,
        trim(gaussians.harmonics)[0],
        trim(gaussians.opacities)[0],
        save_path,
    )




```

# prompts/papers/depthsplat/src/model/types.py

``` py
from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian dim"]
    covariances: Float[Tensor, "batch gaussian dim dim"]
    harmonics: Float[Tensor, "batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "batch gaussian"]


```

# prompts/papers/depthsplat/src/scripts/convert_dl3dv_test.py

``` py
import subprocess
import sys
from pathlib import Path
from typing import Literal, TypedDict
from PIL import Image

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from torch import Tensor
from tqdm import tqdm
import argparse
import json
import os

from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, help="original dataset directory")
parser.add_argument("--output_dir", type=str, help="processed dataset directory")
parser.add_argument(
    "--img_subdir",
    type=str,
    default="images_8",
    help="image directory name",
    choices=[
        "images_4",
        "images_8",
    ],
)
parser.add_argument("--n_test", type=int, default=10, help="test skip")
parser.add_argument("--which_stage", type=str, default=None, help="dataset directory")
parser.add_argument("--detect_overlap", action="store_true")

args = parser.parse_args()


INPUT_DIR = Path(args.input_dir)
OUTPUT_DIR = Path(args.output_dir)


# Target 200 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(2e8)


def get_example_keys(stage: Literal["test", "train"]) -> list[str]:
    image_keys = set(
        example.name
        for example in tqdm(list((INPUT_DIR / stage).iterdir()), desc="Indexing scenes")
        if example.is_dir() and not example.name.startswith(".")
    )
    # keys = image_keys & metadata_keys
    keys = image_keys
    # print(keys)
    print(f"Found {len(keys)} keys.")
    return sorted(list(keys))


def get_size(path: Path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))


def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def load_images(example_path: Path) -> dict[int, UInt8[Tensor, "..."]]:
    """Load JPG images as raw bytes (do not decode)."""

    return {
        int(path.stem.split("_")[-1]): load_raw(path)
        for path in example_path.iterdir()
        if path.suffix.lower() not in [".npz"]
    }


class Metadata(TypedDict):
    url: str
    timestamps: Int[Tensor, " camera"]
    cameras: Float[Tensor, "camera entry"]


class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]


def load_metadata(example_path: Path) -> Metadata:
    blender2opencv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )
    url = str(example_path).split("/")[-3]
    with open(example_path, "r") as f:
        meta_data = json.load(f)

    store_h, store_w = meta_data["h"], meta_data["w"]
    fx, fy, cx, cy = (
        meta_data["fl_x"],
        meta_data["fl_y"],
        meta_data["cx"],
        meta_data["cy"],
    )
    saved_fx = float(fx) / float(store_w)
    saved_fy = float(fy) / float(store_h)
    saved_cx = float(cx) / float(store_w)
    saved_cy = float(cy) / float(store_h)

    timestamps = []
    cameras = []
    opencv_c2ws = []  # will be used to calculate camera distance

    for frame in meta_data["frames"]:
        timestamps.append(
            int(os.path.basename(frame["file_path"]).split(".")[0].split("_")[-1])
        )
        camera = [saved_fx, saved_fy, saved_cx, saved_cy, 0.0, 0.0]
        # transform_matrix is in blender c2w, while we need to store opencv w2c matrix here
        opencv_c2w = np.array(frame["transform_matrix"]) @ blender2opencv
        opencv_c2ws.append(opencv_c2w)
        camera.extend(np.linalg.inv(opencv_c2w)[:3].flatten().tolist())
        cameras.append(np.array(camera))

    # timestamp should be the one that match the above images keys, use for indexing
    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)

    return {"url": url, "timestamps": timestamps, "cameras": cameras}


def partition_train_test_splits(root_dir, n_test=10):
    sub_folders = sorted(glob(os.path.join(root_dir, "*/")))
    test_list = sub_folders[::n_test]
    train_list = [x for x in sub_folders if x not in test_list]
    out_dict = {"train": train_list, "test": test_list}
    return out_dict


def is_image_shape_matched(image_dir, target_shape):
    image_path = sorted(glob(str(image_dir / "*")))
    if len(image_path) == 0:
        return False

    image_path = image_path[0]
    try:
        im = Image.open(image_path)
    except:
        return False
    w, h = im.size
    if (h, w) == target_shape:
        return True
    else:
        return False


def legal_check_for_all_scenes(root_dir, target_shape):
    valid_folders = []
    sub_folders = sorted(glob(os.path.join(root_dir, "*/nerfstudio")))
    for sub_folder in tqdm(sub_folders, desc="checking scenes..."):
        img_dir = os.path.join(sub_folder, "images_8")  # 270x480
        # img_dir = os.path.join(sub_folder, 'images_4')  # 540x960
        if not is_image_shape_matched(Path(img_dir), target_shape):
            print(f"image shape does not match for {sub_folder}")
            continue
        pose_file = os.path.join(sub_folder, "transforms.json")
        if not os.path.isfile(pose_file):
            print(f"cannot find pose file for {sub_folder}")
            continue

        valid_folders.append(sub_folder)

    return valid_folders


if __name__ == "__main__":
    if "images_8" in args.img_subdir:
        target_shape = (270, 480)  # (h, w)
    elif "images_4" in args.img_subdir:
        target_shape = (540, 960)
    else:
        raise ValueError

    print("checking all scenes...")
    valid_scenes = legal_check_for_all_scenes(INPUT_DIR, target_shape)
    print("valid scenes:", len(valid_scenes))

    for stage in ["test"]:

        error_logs = []
        image_dirs = valid_scenes

        chunk_size = 0
        chunk_index = 0
        chunk: list[Example] = []

        def save_chunk():
            global chunk_size
            global chunk_index
            global chunk

            chunk_key = f"{chunk_index:0>6}"
            dir = OUTPUT_DIR / stage
            dir.mkdir(exist_ok=True, parents=True)
            torch.save(chunk, dir / f"{chunk_key}.torch")

            # Reset the chunk.
            chunk_size = 0
            chunk_index += 1
            chunk = []

        for image_dir in tqdm(image_dirs, desc=f"Processing {stage}"):
            key = os.path.basename(os.path.dirname(image_dir.strip("/")))

            image_dir = Path(image_dir) / "images_8"  # 270x480
            # image_dir = Path(image_dir) / 'images_4'  # 540x960

            num_bytes = get_size(image_dir)

            # Read images and metadata.
            try:
                images = load_images(image_dir)
            except:
                print("image loading error")
                continue
            meta_path = image_dir.parent / "transforms.json"
            if not meta_path.is_file():
                error_msg = f"---------> [ERROR] no meta file in {key}, skip."
                print(error_msg)
                error_logs.append(error_msg)
                continue
            example = load_metadata(meta_path)

            # Merge the images into the example.
            try:
                example["images"] = [
                    images[timestamp.item()] for timestamp in example["timestamps"]
                ]
            except:
                error_msg = f"---------> [ERROR] Some images missing in {key}, skip."
                print(error_msg)
                error_logs.append(error_msg)
                continue

            # Add the key to the example.
            example["key"] = key

            chunk.append(example)
            chunk_size += num_bytes

            if chunk_size >= TARGET_BYTES_PER_CHUNK:
                save_chunk()

        if chunk_size > 0:
            save_chunk()


```

# prompts/papers/depthsplat/src/scripts/convert_dl3dv_train.py

``` py
import subprocess
import sys
from pathlib import Path
from typing import Literal, TypedDict
from PIL import Image

import numpy as np
import torch
from jaxtyping import Float, Int, UInt8
from torch import Tensor
from tqdm import tqdm
import argparse
import json
import os

from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, help="original dataset directory")
parser.add_argument("--output_dir", type=str, help="processed dataset directory")
parser.add_argument(
    "--img_subdir",
    type=str,
    default="images_8",
    help="image directory name",
    choices=[
        "images_4",
        "images_8",
    ],
)
parser.add_argument("--n_test", type=int, default=10, help="test skip")
parser.add_argument("--which_stage", type=str, default=None, help="dataset directory")
parser.add_argument("--detect_overlap", action="store_true")

args = parser.parse_args()


INPUT_DIR = Path(args.input_dir)
OUTPUT_DIR = Path(args.output_dir)


# Target 200 MB per chunk.
TARGET_BYTES_PER_CHUNK = int(2e8)


def get_example_keys(stage: Literal["test", "train"]) -> list[str]:
    image_keys = set(
        example.name
        for example in tqdm(list((INPUT_DIR / stage).iterdir()), desc="Indexing scenes")
        if example.is_dir() and not example.name.startswith(".")
    )
    # keys = image_keys & metadata_keys
    keys = image_keys
    # print(keys)
    # assert False
    print(f"Found {len(keys)} keys.")
    return sorted(list(keys))


def get_size(path: Path) -> int:
    """Get file or folder size in bytes."""
    return int(subprocess.check_output(["du", "-b", path]).split()[0].decode("utf-8"))


def load_raw(path: Path) -> UInt8[Tensor, " length"]:
    return torch.tensor(np.memmap(path, dtype="uint8", mode="r"))


def load_images(example_path: Path) -> dict[int, UInt8[Tensor, "..."]]:
    """Load JPG images as raw bytes (do not decode)."""

    return {
        int(path.stem.split("_")[-1]): load_raw(path)
        for path in example_path.iterdir()
        if path.suffix.lower() not in [".npz"]
    }


class Metadata(TypedDict):
    url: str
    timestamps: Int[Tensor, " camera"]
    cameras: Float[Tensor, "camera entry"]


class Example(Metadata):
    key: str
    images: list[UInt8[Tensor, "..."]]


def load_metadata(example_path: Path) -> Metadata:
    blender2opencv = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )
    url = str(example_path).split("/")[-3]
    with open(example_path, "r") as f:
        meta_data = json.load(f)

    store_h, store_w = meta_data["h"], meta_data["w"]
    fx, fy, cx, cy = (
        meta_data["fl_x"],
        meta_data["fl_y"],
        meta_data["cx"],
        meta_data["cy"],
    )
    saved_fx = float(fx) / float(store_w)
    saved_fy = float(fy) / float(store_h)
    saved_cx = float(cx) / float(store_w)
    saved_cy = float(cy) / float(store_h)

    timestamps = []
    cameras = []
    opencv_c2ws = []  # will be used to calculate camera distance

    for frame in meta_data["frames"]:
        timestamps.append(
            int(os.path.basename(frame["file_path"]).split(".")[0].split("_")[-1])
        )
        camera = [saved_fx, saved_fy, saved_cx, saved_cy, 0.0, 0.0]
        # transform_matrix is in blender c2w, while we need to store opencv w2c matrix here
        opencv_c2w = np.array(frame["transform_matrix"]) @ blender2opencv
        opencv_c2ws.append(opencv_c2w)
        camera.extend(np.linalg.inv(opencv_c2w)[:3].flatten().tolist())
        cameras.append(np.array(camera))

    # timestamp should be the one that match the above images keys, use for indexing
    timestamps = torch.tensor(timestamps, dtype=torch.int64)
    cameras = torch.tensor(np.stack(cameras), dtype=torch.float32)

    return {"url": url, "timestamps": timestamps, "cameras": cameras}


def partition_train_test_splits(root_dir, n_test=10):
    sub_folders = sorted(glob(os.path.join(root_dir, "*/")))
    test_list = sub_folders[::n_test]
    train_list = [x for x in sub_folders if x not in test_list]
    out_dict = {"train": train_list, "test": test_list}
    return out_dict


def is_image_shape_matched(image_dir, target_shape):
    image_path = sorted(glob(str(image_dir / "*")))
    if len(image_path) == 0:
        return False

    image_path = image_path[0]
    try:
        im = Image.open(image_path)
    except:
        return False
    w, h = im.size
    if (h, w) == target_shape:
        return True
    else:
        return False


def legal_check_for_all_scenes(root_dir, target_shape):
    valid_folders = []
    sub_folders = sorted(glob(os.path.join(root_dir, "*/*")))
    for sub_folder in tqdm(sub_folders, desc="checking scenes..."):
        # img_dir = os.path.join(sub_folder, 'images_8')
        img_dir = os.path.join(sub_folder, "images_4")
        if not is_image_shape_matched(Path(img_dir), target_shape):
            print(f"image shape does not match for {sub_folder}")
            continue
        pose_file = os.path.join(sub_folder, "transforms.json")
        if not os.path.isfile(pose_file):
            print(f"cannot find pose file for {sub_folder}")
            continue

        valid_folders.append(sub_folder)

    return valid_folders


if __name__ == "__main__":
    if "images_8" in args.img_subdir:
        target_shape = (270, 480)  # (h, w)
    elif "images_4" in args.img_subdir:
        target_shape = (540, 960)
    else:
        raise ValueError

    print("checking all scenes...")
    valid_scenes = legal_check_for_all_scenes(INPUT_DIR, target_shape)
    print("valid scenes:", len(valid_scenes))

    # test scenes
    test_scenes = "your_test_set_index.json"
    with open(test_scenes, "r") as f:
        overlap_scenes = json.load(f)

    assert len(overlap_scenes) == 140, "test scenes should contain 140 scenes"

    for stage in ["train"]:

        error_logs = []
        image_dirs = valid_scenes

        chunk_size = 0
        chunk_index = 0
        chunk: list[Example] = []

        def save_chunk():
            global chunk_size
            global chunk_index
            global chunk

            chunk_key = f"{chunk_index:0>6}"
            dir = OUTPUT_DIR / stage
            dir.mkdir(exist_ok=True, parents=True)
            torch.save(chunk, dir / f"{chunk_key}.torch")

            # Reset the chunk.
            chunk_size = 0
            chunk_index += 1
            chunk = []

        for image_dir in tqdm(image_dirs, desc=f"Processing {stage}"):
            key = os.path.basename(image_dir.strip("/"))
            # skip test scenes
            if key in overlap_scenes:
                print(f"scene {key} in benchmark, skip.")
                continue

            image_dir = Path(image_dir) / "images_8"  # 270x480
            # image_dir = Path(image_dir) / 'images_4'  # 540x960

            num_bytes = get_size(image_dir)

            # Read images and metadata.
            try:
                images = load_images(image_dir)
            except:
                print("image loading error")
                continue
            meta_path = image_dir.parent / "transforms.json"
            if not meta_path.is_file():
                error_msg = f"---------> [ERROR] no meta file in {key}, skip."
                print(error_msg)
                error_logs.append(error_msg)
                continue
            example = load_metadata(meta_path)

            # Merge the images into the example.
            try:
                example["images"] = [
                    images[timestamp.item()] for timestamp in example["timestamps"]
                ]
            except:
                error_msg = f"---------> [ERROR] Some images missing in {key}, skip."
                print(error_msg)
                error_logs.append(error_msg)
                continue

            # Add the key to the example.
            example["key"] = "dl3dv_" + key

            chunk.append(example)
            chunk_size += num_bytes

            if chunk_size >= TARGET_BYTES_PER_CHUNK:
                save_chunk()

        if chunk_size > 0:
            save_chunk()


```

# prompts/papers/depthsplat/src/scripts/generate_dl3dv_index.py

``` py
import json
from pathlib import Path

import torch
from tqdm import tqdm

DATASET_PATH = Path("your_dataset_path")

if __name__ == "__main__":
    # "train" or "test"
    for stage in ["test"]:
        stage = DATASET_PATH / stage

        index = {}
        for chunk_path in tqdm(
            sorted(list(stage.iterdir())), desc=f"Indexing {stage.name}"
        ):
            if chunk_path.suffix == ".torch":
                chunk = torch.load(chunk_path)
                for example in chunk:
                    index[example["key"]] = str(chunk_path.relative_to(stage))
        with (stage / "index.json").open("w") as f:
            json.dump(index, f)


```

# prompts/papers/depthsplat/src/visualization/annotation.py

``` py
from pathlib import Path
from string import ascii_letters, digits, punctuation

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

from .layout import vcat

EXPECTED_CHARACTERS = digits + punctuation + ascii_letters


def draw_label(
    text: str,
    font: Path,
    font_size: int,
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "3 height width"]:
    """Draw a black label on a white background with no border."""
    try:
        font = ImageFont.truetype(str(font), font_size)
    except OSError:
        font = ImageFont.load_default()
    left, _, right, _ = font.getbbox(text)
    width = right - left
    _, top, _, bottom = font.getbbox(EXPECTED_CHARACTERS)
    height = bottom - top
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=font, fill="black")
    image = torch.tensor(np.array(image) / 255, dtype=torch.float32, device=device)
    return rearrange(image, "h w c -> c h w")


def add_label(
    image: Float[Tensor, "3 width height"],
    label: str,
    font: Path = Path("assets/Inter-Regular.otf"),
    font_size: int = 24,
) -> Float[Tensor, "3 width_with_label height_with_label"]:
    return vcat(
        draw_label(label, font, font_size, image.device),
        image,
        align="left",
        gap=4,
    )


```

# prompts/papers/depthsplat/src/visualization/camera_trajectory/interpolation.py

``` py
import torch
from einops import einsum, rearrange, reduce
from jaxtyping import Float
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def interpolate_intrinsics(
    initial: Float[Tensor, "*#batch 3 3"],
    final: Float[Tensor, "*#batch 3 3"],
    t: Float[Tensor, " time_step"],
) -> Float[Tensor, "*batch time_step 3 3"]:
    initial = rearrange(initial, "... i j -> ... () i j")
    final = rearrange(final, "... i j -> ... () i j")
    t = rearrange(t, "t -> t () ()")
    return initial + (final - initial) * t


def intersect_rays(
    a_origins: Float[Tensor, "*#batch dim"],
    a_directions: Float[Tensor, "*#batch dim"],
    b_origins: Float[Tensor, "*#batch dim"],
    b_directions: Float[Tensor, "*#batch dim"],
) -> Float[Tensor, "*batch dim"]:
    """Compute the least-squares intersection of rays. Uses the math from here:
    https://math.stackexchange.com/a/1762491/286022
    """

    # Broadcast and stack the tensors.
    a_origins, a_directions, b_origins, b_directions = torch.broadcast_tensors(
        a_origins, a_directions, b_origins, b_directions
    )
    origins = torch.stack((a_origins, b_origins), dim=-2)
    directions = torch.stack((a_directions, b_directions), dim=-2)

    # Compute n_i * n_i^T - eye(3) from the equation.
    n = einsum(directions, directions, "... n i, ... n j -> ... n i j")
    n = n - torch.eye(3, dtype=origins.dtype, device=origins.device)

    # Compute the left-hand side of the equation.
    lhs = reduce(n, "... n i j -> ... i j", "sum")

    # Compute the right-hand side of the equation.
    rhs = einsum(n, origins, "... n i j, ... n j -> ... n i")
    rhs = reduce(rhs, "... n i -> ... i", "sum")

    # Left-matrix-multiply both sides by the inverse of lhs to find p.
    return torch.linalg.lstsq(lhs, rhs).solution


def normalize(a: Float[Tensor, "*#batch dim"]) -> Float[Tensor, "*#batch dim"]:
    return a / a.norm(dim=-1, keepdim=True)


def generate_coordinate_frame(
    y: Float[Tensor, "*#batch 3"],
    z: Float[Tensor, "*#batch 3"],
) -> Float[Tensor, "*batch 3 3"]:
    """Generate a coordinate frame given perpendicular, unit-length Y and Z vectors."""
    y, z = torch.broadcast_tensors(y, z)
    return torch.stack([y.cross(z), y, z], dim=-1)


def generate_rotation_coordinate_frame(
    a: Float[Tensor, "*#batch 3"],
    b: Float[Tensor, "*#batch 3"],
    eps: float = 1e-4,
) -> Float[Tensor, "*batch 3 3"]:
    """Generate a coordinate frame where the Y direction is normal to the plane defined
    by unit vectors a and b. The other axes are arbitrary."""
    device = a.device

    # Replace every entry in b that's parallel to the corresponding entry in a with an
    # arbitrary vector.
    b = b.detach().clone()
    parallel = (einsum(a, b, "... i, ... i -> ...").abs() - 1).abs() < eps
    b[parallel] = torch.tensor([0, 0, 1], dtype=b.dtype, device=device)
    parallel = (einsum(a, b, "... i, ... i -> ...").abs() - 1).abs() < eps
    b[parallel] = torch.tensor([0, 1, 0], dtype=b.dtype, device=device)

    # Generate the coordinate frame. The initial cross product defines the plane.
    return generate_coordinate_frame(normalize(a.cross(b)), a)


def matrix_to_euler(
    rotations: Float[Tensor, "*batch 3 3"],
    pattern: str,
) -> Float[Tensor, "*batch 3"]:
    *batch, _, _ = rotations.shape
    rotations = rotations.reshape(-1, 3, 3)
    angles_np = R.from_matrix(rotations.detach().cpu().numpy()).as_euler(pattern)
    rotations = torch.tensor(angles_np, dtype=rotations.dtype, device=rotations.device)
    return rotations.reshape(*batch, 3)


def euler_to_matrix(
    rotations: Float[Tensor, "*batch 3"],
    pattern: str,
) -> Float[Tensor, "*batch 3 3"]:
    *batch, _ = rotations.shape
    rotations = rotations.reshape(-1, 3)
    matrix_np = R.from_euler(pattern, rotations.detach().cpu().numpy()).as_matrix()
    rotations = torch.tensor(matrix_np, dtype=rotations.dtype, device=rotations.device)
    return rotations.reshape(*batch, 3, 3)


def extrinsics_to_pivot_parameters(
    extrinsics: Float[Tensor, "*#batch 4 4"],
    pivot_coordinate_frame: Float[Tensor, "*#batch 3 3"],
    pivot_point: Float[Tensor, "*#batch 3"],
) -> Float[Tensor, "*batch 5"]:
    """Convert the extrinsics to a representation with 5 degrees of freedom:
    1. Distance from pivot point in the "X" (look cross pivot axis) direction.
    2. Distance from pivot point in the "Y" (pivot axis) direction.
    3. Distance from pivot point in the Z (look) direction
    4. Angle in plane
    5. Twist (rotation not in plane)
    """

    # The pivot coordinate frame's Z axis is normal to the plane.
    pivot_axis = pivot_coordinate_frame[..., :, 1]

    # Compute the translation elements of the pivot parametrization.
    translation_frame = generate_coordinate_frame(pivot_axis, extrinsics[..., :3, 2])
    origin = extrinsics[..., :3, 3]
    delta = pivot_point - origin
    translation = einsum(translation_frame, delta, "... i j, ... i -> ... j")

    # Add the rotation elements of the pivot parametrization.
    inverted = pivot_coordinate_frame.inverse() @ extrinsics[..., :3, :3]
    y, _, z = matrix_to_euler(inverted, "YXZ").unbind(dim=-1)

    return torch.cat([translation, y[..., None], z[..., None]], dim=-1)


def pivot_parameters_to_extrinsics(
    parameters: Float[Tensor, "*#batch 5"],
    pivot_coordinate_frame: Float[Tensor, "*#batch 3 3"],
    pivot_point: Float[Tensor, "*#batch 3"],
) -> Float[Tensor, "*batch 4 4"]:
    translation, y, z = parameters.split((3, 1, 1), dim=-1)

    euler = torch.cat((y, torch.zeros_like(y), z), dim=-1)
    rotation = pivot_coordinate_frame @ euler_to_matrix(euler, "YXZ")

    # The pivot coordinate frame's Z axis is normal to the plane.
    pivot_axis = pivot_coordinate_frame[..., :, 1]

    translation_frame = generate_coordinate_frame(pivot_axis, rotation[..., :3, 2])
    delta = einsum(translation_frame, translation, "... i j, ... j -> ... i")
    origin = pivot_point - delta

    *batch, _ = origin.shape
    extrinsics = torch.eye(4, dtype=parameters.dtype, device=parameters.device)
    extrinsics = extrinsics.broadcast_to((*batch, 4, 4)).clone()
    extrinsics[..., 3, 3] = 1
    extrinsics[..., :3, :3] = rotation
    extrinsics[..., :3, 3] = origin
    return extrinsics


def interpolate_circular(
    a: Float[Tensor, "*#batch"],
    b: Float[Tensor, "*#batch"],
    t: Float[Tensor, "*#batch"],
) -> Float[Tensor, " *batch"]:
    a, b, t = torch.broadcast_tensors(a, b, t)

    tau = 2 * torch.pi
    a = a % tau
    b = b % tau

    # Consider piecewise edge cases.
    d = (b - a).abs()
    a_left = a - tau
    d_left = (b - a_left).abs()
    a_right = a + tau
    d_right = (b - a_right).abs()
    use_d = (d < d_left) & (d < d_right)
    use_d_left = (d_left < d_right) & (~use_d)
    use_d_right = (~use_d) & (~use_d_left)

    result = a + (b - a) * t
    result[use_d_left] = (a_left + (b - a_left) * t)[use_d_left]
    result[use_d_right] = (a_right + (b - a_right) * t)[use_d_right]

    return result


def interpolate_pivot_parameters(
    initial: Float[Tensor, "*#batch 5"],
    final: Float[Tensor, "*#batch 5"],
    t: Float[Tensor, " time_step"],
) -> Float[Tensor, "*batch time_step 5"]:
    initial = rearrange(initial, "... d -> ... () d")
    final = rearrange(final, "... d -> ... () d")
    t = rearrange(t, "t -> t ()")
    ti, ri = initial.split((3, 2), dim=-1)
    tf, rf = final.split((3, 2), dim=-1)

    t_lerp = ti + (tf - ti) * t
    r_lerp = interpolate_circular(ri, rf, t)

    return torch.cat((t_lerp, r_lerp), dim=-1)


@torch.no_grad()
def interpolate_extrinsics(
    initial: Float[Tensor, "*#batch 4 4"],
    final: Float[Tensor, "*#batch 4 4"],
    t: Float[Tensor, " time_step"],
    eps: float = 1e-4,
) -> Float[Tensor, "*batch time_step 4 4"]:
    """Interpolate extrinsics by rotating around their "focus point," which is the
    least-squares intersection between the look vectors of the initial and final
    extrinsics.
    """

    initial = initial.type(torch.float64)
    final = final.type(torch.float64)
    t = t.type(torch.float64)

    # Based on the dot product between the look vectors, pick from one of two cases:
    # 1. Look vectors are parallel: interpolate about their origins' midpoint.
    # 3. Look vectors aren't parallel: interpolate about their focus point.
    initial_look = initial[..., :3, 2]
    final_look = final[..., :3, 2]
    dot_products = einsum(initial_look, final_look, "... i, ... i -> ...")
    parallel_mask = (dot_products.abs() - 1).abs() < eps

    # Pick focus points.
    initial_origin = initial[..., :3, 3]
    final_origin = final[..., :3, 3]
    pivot_point = 0.5 * (initial_origin + final_origin)
    pivot_point[~parallel_mask] = intersect_rays(
        initial_origin[~parallel_mask],
        initial_look[~parallel_mask],
        final_origin[~parallel_mask],
        final_look[~parallel_mask],
    )

    # Convert to pivot parameters.
    pivot_frame = generate_rotation_coordinate_frame(initial_look, final_look, eps=eps)
    initial_params = extrinsics_to_pivot_parameters(initial, pivot_frame, pivot_point)
    final_params = extrinsics_to_pivot_parameters(final, pivot_frame, pivot_point)

    # Interpolate the pivot parameters.
    interpolated_params = interpolate_pivot_parameters(initial_params, final_params, t)

    # Convert back.
    return pivot_parameters_to_extrinsics(
        interpolated_params.type(torch.float32),
        rearrange(pivot_frame, "... i j -> ... () i j").type(torch.float32),
        rearrange(pivot_point, "... xyz -> ... () xyz").type(torch.float32),
    )


```

# prompts/papers/depthsplat/src/visualization/camera_trajectory/spin.py

``` py
import numpy as np
import torch
from einops import repeat
from jaxtyping import Float
from scipy.spatial.transform import Rotation as R
from torch import Tensor


def generate_spin(
    num_frames: int,
    device: torch.device,
    elevation: float,
    radius: float,
) -> Float[Tensor, "frame 4 4"]:
    # Translate back along the camera's look vector.
    tf_translation = torch.eye(4, dtype=torch.float32, device=device)
    tf_translation[:2] *= -1
    tf_translation[2, 3] = -radius

    # Generate the transformation for the azimuth.
    phi = 2 * np.pi * (np.arange(num_frames) / num_frames)
    rotation_vectors = np.stack([np.zeros_like(phi), phi, np.zeros_like(phi)], axis=-1)

    azimuth = R.from_rotvec(rotation_vectors).as_matrix()
    azimuth = torch.tensor(azimuth, dtype=torch.float32, device=device)
    tf_azimuth = torch.eye(4, dtype=torch.float32, device=device)
    tf_azimuth = repeat(tf_azimuth, "i j -> b i j", b=num_frames).clone()
    tf_azimuth[:, :3, :3] = azimuth

    # Generate the transformation for the elevation.
    deg_elevation = np.deg2rad(elevation)
    elevation = R.from_rotvec(np.array([deg_elevation, 0, 0], dtype=np.float32))
    elevation = torch.tensor(elevation.as_matrix())
    tf_elevation = torch.eye(4, dtype=torch.float32, device=device)
    tf_elevation[:3, :3] = elevation

    return tf_azimuth @ tf_elevation @ tf_translation


```

# prompts/papers/depthsplat/src/visualization/camera_trajectory/wobble.py

``` py
import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor


@torch.no_grad()
def generate_wobble_transformation(
    radius: Float[Tensor, "*#batch"],
    t: Float[Tensor, " time_step"],
    num_rotations: int = 1,
    scale_radius_with_t: bool = True,
) -> Float[Tensor, "*batch time_step 4 4"]:
    # Generate a translation in the image plane.
    tf = torch.eye(4, dtype=torch.float32, device=t.device)
    tf = tf.broadcast_to((*radius.shape, t.shape[0], 4, 4)).clone()
    radius = radius[..., None]
    if scale_radius_with_t:
        radius = radius * t
    tf[..., 0, 3] = torch.sin(2 * torch.pi * num_rotations * t) * radius
    tf[..., 1, 3] = -torch.cos(2 * torch.pi * num_rotations * t) * radius
    return tf


@torch.no_grad()
def generate_wobble(
    extrinsics: Float[Tensor, "*#batch 4 4"],
    radius: Float[Tensor, "*#batch"],
    t: Float[Tensor, " time_step"],
) -> Float[Tensor, "*batch time_step 4 4"]:
    tf = generate_wobble_transformation(radius, t)
    return rearrange(extrinsics, "... i j -> ... () i j") @ tf


```

# prompts/papers/depthsplat/src/visualization/color_map.py

``` py
import torch
from colorspacious import cspace_convert
from einops import rearrange
from jaxtyping import Float
from matplotlib import cm
from torch import Tensor


def apply_color_map(
    x: Float[Tensor, " *batch"],
    color_map: str = "inferno",
) -> Float[Tensor, "*batch 3"]:
    cmap = cm.get_cmap(color_map)

    # Convert to NumPy so that Matplotlib color maps can be used.
    mapped = cmap(x.detach().clip(min=0, max=1).cpu().numpy())[..., :3]

    # Convert back to the original format.
    return torch.tensor(mapped, device=x.device, dtype=torch.float32)


def apply_color_map_to_image(
    image: Float[Tensor, "*batch height width"],
    color_map: str = "inferno",
) -> Float[Tensor, "*batch 3 height with"]:
    image = apply_color_map(image, color_map)
    return rearrange(image, "... h w c -> ... c h w")


def apply_color_map_2d(
    x: Float[Tensor, "*#batch"],
    y: Float[Tensor, "*#batch"],
) -> Float[Tensor, "*batch 3"]:
    red = cspace_convert((189, 0, 0), "sRGB255", "CIELab")
    blue = cspace_convert((0, 45, 255), "sRGB255", "CIELab")
    white = cspace_convert((255, 255, 255), "sRGB255", "CIELab")
    x_np = x.detach().clip(min=0, max=1).cpu().numpy()[..., None]
    y_np = y.detach().clip(min=0, max=1).cpu().numpy()[..., None]

    # Interpolate between red and blue on the x axis.
    interpolated = x_np * red + (1 - x_np) * blue

    # Interpolate between color and white on the y axis.
    interpolated = y_np * interpolated + (1 - y_np) * white

    # Convert to RGB.
    rgb = cspace_convert(interpolated, "CIELab", "sRGB1")
    return torch.tensor(rgb, device=x.device, dtype=torch.float32).clip(min=0, max=1)


```

# prompts/papers/depthsplat/src/visualization/colors.py

``` py
from PIL import ImageColor

# https://sashamaps.net/docs/resources/20-colors/
DISTINCT_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#808080",
    "#ffffff",
    "#000000",
]


def get_distinct_color(index: int) -> tuple[float, float, float]:
    hex = DISTINCT_COLORS[index % len(DISTINCT_COLORS)]
    return tuple(x / 255 for x in ImageColor.getcolor(hex, "RGB"))


```

# prompts/papers/depthsplat/src/visualization/drawing/cameras.py

``` py
from typing import Optional

import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from ...geometry.projection import unproject
from ..annotation import add_label
from .lines import draw_lines
from .types import Scalar, sanitize_scalar


def draw_cameras(
    resolution: int,
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    color: Float[Tensor, "batch 3"],
    near: Optional[Scalar] = None,
    far: Optional[Scalar] = None,
    margin: float = 0.1,  # relative to AABB
    frustum_scale: float = 0.05,  # relative to image resolution
) -> Float[Tensor, "3 3 height width"]:
    device = extrinsics.device

    # Compute scene bounds.
    minima, maxima = compute_aabb(extrinsics, intrinsics, near, far)
    scene_minima, scene_maxima = compute_equal_aabb_with_margin(
        minima, maxima, margin=margin
    )
    span = (scene_maxima - scene_minima).max()

    # Compute frustum locations.
    corner_depth = (span * frustum_scale)[None]
    frustum_corners = unproject_frustum_corners(extrinsics, intrinsics, corner_depth)
    if near is not None:
        near_corners = unproject_frustum_corners(extrinsics, intrinsics, near)
    if far is not None:
        far_corners = unproject_frustum_corners(extrinsics, intrinsics, far)

    # Project the cameras onto each axis-aligned plane.
    projections = []
    for projected_axis in range(3):
        image = torch.zeros(
            (3, resolution, resolution),
            dtype=torch.float32,
            device=device,
        )
        image_x_axis = (projected_axis + 1) % 3
        image_y_axis = (projected_axis + 2) % 3

        def project(points: Float[Tensor, "*batch 3"]) -> Float[Tensor, "*batch 2"]:
            x = points[..., image_x_axis]
            y = points[..., image_y_axis]
            return torch.stack([x, y], dim=-1)

        x_range, y_range = torch.stack(
            (project(scene_minima), project(scene_maxima)), dim=-1
        )

        # Draw near and far planes.
        if near is not None:
            projected_near_corners = project(near_corners)
            image = draw_lines(
                image,
                rearrange(projected_near_corners, "b p xy -> (b p) xy"),
                rearrange(projected_near_corners.roll(1, 1), "b p xy -> (b p) xy"),
                color=0.25,
                width=2,
                x_range=x_range,
                y_range=y_range,
            )
        if far is not None:
            projected_far_corners = project(far_corners)
            image = draw_lines(
                image,
                rearrange(projected_far_corners, "b p xy -> (b p) xy"),
                rearrange(projected_far_corners.roll(1, 1), "b p xy -> (b p) xy"),
                color=0.25,
                width=2,
                x_range=x_range,
                y_range=y_range,
            )
        if near is not None and far is not None:
            image = draw_lines(
                image,
                rearrange(projected_near_corners, "b p xy -> (b p) xy"),
                rearrange(projected_far_corners, "b p xy -> (b p) xy"),
                color=0.25,
                width=2,
                x_range=x_range,
                y_range=y_range,
            )

        # Draw the camera frustums themselves.
        projected_origins = project(extrinsics[:, :3, 3])
        projected_frustum_corners = project(frustum_corners)
        start = [
            repeat(projected_origins, "b xy -> (b p) xy", p=4),
            rearrange(projected_frustum_corners.roll(1, 1), "b p xy -> (b p) xy"),
        ]
        start = rearrange(torch.cat(start, dim=0), "(r b p) xy -> (b r p) xy", r=2, p=4)
        image = draw_lines(
            image,
            start,
            repeat(projected_frustum_corners, "b p xy -> (b r p) xy", r=2),
            color=repeat(color, "b c -> (b r p) c", r=2, p=4),
            width=2,
            x_range=x_range,
            y_range=y_range,
        )

        x_name = "XYZ"[image_x_axis]
        y_name = "XYZ"[image_y_axis]
        image = add_label(image, f"{x_name}{y_name} Projection")

        # TODO: Draw axis indicators.
        projections.append(image)

    return torch.stack(projections)


def compute_aabb(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Optional[Scalar] = None,
    far: Optional[Scalar] = None,
) -> tuple[
    Float[Tensor, "3"],  # minima of the scene
    Float[Tensor, "3"],  # maxima of the scene
]:
    """Compute an axis-aligned bounding box for the camera frustums."""

    device = extrinsics.device

    # These points are included in the AABB.
    points = [extrinsics[:, :3, 3]]

    if near is not None:
        near = sanitize_scalar(near, device)
        corners = unproject_frustum_corners(extrinsics, intrinsics, near)
        points.append(rearrange(corners, "b p xyz -> (b p) xyz"))

    if far is not None:
        far = sanitize_scalar(far, device)
        corners = unproject_frustum_corners(extrinsics, intrinsics, far)
        points.append(rearrange(corners, "b p xyz -> (b p) xyz"))

    points = torch.cat(points, dim=0)
    return points.min(dim=0).values, points.max(dim=0).values


def compute_equal_aabb_with_margin(
    minima: Float[Tensor, "*#batch 3"],
    maxima: Float[Tensor, "*#batch 3"],
    margin: float = 0.1,
) -> tuple[
    Float[Tensor, "*batch 3"],  # minima of the scene
    Float[Tensor, "*batch 3"],  # maxima of the scene
]:
    midpoint = (maxima + minima) * 0.5
    span = (maxima - minima).max() * (1 + margin)
    scene_minima = midpoint - 0.5 * span
    scene_maxima = midpoint + 0.5 * span
    return scene_minima, scene_maxima


def unproject_frustum_corners(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    depth: Float[Tensor, "#batch"],
) -> Float[Tensor, "batch 4 3"]:
    device = extrinsics.device

    # Get coordinates for the corners. Following them in a circle makes a rectangle.
    xy = torch.linspace(0, 1, 2, device=device)
    xy = torch.stack(torch.meshgrid(xy, xy, indexing="xy"), dim=-1)
    xy = rearrange(xy, "i j xy -> (i j) xy")
    xy = xy[torch.tensor([0, 1, 3, 2], device=device)]

    # Get ray directions in camera space.
    directions = unproject(
        xy,
        torch.ones(1, dtype=torch.float32, device=device),
        rearrange(intrinsics, "b i j -> b () i j"),
    )

    # Divide by the z coordinate so that multiplying by depth will produce orthographic
    # depth (z depth) as opposed to Euclidean depth (distance from the camera).
    directions = directions / directions[..., -1:]
    directions = einsum(extrinsics[..., :3, :3], directions, "b i j, b r j -> b r i")

    origins = rearrange(extrinsics[:, :3, 3], "b xyz -> b () xyz")
    depth = rearrange(depth, "b -> b () ()")
    return origins + depth * directions


```

# prompts/papers/depthsplat/src/visualization/drawing/coordinate_conversion.py

``` py
from typing import Optional, Protocol, runtime_checkable

import torch
from jaxtyping import Float
from torch import Tensor

from .types import Pair, sanitize_pair


@runtime_checkable
class ConversionFunction(Protocol):
    def __call__(
        self,
        xy: Float[Tensor, "*batch 2"],
    ) -> Float[Tensor, "*batch 2"]:
        pass


def generate_conversions(
    shape: tuple[int, int],
    device: torch.device,
    x_range: Optional[Pair] = None,
    y_range: Optional[Pair] = None,
) -> tuple[
    ConversionFunction,  # conversion from world coordinates to pixel coordinates
    ConversionFunction,  # conversion from pixel coordinates to world coordinates
]:
    h, w = shape
    x_range = sanitize_pair((0, w) if x_range is None else x_range, device)
    y_range = sanitize_pair((0, h) if y_range is None else y_range, device)
    minima, maxima = torch.stack((x_range, y_range), dim=-1)
    wh = torch.tensor((w, h), dtype=torch.float32, device=device)

    def convert_world_to_pixel(
        xy: Float[Tensor, "*batch 2"],
    ) -> Float[Tensor, "*batch 2"]:
        return (xy - minima) / (maxima - minima) * wh

    def convert_pixel_to_world(
        xy: Float[Tensor, "*batch 2"],
    ) -> Float[Tensor, "*batch 2"]:
        return xy / wh * (maxima - minima) + minima

    return convert_world_to_pixel, convert_pixel_to_world


```

# prompts/papers/depthsplat/src/visualization/drawing/lines.py

``` py
from typing import Literal, Optional

import torch
from einops import einsum, repeat
from jaxtyping import Float
from torch import Tensor

from .coordinate_conversion import generate_conversions
from .rendering import render_over_image
from .types import Pair, Scalar, Vector, sanitize_scalar, sanitize_vector


def draw_lines(
    image: Float[Tensor, "3 height width"],
    start: Vector,
    end: Vector,
    color: Vector,
    width: Scalar,
    cap: Literal["butt", "round", "square"] = "round",
    num_msaa_passes: int = 1,
    x_range: Optional[Pair] = None,
    y_range: Optional[Pair] = None,
) -> Float[Tensor, "3 height width"]:
    device = image.device
    start = sanitize_vector(start, 2, device)
    end = sanitize_vector(end, 2, device)
    color = sanitize_vector(color, 3, device)
    width = sanitize_scalar(width, device)
    (num_lines,) = torch.broadcast_shapes(
        start.shape[0],
        end.shape[0],
        color.shape[0],
        width.shape,
    )

    # Convert world-space points to pixel space.
    _, h, w = image.shape
    world_to_pixel, _ = generate_conversions((h, w), device, x_range, y_range)
    start = world_to_pixel(start)
    end = world_to_pixel(end)

    def color_function(
        xy: Float[Tensor, "point 2"],
    ) -> Float[Tensor, "point 4"]:
        # Define a vector between the start and end points.
        delta = end - start
        delta_norm = delta.norm(dim=-1, keepdim=True)
        u_delta = delta / delta_norm

        # Define a vector between each sample and the start point.
        indicator = xy - start[:, None]

        # Determine whether each sample is inside the line in the parallel direction.
        extra = 0.5 * width[:, None] if cap == "square" else 0
        parallel = einsum(u_delta, indicator, "l xy, l s xy -> l s")
        parallel_inside_line = (parallel <= delta_norm + extra) & (parallel > -extra)

        # Determine whether each sample is inside the line perpendicularly.
        perpendicular = indicator - parallel[..., None] * u_delta[:, None]
        perpendicular_inside_line = perpendicular.norm(dim=-1) < 0.5 * width[:, None]

        inside_line = parallel_inside_line & perpendicular_inside_line

        # Compute round caps.
        if cap == "round":
            near_start = indicator.norm(dim=-1) < 0.5 * width[:, None]
            inside_line |= near_start
            end_indicator = indicator = xy - end[:, None]
            near_end = end_indicator.norm(dim=-1) < 0.5 * width[:, None]
            inside_line |= near_end

        # Determine the sample's color.
        selectable_color = color.broadcast_to((num_lines, 3))
        arrangement = inside_line * torch.arange(num_lines, device=device)[:, None]
        top_color = selectable_color.gather(
            dim=0,
            index=repeat(arrangement.argmax(dim=0), "s -> s c", c=3),
        )
        rgba = torch.cat((top_color, inside_line.any(dim=0).float()[:, None]), dim=-1)

        return rgba

    return render_over_image(image, color_function, device, num_passes=num_msaa_passes)


```

# prompts/papers/depthsplat/src/visualization/drawing/points.py

``` py
from typing import Optional

import torch
from einops import repeat
from jaxtyping import Float
from torch import Tensor

from .coordinate_conversion import generate_conversions
from .rendering import render_over_image
from .types import Pair, Scalar, Vector, sanitize_scalar, sanitize_vector


def draw_points(
    image: Float[Tensor, "3 height width"],
    points: Vector,
    color: Vector = [1, 1, 1],
    radius: Scalar = 1,
    inner_radius: Scalar = 0,
    num_msaa_passes: int = 1,
    x_range: Optional[Pair] = None,
    y_range: Optional[Pair] = None,
) -> Float[Tensor, "3 height width"]:
    device = image.device
    points = sanitize_vector(points, 2, device)
    color = sanitize_vector(color, 3, device)
    radius = sanitize_scalar(radius, device)
    inner_radius = sanitize_scalar(inner_radius, device)
    (num_points,) = torch.broadcast_shapes(
        points.shape[0],
        color.shape[0],
        radius.shape,
        inner_radius.shape,
    )

    # Convert world-space points to pixel space.
    _, h, w = image.shape
    world_to_pixel, _ = generate_conversions((h, w), device, x_range, y_range)
    points = world_to_pixel(points)

    def color_function(
        xy: Float[Tensor, "point 2"],
    ) -> Float[Tensor, "point 4"]:
        # Define a vector between the start and end points.
        delta = xy[:, None] - points[None]
        delta_norm = delta.norm(dim=-1)
        mask = (delta_norm >= inner_radius[None]) & (delta_norm <= radius[None])

        # Determine the sample's color.
        selectable_color = color.broadcast_to((num_points, 3))
        arrangement = mask * torch.arange(num_points, device=device)
        top_color = selectable_color.gather(
            dim=0,
            index=repeat(arrangement.argmax(dim=1), "s -> s c", c=3),
        )
        rgba = torch.cat((top_color, mask.any(dim=1).float()[:, None]), dim=-1)

        return rgba

    return render_over_image(image, color_function, device, num_passes=num_msaa_passes)


```

# prompts/papers/depthsplat/src/visualization/drawing/rendering.py

``` py
from typing import Protocol, runtime_checkable

import torch
from einops import rearrange, reduce
from jaxtyping import Bool, Float
from torch import Tensor


@runtime_checkable
class ColorFunction(Protocol):
    def __call__(
        self,
        xy: Float[Tensor, "point 2"],
    ) -> Float[Tensor, "point 4"]:  # RGBA color
        pass


def generate_sample_grid(
    shape: tuple[int, int],
    device: torch.device,
) -> Float[Tensor, "height width 2"]:
    h, w = shape
    x = torch.arange(w, device=device) + 0.5
    y = torch.arange(h, device=device) + 0.5
    x, y = torch.meshgrid(x, y, indexing="xy")
    return torch.stack([x, y], dim=-1)


def detect_msaa_pixels(
    image: Float[Tensor, "batch 4 height width"],
) -> Bool[Tensor, "batch height width"]:
    b, _, h, w = image.shape

    mask = torch.zeros((b, h, w), dtype=torch.bool, device=image.device)

    # Detect horizontal differences.
    horizontal = (image[:, :, :, 1:] != image[:, :, :, :-1]).any(dim=1)
    mask[:, :, 1:] |= horizontal
    mask[:, :, :-1] |= horizontal

    # Detect vertical differences.
    vertical = (image[:, :, 1:, :] != image[:, :, :-1, :]).any(dim=1)
    mask[:, 1:, :] |= vertical
    mask[:, :-1, :] |= vertical

    # Detect diagonal (top left to bottom right) differences.
    tlbr = (image[:, :, 1:, 1:] != image[:, :, :-1, :-1]).any(dim=1)
    mask[:, 1:, 1:] |= tlbr
    mask[:, :-1, :-1] |= tlbr

    # Detect diagonal (top right to bottom left) differences.
    trbl = (image[:, :, :-1, 1:] != image[:, :, 1:, :-1]).any(dim=1)
    mask[:, :-1, 1:] |= trbl
    mask[:, 1:, :-1] |= trbl

    return mask


def reduce_straight_alpha(
    rgba: Float[Tensor, "batch 4 height width"],
) -> Float[Tensor, "batch 4"]:
    color, alpha = rgba.split((3, 1), dim=1)

    # Color becomes a weighted average of color (weighted by alpha).
    weighted_color = reduce(color * alpha, "b c h w -> b c", "sum")
    alpha_sum = reduce(alpha, "b c h w -> b c", "sum")
    color = weighted_color / (alpha_sum + 1e-10)

    # Alpha becomes mean alpha.
    alpha = reduce(alpha, "b c h w -> b c", "mean")

    return torch.cat((color, alpha), dim=-1)


@torch.no_grad()
def run_msaa_pass(
    xy: Float[Tensor, "batch height width 2"],
    color_function: ColorFunction,
    scale: float,
    subdivision: int,
    remaining_passes: int,
    device: torch.device,
    batch_size: int = int(2**16),
) -> Float[Tensor, "batch 4 height width"]:  # color (RGBA with straight alpha)
    # Sample the color function.
    b, h, w, _ = xy.shape
    color = [
        color_function(batch)
        for batch in rearrange(xy, "b h w xy -> (b h w) xy").split(batch_size)
    ]
    color = torch.cat(color, dim=0)
    color = rearrange(color, "(b h w) c -> b c h w", b=b, h=h, w=w)

    # If any MSAA passes remain, subdivide.
    if remaining_passes > 0:
        mask = detect_msaa_pixels(color)
        batch_index, row_index, col_index = torch.where(mask)
        xy = xy[batch_index, row_index, col_index]

        offsets = generate_sample_grid((subdivision, subdivision), device)
        offsets = (offsets / subdivision - 0.5) * scale

        color_fine = run_msaa_pass(
            xy[:, None, None] + offsets,
            color_function,
            scale / subdivision,
            subdivision,
            remaining_passes - 1,
            device,
            batch_size=batch_size,
        )
        color[batch_index, :, row_index, col_index] = reduce_straight_alpha(color_fine)

    return color


@torch.no_grad()
def render(
    shape: tuple[int, int],
    color_function: ColorFunction,
    device: torch.device,
    subdivision: int = 8,
    num_passes: int = 2,
) -> Float[Tensor, "4 height width"]:  # color (RGBA with straight alpha)
    xy = generate_sample_grid(shape, device)
    return run_msaa_pass(
        xy[None],
        color_function,
        1.0,
        subdivision,
        num_passes,
        device,
    )[0]


def render_over_image(
    image: Float[Tensor, "3 height width"],
    color_function: ColorFunction,
    device: torch.device,
    subdivision: int = 8,
    num_passes: int = 1,
) -> Float[Tensor, "3 height width"]:
    _, h, w = image.shape
    overlay = render(
        (h, w),
        color_function,
        device,
        subdivision=subdivision,
        num_passes=num_passes,
    )
    color, alpha = overlay.split((3, 1), dim=0)
    return image * (1 - alpha) + color * alpha


```

# prompts/papers/depthsplat/src/visualization/drawing/types.py

``` py
from typing import Iterable, Union

import torch
from einops import repeat
from jaxtyping import Float, Shaped
from torch import Tensor

Real = Union[float, int]

Vector = Union[
    Real,
    Iterable[Real],
    Shaped[Tensor, "3"],
    Shaped[Tensor, "batch 3"],
]


def sanitize_vector(
    vector: Vector,
    dim: int,
    device: torch.device,
) -> Float[Tensor, "*#batch dim"]:
    if isinstance(vector, Tensor):
        vector = vector.type(torch.float32).to(device)
    else:
        vector = torch.tensor(vector, dtype=torch.float32, device=device)
    while vector.ndim < 2:
        vector = vector[None]
    if vector.shape[-1] == 1:
        vector = repeat(vector, "... () -> ... c", c=dim)
    assert vector.shape[-1] == dim
    assert vector.ndim == 2
    return vector


Scalar = Union[
    Real,
    Iterable[Real],
    Shaped[Tensor, ""],
    Shaped[Tensor, " batch"],
]


def sanitize_scalar(scalar: Scalar, device: torch.device) -> Float[Tensor, "*#batch"]:
    if isinstance(scalar, Tensor):
        scalar = scalar.type(torch.float32).to(device)
    else:
        scalar = torch.tensor(scalar, dtype=torch.float32, device=device)
    while scalar.ndim < 1:
        scalar = scalar[None]
    assert scalar.ndim == 1
    return scalar


Pair = Union[
    Iterable[Real],
    Shaped[Tensor, "2"],
]


def sanitize_pair(pair: Pair, device: torch.device) -> Float[Tensor, "2"]:
    if isinstance(pair, Tensor):
        pair = pair.type(torch.float32).to(device)
    else:
        pair = torch.tensor(pair, dtype=torch.float32, device=device)
    assert pair.shape == (2,)
    return pair


```

# prompts/papers/depthsplat/src/visualization/layout.py

``` py
"""This file contains useful layout utilities for images. They are:

- add_border: Add a border to an image.
- cat/hcat/vcat: Join images by arranging them in a line. If the images have different
  sizes, they are aligned as specified (start, end, center). Allows you to specify a gap
  between images.

Images are assumed to be float32 tensors with shape (channel, height, width).
"""

from typing import Any, Generator, Iterable, Literal, Optional, Union

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

Alignment = Literal["start", "center", "end"]
Axis = Literal["horizontal", "vertical"]
Color = Union[
    int,
    float,
    Iterable[int],
    Iterable[float],
    Float[Tensor, "#channel"],
    Float[Tensor, ""],
]


def _sanitize_color(color: Color) -> Float[Tensor, "#channel"]:
    # Convert tensor to list (or individual item).
    if isinstance(color, torch.Tensor):
        color = color.tolist()

    # Turn iterators and individual items into lists.
    if isinstance(color, Iterable):
        color = list(color)
    else:
        color = [color]

    return torch.tensor(color, dtype=torch.float32)


def _intersperse(iterable: Iterable, delimiter: Any) -> Generator[Any, None, None]:
    it = iter(iterable)
    yield next(it)
    for item in it:
        yield delimiter
        yield item


def _get_main_dim(main_axis: Axis) -> int:
    return {
        "horizontal": 2,
        "vertical": 1,
    }[main_axis]


def _get_cross_dim(main_axis: Axis) -> int:
    return {
        "horizontal": 1,
        "vertical": 2,
    }[main_axis]


def _compute_offset(base: int, overlay: int, align: Alignment) -> slice:
    assert base >= overlay
    offset = {
        "start": 0,
        "center": (base - overlay) // 2,
        "end": base - overlay,
    }[align]
    return slice(offset, offset + overlay)


def overlay(
    base: Float[Tensor, "channel base_height base_width"],
    overlay: Float[Tensor, "channel overlay_height overlay_width"],
    main_axis: Axis,
    main_axis_alignment: Alignment,
    cross_axis_alignment: Alignment,
) -> Float[Tensor, "channel base_height base_width"]:
    # The overlay must be smaller than the base.
    _, base_height, base_width = base.shape
    _, overlay_height, overlay_width = overlay.shape
    assert base_height >= overlay_height and base_width >= overlay_width

    # Compute spacing on the main dimension.
    main_dim = _get_main_dim(main_axis)
    main_slice = _compute_offset(
        base.shape[main_dim], overlay.shape[main_dim], main_axis_alignment
    )

    # Compute spacing on the cross dimension.
    cross_dim = _get_cross_dim(main_axis)
    cross_slice = _compute_offset(
        base.shape[cross_dim], overlay.shape[cross_dim], cross_axis_alignment
    )

    # Combine the slices and paste the overlay onto the base accordingly.
    selector = [..., None, None]
    selector[main_dim] = main_slice
    selector[cross_dim] = cross_slice
    result = base.clone()
    result[selector] = overlay
    return result


def cat(
    main_axis: Axis,
    *images: Iterable[Float[Tensor, "channel _ _"]],
    align: Alignment = "center",
    gap: int = 8,
    gap_color: Color = 1,
) -> Float[Tensor, "channel height width"]:
    """Arrange images in a line. The interface resembles a CSS div with flexbox."""
    device = images[0].device
    gap_color = _sanitize_color(gap_color).to(device)

    # Find the maximum image side length in the cross axis dimension.
    cross_dim = _get_cross_dim(main_axis)
    cross_axis_length = max(image.shape[cross_dim] for image in images)

    # Pad the images.
    padded_images = []
    for image in images:
        # Create an empty image with the correct size.
        padded_shape = list(image.shape)
        padded_shape[cross_dim] = cross_axis_length
        base = torch.ones(padded_shape, dtype=torch.float32, device=device)
        base = base * gap_color[:, None, None]
        padded_images.append(overlay(base, image, main_axis, "start", align))

    # Intersperse separators if necessary.
    if gap > 0:
        # Generate a separator.
        c, _, _ = images[0].shape
        separator_size = [gap, gap]
        separator_size[cross_dim - 1] = cross_axis_length
        separator = torch.ones((c, *separator_size), dtype=torch.float32, device=device)
        separator = separator * gap_color[:, None, None]

        # Intersperse the separator between the images.
        padded_images = list(_intersperse(padded_images, separator))

    return torch.cat(padded_images, dim=_get_main_dim(main_axis))


def hcat(
    *images: Iterable[Float[Tensor, "channel _ _"]],
    align: Literal["start", "center", "end", "top", "bottom"] = "start",
    gap: int = 8,
    gap_color: Color = 1,
):
    """Shorthand for a horizontal linear concatenation."""
    return cat(
        "horizontal",
        *images,
        align={
            "start": "start",
            "center": "center",
            "end": "end",
            "top": "start",
            "bottom": "end",
        }[align],
        gap=gap,
        gap_color=gap_color,
    )


def vcat(
    *images: Iterable[Float[Tensor, "channel _ _"]],
    align: Literal["start", "center", "end", "left", "right"] = "start",
    gap: int = 8,
    gap_color: Color = 1,
):
    """Shorthand for a horizontal linear concatenation."""
    return cat(
        "vertical",
        *images,
        align={
            "start": "start",
            "center": "center",
            "end": "end",
            "left": "start",
            "right": "end",
        }[align],
        gap=gap,
        gap_color=gap_color,
    )


def add_border(
    image: Float[Tensor, "channel height width"],
    border: int = 8,
    color: Color = 1,
) -> Float[Tensor, "channel new_height new_width"]:
    color = _sanitize_color(color).to(image)
    c, h, w = image.shape
    result = torch.empty(
        (c, h + 2 * border, w + 2 * border), dtype=torch.float32, device=image.device
    )
    result[:] = color[:, None, None]
    result[:, border : h + border, border : w + border] = image
    return result


def resize(
    image: Float[Tensor, "channel height width"],
    shape: Optional[tuple[int, int]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Float[Tensor, "channel new_height new_width"]:
    assert (shape is not None) + (width is not None) + (height is not None) == 1
    _, h, w = image.shape

    if width is not None:
        shape = (int(h * width / w), width)
    elif height is not None:
        shape = (height, int(w * height / h))

    return F.interpolate(
        image[None],
        shape,
        mode="bilinear",
        align_corners=False,
        antialias="bilinear",
    )[0]


```

# prompts/papers/depthsplat/src/visualization/validation_in_3d.py

``` py
import torch
from jaxtyping import Float, Shaped
from torch import Tensor

from ..model.decoder.cuda_splatting import render_cuda_orthographic
from ..model.types import Gaussians
from ..visualization.annotation import add_label
from ..visualization.drawing.cameras import draw_cameras
from .drawing.cameras import compute_equal_aabb_with_margin


def pad(images: list[Shaped[Tensor, "..."]]) -> list[Shaped[Tensor, "..."]]:
    shapes = torch.stack([torch.tensor(x.shape) for x in images])
    padded_shape = shapes.max(dim=0)[0]
    results = [
        torch.ones(padded_shape.tolist(), dtype=x.dtype, device=x.device)
        for x in images
    ]
    for image, result in zip(images, results):
        slices = [slice(0, x) for x in image.shape]
        result[slices] = image[slices]
    return results


def render_projections(
    gaussians: Gaussians,
    resolution: int,
    margin: float = 0.1,
    draw_label: bool = True,
    extra_label: str = "",
) -> Float[Tensor, "batch 3 3 height width"]:
    device = gaussians.means.device
    b, _, _ = gaussians.means.shape

    # Compute the minima and maxima of the scene.
    minima = gaussians.means.min(dim=1).values
    maxima = gaussians.means.max(dim=1).values
    scene_minima, scene_maxima = compute_equal_aabb_with_margin(
        minima, maxima, margin=margin
    )

    projections = []
    for look_axis in range(3):
        right_axis = (look_axis + 1) % 3
        down_axis = (look_axis + 2) % 3

        # Define the extrinsics for rendering.
        extrinsics = torch.zeros((b, 4, 4), dtype=torch.float32, device=device)
        extrinsics[:, right_axis, 0] = 1
        extrinsics[:, down_axis, 1] = 1
        extrinsics[:, look_axis, 2] = 1
        extrinsics[:, right_axis, 3] = 0.5 * (
            scene_minima[:, right_axis] + scene_maxima[:, right_axis]
        )
        extrinsics[:, down_axis, 3] = 0.5 * (
            scene_minima[:, down_axis] + scene_maxima[:, down_axis]
        )
        extrinsics[:, look_axis, 3] = scene_minima[:, look_axis]
        extrinsics[:, 3, 3] = 1

        # Define the intrinsics for rendering.
        extents = scene_maxima - scene_minima
        far = extents[:, look_axis]
        near = torch.zeros_like(far)
        width = extents[:, right_axis]
        height = extents[:, down_axis]

        projection = render_cuda_orthographic(
            extrinsics,
            width,
            height,
            near,
            far,
            (resolution, resolution),
            torch.zeros((b, 3), dtype=torch.float32, device=device),
            gaussians.means,
            gaussians.covariances,
            gaussians.harmonics,
            gaussians.opacities,
            fov_degrees=10.0,
        )
        if draw_label:
            right_axis_name = "XYZ"[right_axis]
            down_axis_name = "XYZ"[down_axis]
            label = f"{right_axis_name}{down_axis_name} Projection {extra_label}"
            projection = torch.stack([add_label(x, label) for x in projection])

        projections.append(projection)

    return torch.stack(pad(projections), dim=1)


def render_cameras(batch: dict, resolution: int) -> Float[Tensor, "3 3 height width"]:
    # Define colors for context and target views.
    num_context_views = batch["context"]["extrinsics"].shape[1]
    num_target_views = batch["target"]["extrinsics"].shape[1]
    color = torch.ones(
        (num_target_views + num_context_views, 3),
        dtype=torch.float32,
        device=batch["target"]["extrinsics"].device,
    )
    color[num_context_views:, 1:] = 0

    return draw_cameras(
        resolution,
        torch.cat(
            (batch["context"]["extrinsics"][0], batch["target"]["extrinsics"][0])
        ),
        torch.cat(
            (batch["context"]["intrinsics"][0], batch["target"]["intrinsics"][0])
        ),
        color,
        torch.cat((batch["context"]["near"][0], batch["target"]["near"][0])),
        torch.cat((batch["context"]["far"][0], batch["target"]["far"][0])),
    )


```

# prompts/papers/depthsplat/src/visualization/vis_depth.py

``` py
import torch
import torch.utils.data
import numpy as np
import torchvision.utils as vutils
import cv2
from matplotlib.cm import get_cmap
import matplotlib as mpl
import matplotlib.cm as cm


# https://github.com/autonomousvision/unimatch/blob/master/utils/visualization.py


def vis_disparity(disp):
    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis = disp_vis.astype("uint8")
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_INFERNO)

    return disp_vis


def viz_depth_tensor(disp, return_numpy=False, colormap="plasma"):
    # visualize inverse depth
    assert isinstance(disp, torch.Tensor)

    disp = disp.numpy()
    vmax = np.percentile(disp, 95)
    normalizer = mpl.colors.Normalize(vmin=disp.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=colormap)
    colormapped_im = (mapper.to_rgba(disp)[:, :, :3] * 255).astype(
        np.uint8
    )  # [H, W, 3]

    if return_numpy:
        return colormapped_im

    viz = torch.from_numpy(colormapped_im).permute(2, 0, 1)  # [3, H, W]

    return viz


```

