# prompts/papers/depthsplat/config/dataset/dl3dv.yaml

``` yaml
defaults:
  - view_sampler: boundedv2_360

name: dl3dv
roots: [datasets/dl3dv]
make_baseline_1: false
augment: true

image_shape: [270, 480]
background_color: [0.0, 0.0, 0.0]
cameras_are_circular: false

baseline_epsilon: 1e-3
max_fov: 100.0

skip_bad_shape: true
near: -1.
far: -1.
baseline_scale_bounds: false
shuffle_val: true
test_len: -1
test_chunk_interval: 1
sort_target_index: true
sort_context_index: true

train_times_per_scene: 1
test_times_per_scene: 1
ori_image_shape: [270, 480]
overfit_max_views: 148
use_index_to_load_chunk: false

mix_tartanair: false
no_mix_test_set: true
load_depth: false


```

# prompts/papers/depthsplat/config/dataset/re10k.yaml

``` yaml
defaults:
  - view_sampler: bounded

name: re10k
roots: [datasets/re10k]
make_baseline_1: false
augment: true

image_shape: [180, 320]
background_color: [0.0, 0.0, 0.0]
cameras_are_circular: false

baseline_epsilon: 1e-3
max_fov: 100.0

skip_bad_shape: true
near: -1.
far: -1.
baseline_scale_bounds: true
shuffle_val: true
test_len: -1
test_chunk_interval: 1

use_index_to_load_chunk: false

```

# prompts/papers/depthsplat/config/dataset/view_sampler/all.yaml

``` yaml
name: all


```

# prompts/papers/depthsplat/config/dataset/view_sampler/arbitrary.yaml

``` yaml
name: arbitrary

num_target_views: 1
num_context_views: 2

# If you want to hard-code context views, do so here.
context_views: null


```

# prompts/papers/depthsplat/config/dataset/view_sampler/bounded.yaml

``` yaml
name: bounded

num_target_views: 1
num_context_views: 2

min_distance_between_context_views: 2
max_distance_between_context_views: 6
min_distance_to_context_views: 0

warm_up_steps: 0
initial_min_distance_between_context_views: 2
initial_max_distance_between_context_views: 6

```

# prompts/papers/depthsplat/config/dataset/view_sampler/boundedv2.yaml

``` yaml
name: boundedv2

num_target_views: 1
num_context_views: 2

min_distance_between_context_views: 2
max_distance_between_context_views: 6
max_distance_to_context_views: 0

context_gap_warm_up_steps: 0
target_gap_warm_up_steps: 0

initial_min_distance_between_context_views: 2
initial_max_distance_between_context_views: 6
initial_max_distance_to_context_views: 0


```

# prompts/papers/depthsplat/config/dataset/view_sampler/boundedv2_360.yaml

``` yaml
name: boundedv2

num_target_views: 4
num_context_views: 4

min_distance_between_context_views: 20
max_distance_between_context_views: 50
max_distance_to_context_views: 0

context_gap_warm_up_steps: 10000
target_gap_warm_up_steps: 0

initial_min_distance_between_context_views: 15
initial_max_distance_between_context_views: 30
initial_max_distance_to_context_views: 0
extra_views_sampling_strategy: farthest_point
target_views_replace_sample: false


```

# prompts/papers/depthsplat/config/dataset/view_sampler/evaluation.yaml

``` yaml
name: evaluation

index_path: assets/evaluation_index_re10k_video.json
num_context_views: 2


```

# prompts/papers/depthsplat/config/dataset/view_sampler_dataset_specific_config/bounded_re10k.yaml

``` yaml
# @package _global_

dataset:
  view_sampler:
    min_distance_between_context_views: 45
    max_distance_between_context_views: 135
    min_distance_to_context_views: 0
    warm_up_steps: 30000
    initial_min_distance_between_context_views: 25
    initial_max_distance_between_context_views: 45
    num_target_views: 4


```

# prompts/papers/depthsplat/config/dataset/view_sampler_dataset_specific_config/boundedv2_dl3dv.yaml

``` yaml
# @package _global_

dataset:
  view_sampler:
    min_distance_between_context_views: 20
    max_distance_between_context_views: 50
    max_distance_to_context_views: 0
    context_gap_warm_up_steps: 10000 
    target_gap_warm_up_steps: 0
    initial_min_distance_between_context_views: 15
    initial_max_distance_between_context_views: 30
    initial_max_distance_to_context_views: 0
    extra_views_sampling_strategy: farthest_point
    num_target_views: 4


```

# prompts/papers/depthsplat/config/dataset/view_sampler_dataset_specific_config/evaluation_dl3dv.yaml

``` yaml
# @package _global_

dataset:
  view_sampler:
    index_path: assets/dl3dv_360_v5.json


```

# prompts/papers/depthsplat/config/dataset/view_sampler_dataset_specific_config/evaluation_re10k.yaml

``` yaml
# @package _global_

dataset:
  view_sampler:
    index_path: assets/evaluation_index_re10k.json


```

# prompts/papers/depthsplat/config/experiment/dl3dv.yaml

``` yaml
# @package _global_

defaults:
  - override /dataset: dl3dv
  - override /model/encoder: depthsplat
  - override /loss: [mse, lpips]
  - override /dataset/view_sampler: boundedv2_360

wandb:
  name: dl3dv
  tags: [dl3dv, 270x480]

data_loader:
  train:
    batch_size: 1

trainer:
  max_steps: 300_001
  num_nodes: 1

model:
  encoder:
    num_depth_candidates: 128
    costvolume_unet_feat_dim: 128
    costvolume_unet_channel_mult: [1,1,1]
    costvolume_unet_attn_res: [4]
    gaussians_per_pixel: 1
    depth_unet_feat_dim: 32
    depth_unet_attn_res: [16]
    depth_unet_channel_mult: [1,1,1,1,1]
    shim_patch_size: 16

# lpips loss
loss:
  lpips:
    apply_after_step: 0
    weight: 0.05

dataset: 
  near: 0.5
  far: 200.
  baseline_scale_bounds: false
  make_baseline_1: false
  min_views: 0
  max_views: 0
  highres: false

test:
  eval_time_skip_steps: 0
  compute_scores: true
  dec_chunk_size: 30


```

# prompts/papers/depthsplat/config/experiment/re10k.yaml

``` yaml
# @package _global_

defaults:
  - override /dataset: re10k
  - override /model/encoder: depthsplat
  - override /loss: [mse, lpips]

wandb:
  name: re10k
  tags: [re10k, 256x256]

data_loader:
  train:
    batch_size: 14

trainer:
  max_steps: 300_001
  num_nodes: 1

model:
  encoder:
    num_depth_candidates: 128
    costvolume_unet_feat_dim: 128
    costvolume_unet_channel_mult: [1,1,1]
    costvolume_unet_attn_res: [4]
    gaussians_per_pixel: 1
    depth_unet_feat_dim: 32
    depth_unet_attn_res: [16]
    depth_unet_channel_mult: [1,1,1,1,1]

# lpips loss
loss:
  lpips:
    apply_after_step: 0
    weight: 0.05

dataset: 
  image_shape: [256, 256]
  roots: [datasets/re10k]
  near: 0.5
  far: 100.
  baseline_scale_bounds: false
  make_baseline_1: false
  train_times_per_scene: 1
  highres: false

test:
  eval_time_skip_steps: 5
  compute_scores: true


```

# prompts/papers/depthsplat/config/loss/lpips.yaml

``` yaml
lpips:
  weight: 0.05
  apply_after_step: 150_000


```

# prompts/papers/depthsplat/config/loss/mse.yaml

``` yaml
mse:
  weight: 1.0


```

# prompts/papers/depthsplat/config/main.yaml

``` yaml
defaults:
  - dataset: re10k
  - optional dataset/view_sampler_dataset_specific_config: ${dataset/view_sampler}_${dataset}
  - model/encoder: depthsplat
  - model/decoder: splatting_cuda
  - loss: [mse]

wandb:
  project: depthsplat
  entity: placeholder
  name: placeholder
  mode: online
  id: null

mode: train

dataset:
  overfit_to_scene: null

data_loader:
  train:
    num_workers: 10
    persistent_workers: true
    batch_size: 4
    seed: 1234
  test:
    num_workers: 4
    persistent_workers: false
    batch_size: 1
    seed: 2345
  val:
    num_workers: 1
    persistent_workers: true
    batch_size: 1
    seed: 3456

optimizer:
  lr: 2.e-4
  lr_monodepth: 2.e-6
  warm_up_steps: 2000
  weight_decay: 0.01

checkpointing:
  load: null
  every_n_train_steps: 5000
  save_top_k: 5
  pretrained_model: null
  pretrained_monodepth: null
  pretrained_mvdepth: null
  pretrained_depth: null
  no_strict_load: false
  resume: false

train:
  depth_mode: null
  extended_visualization: false
  print_log_every_n_steps: 100
  eval_model_every_n_val: 2  # quantitative evaluation every n val
  eval_data_length: 999999
  eval_deterministic: false
  eval_time_skip_steps: 3
  eval_save_model: true
  l1_loss: false
  intermediate_loss_weight: 0.9
  no_viz_video: false
  viz_depth: false
  forward_depth_only: false
  train_ignore_large_loss: 0.
  no_log_projections: false

test:
  output_path: outputs/test
  compute_scores: true
  eval_time_skip_steps: 0
  save_image: false
  save_video: false
  save_gt_image: false
  save_input_images: false
  save_depth: false
  save_depth_npy: false
  save_depth_concat_img: false
  save_gaussian: false
  render_chunk_size: null
  stablize_camera: false
  stab_camera_kernel: 50

seed: 111123

trainer:
  max_steps: -1
  val_check_interval: 0.5
  gradient_clip_val: 0.5
  num_sanity_val_steps: 2

output_dir: outputs/tmp

use_plugins: false


```

# prompts/papers/depthsplat/config/model/decoder/splatting_cuda.yaml

``` yaml
name: splatting_cuda


```

# prompts/papers/depthsplat/config/model/encoder/depthsplat.yaml

``` yaml
name: depthsplat

num_depth_candidates: 128
num_surfaces: 1

gaussians_per_pixel: 1

gaussian_adapter:
  gaussian_scale_min: 1e-10
  gaussian_scale_max: 3.
  sh_degree: 2

d_feature: 128

visualizer:
  num_samples: 8
  min_resolution: 256
  export_ply: false

unimatch_weights_path: "pretrained/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth"
multiview_trans_attn_split: 2
costvolume_unet_feat_dim: 128
costvolume_unet_channel_mult: [1,1,1]
costvolume_unet_attn_res: []
depth_unet_feat_dim: 64
depth_unet_attn_res: []
depth_unet_channel_mult: [1, 1, 1]
downscale_factor: 4
shim_patch_size: 4

local_mv_match: 2

# monodepth
monodepth_vit_type: vits

# return depth
supervise_intermediate_depth: true
return_depth: true

# mv_unimatch
num_scales: 1
upsample_factor: 4
lowest_feature_resolution: 4
depth_unet_channels: 128
grid_sample_disable_cudnn: false

# depthsplat color branch
large_gaussian_head: false
color_large_unet: false
init_sh_input_img: true
feature_upsampler_channels: 64
gaussian_regressor_channels: 64

# only depth
train_depth_only: false

```

