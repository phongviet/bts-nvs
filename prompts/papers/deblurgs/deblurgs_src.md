# prompts/papers/deblurgs/arguments/__init__.py

``` py
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 2
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.llffhold = 0
        self.num_initial_pcd = -1
        
        self.num_subframes = 21
        self.curve_order = 9
        self.curve_type = "se3" # ["quarternion_cartesian", "se3"]
        
        self.z_near = 0.2
        self.z_far = 100.0

        self.random_init = False
        self.alpha_lower_bound=0.0 # rm
        self.scale_lb=0.0 # rm
        self.scale_ub=-1.0 # rm
        self.tone_mapping_type = "gamma"
        self.activation = "relu"
        self.use_isotrophic = False
        self.curve_random_sample = False
        
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 150_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.noise_init = 0.0
        self.noise_final = 0.0 
        self.lambda_t_smooth_init = 1e-3
        self.lambda_t_smooth_final = 1e-5 
        
        self.lambda_depth_tv = 0.0
        self.lambda_hinge = 0.1
        
        self.densification_interval = 200
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 75_000
        self.densify_grad_threshold_init = 4e-4
        self.densify_grad_threshold_final = 2e-4
        self.densify_annealing_until = 25_000
        self.clip_grad = -1.0
                
        # curve optimization factors.
        self.curve_controlpoints_lr = 1e-2
        self.curve_rotation_lr = 1e-3
        self.curve_alignment_lr = 0.0 # 3e-3
        self.curve_alignment_start = 30_000
        self.curve_lr_half_iter = 15_000
        self.curve_start_iter = 1000
        self.curve_end_iter = 100_000
        self.random_sample_until = 100000
        self.drop_alignment = 1.0
        
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)


```

# prompts/papers/deblurgs/gaussian_renderer/__init__.py

``` py
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        z_near = pc.z_near,
        z_far = pc.z_far,
        use_sigmoid=pc.use_sigmoid, 
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # No precomputed cov3D.
    cov3D_precomp = None
    scales = pc.get_scaling
    rotations = pc.get_rotation
 
    # No precomputed color
    shs = None
    colors_precomp = None
    if override_color is None:
        shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, rendered_depth, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "depth": rendered_depth, 
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


```

# prompts/papers/deblurgs/lpipsPyTorch/__init__.py

``` py
import torch

from .modules.lpips import LPIPS


def lpips(x: torch.Tensor,
          y: torch.Tensor,
          net_type: str = 'alex',
          version: str = '0.1'):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    device = x.device
    criterion = LPIPS(net_type, version).to(device)
    return criterion(x, y)


```

# prompts/papers/deblurgs/lpipsPyTorch/modules/lpips.py

``` py
import torch
import torch.nn as nn

from .networks import get_network, LinLayers
from .utils import get_state_dict


class LPIPS(nn.Module):
    r"""Creates a criterion that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        net_type (str): the network type to compare the features: 
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    def __init__(self, net_type: str = 'alex', version: str = '0.1'):

        assert version in ['0.1'], 'v0.1 is only supported now'

        super(LPIPS, self).__init__()

        # pretrained network
        self.net = get_network(net_type)

        # linear layers
        self.lin = LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(get_state_dict(net_type, version))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        feat_x, feat_y = self.net(x), self.net(y)

        diff = [(fx - fy) ** 2 for fx, fy in zip(feat_x, feat_y)]
        res = [l(d).mean((2, 3), True) for d, l in zip(diff, self.lin)]

        return torch.sum(torch.cat(res, 0), 0, True)


```

# prompts/papers/deblurgs/lpipsPyTorch/modules/networks.py

``` py
from typing import Sequence

from itertools import chain

import torch
import torch.nn as nn
from torchvision import models

from .utils import normalize_activation


def get_network(net_type: str):
    if net_type == 'alex':
        return AlexNet()
    elif net_type == 'squeeze':
        return SqueezeNet()
    elif net_type == 'vgg':
        return VGG16()
    else:
        raise NotImplementedError('choose net_type from [alex, squeeze, vgg].')


class LinLayers(nn.ModuleList):
    def __init__(self, n_channels_list: Sequence[int]):
        super(LinLayers, self).__init__([
            nn.Sequential(
                nn.Identity(),
                nn.Conv2d(nc, 1, 1, 1, 0, bias=False)
            ) for nc in n_channels_list
        ])

        for param in self.parameters():
            param.requires_grad = False


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        # register buffer
        self.register_buffer(
            'mean', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer(
            'std', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def set_requires_grad(self, state: bool):
        for param in chain(self.parameters(), self.buffers()):
            param.requires_grad = state

    def z_score(self, x: torch.Tensor):
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor):
        x = self.z_score(x)

        output = []
        for i, (_, layer) in enumerate(self.layers._modules.items(), 1):
            x = layer(x)
            if i in self.target_layers:
                output.append(normalize_activation(x))
            if len(output) == len(self.target_layers):
                break
        return output


class SqueezeNet(BaseNet):
    def __init__(self):
        super(SqueezeNet, self).__init__()

        self.layers = models.squeezenet1_1(True).features
        self.target_layers = [2, 5, 8, 10, 11, 12, 13]
        self.n_channels_list = [64, 128, 256, 384, 384, 512, 512]

        self.set_requires_grad(False)


class AlexNet(BaseNet):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.layers = models.alexnet(True).features
        self.target_layers = [2, 5, 8, 10, 12]
        self.n_channels_list = [64, 192, 384, 256, 256]

        self.set_requires_grad(False)


class VGG16(BaseNet):
    def __init__(self):
        super(VGG16, self).__init__()

        self.layers = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.target_layers = [4, 9, 16, 23, 30]
        self.n_channels_list = [64, 128, 256, 512, 512]

        self.set_requires_grad(False)


```

# prompts/papers/deblurgs/lpipsPyTorch/modules/utils.py

``` py
from collections import OrderedDict

import torch


def normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def get_state_dict(net_type: str = 'alex', version: str = '0.1'):
    # build url
    url = 'https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/' \
        + f'master/lpips/weights/v{version}/{net_type}.pth'

    # download
    old_state_dict = torch.hub.load_state_dict_from_url(
        url, progress=True,
        map_location=None if torch.cuda.is_available() else torch.device('cpu')
    )

    # rename keys
    new_state_dict = OrderedDict()
    for key, val in old_state_dict.items():
        new_key = key
        new_key = new_key.replace('lin', '')
        new_key = new_key.replace('model.', '')
        new_state_dict[new_key] = val

    return new_state_dict


```

# prompts/papers/deblurgs/metrics.py

``` py
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)


```

# prompts/papers/deblurgs/render_spiral.py

``` py
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import ModelParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from utils.export_utils import get_render_path, make_video

def render_set(model_path,  gaussians:GaussianModel, scene:Scene , background, args):
    
    views = get_render_path(scene=scene, 
                            spin_for=args.spin_for)
    imgs = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, background)
        img = scene.tone_mapping(render_pkg['render'])
        imgs.append(img)
    imgs = torch.stack(imgs)
    imgs = (imgs.permute(0,2,3,1).cpu().numpy().clip(0.0,1.0) * 255.0 ).astype(np.uint8)

    make_video(imgs, os.path.join(model_path, "render_img.mp4"), args.fps)


@torch.no_grad()
def render_sets(dataset: ModelParams, iteration : int, args):
    
    # [HARDCODING] If hold exists, forcefully turn on the eval mode.
    data_path = dataset.source_path
    if len( [e for e in os.listdir(data_path) if "hold" in e] ) == 1:
        dataset.eval= True
    
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, curve_model=True)
    scene.camera_motion_module.load(dataset.model_path)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_set(dataset.model_path, gaussians, scene, background, args)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=False)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fps", default=32, type=int)
    parser.add_argument("--spin_for", default=2, type=int)
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    render_sets(model.extract(args), args.iteration, args)

```

# prompts/papers/deblurgs/render_trainview.py

``` py
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from gaussian_renderer import render
from argparse import ArgumentParser
from arguments import ModelParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from utils.export_utils import make_video, center_crop_with_ratio

def render_set(model_path,  gaussians:GaussianModel, scene:Scene , background, args):
    
    views = scene.camera_motion_module.get_middle_cams()
    imgs = []
    gts = []

    start_idx = args.start_index
    length = args.fps * args.duration
    crop_ratio = args.crop_ratio
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if start_idx<=idx<=start_idx+length:
            render_pkg = render(view, gaussians, background)
            img = scene.tone_mapping(render_pkg['render'])
            gt = scene.getTrainCameras()[idx].original_image
            imgs.append(img)
            gts.append(gt)
        
    imgs = torch.stack(imgs)
    gts = torch.stack(gts)

    imgs = (imgs.permute(0,2,3,1).cpu().numpy().clip(0.0,1.0) * 255.0 ).astype(np.uint8)
    gts = (gts.permute(0,2,3,1).cpu().numpy().clip(0.0,1.0) * 255.0 ).astype(np.uint8)

    imgs = center_crop_with_ratio(imgs, ratio=crop_ratio)
    gts = center_crop_with_ratio(gts, ratio=crop_ratio)

    make_video(imgs, os.path.join(model_path, "render_trainview_img.mp4"), args.fps)
    make_video(gts, os.path.join(model_path, "render_trainview_gt.mp4"), args.fps)
    make_video(np.concatenate([gts,imgs],axis=2), os.path.join(model_path, "render_trainview_all.mp4"), args.fps)


@torch.no_grad()
def render_sets(dataset: ModelParams, iteration : int, args):
    
    # [HARDCODING] If hold exists, forcefully turn on the eval mode.
    data_path = dataset.source_path
    if len( [e for e in os.listdir(data_path) if "hold" in e] ) == 1:
        dataset.eval= True
    
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, curve_model=True)
    scene.camera_motion_module.load(dataset.model_path)

    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_set(dataset.model_path, gaussians, scene, background, args)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=False)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--fps", default=10, type=int)
    parser.add_argument("--start_index", default=0, type=int)
    parser.add_argument("--duration", default=20.0, type=float)
    parser.add_argument("--crop_ratio", default=0.95, type=float)
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    render_sets(model.extract(args), args.iteration, args)

```

# prompts/papers/deblurgs/scene/__init__.py

``` py
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

from scene.motion import CameraMotionModule
from scene.tonemapping import ToneMapping
from scene.gaussian_model import BasicPointCloud
from utils.camera_utils import Camera

import numpy as np

class Scene:

    gaussians : GaussianModel
    
    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], curve_model=True, load_path=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        if os.path.exists(os.path.join(args.source_path, "sparse")) or os.path.exists(os.path.join(args.source_path, "poses_bounds.npy")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"


        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if curve_model:
            self.camera_motion_module = CameraMotionModule(cam_infos=scene_info.train_cameras, args=args)
            self.camera_motion_module.link_gaussian(gaussians=gaussians)
        else:
            self.camera_motion_module = None
        self.cam_order = []

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(self.cam_order)
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            # print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
       
        if load_path:
            loaded_iter = searchForMaxIteration(os.path.join(load_path, "point_cloud"))
            ply_path = os.path.join(load_path,"point_cloud",f"iteration_{loaded_iter}","point_cloud.ply")

            self.gaussians.load_ply(ply_path)
        elif self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            # scene_info = scene_info._replace(point_cloud=pcd_filter(scene_info.point_cloud, self.getTrainCameras() ))
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
        
        # Tone Mapping Function.
        self.tone_mapping = ToneMapping(args.tone_mapping_type)
        

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    
    def get_random_cam_idx(self):
        """
        randomly choose cam idx.

        """
        if len(self.cam_order) == 0:
            self.cam_order = list(range(len(self.camera_motion_module)))
            random.shuffle(self.cam_order)
        
        idx = self.cam_order.pop()
        return idx

```

# prompts/papers/deblurgs/scene/bezier.py

``` py


import torch
import torch.nn as nn
import roma
import numpy as np
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import CameraInfo
from arguments import OptimizationParams
from arguments import ModelParams
import utils.pytorch3d_functions as torch3d
from scene.cameras import Camera, MiniCam
from utils.camera_utils import cameraList_from_camInfos
from scipy.spatial.transform import Rotation
import open3d as o3d
import os
import scipy.special
from utils.general_utils import inverse_sigmoid
import random


class BezierModel(nn.Module):

    def __init__(self, initial_points, curve_order, initial_noise=0.001):
        """
        ARGUMENTS
        ---------
        initial_points: torch tensor [n,d]. Starting points for curves.
            where n is number of curves
            and d is dimension of the space.
            e.g.) 5 curves on cartesian space: 
                needs shape of [5,3]
        curve_order (int):
           order of bezier curve.
           set to 1 for linear model.
        initial_noise (float): initial noise for optimization.
        """ 
        super().__init__()
        
        self.curve_order = curve_order
        
        initial_points = initial_points.float().cuda() #  [n,d]
        initial_points = initial_points[:,None,:].repeat(1, curve_order+1, 1) # [n,c+1,d]
        initial_points = initial_points + torch.randn_like(initial_points)*initial_noise # [n,c+1,d]

        self._control_points = nn.Parameter(initial_points.clone().contiguous().requires_grad_(True))
    
        self._bezier_binom_coeff = torch.tensor([scipy.special.binom(self.curve_order, k) for k in range(self.curve_order+1)]).cuda() # [C+1]

    @property
    def device(self):
        return self._control_points.device
    
    def _get_bezier_coeff(self, t):
        """
        ARGUMENTS
        ---------
        t: tensor size of [f, ], ranging from 0.0 to 1.0
        """
        C = self.curve_order

        coeff = (t[:,None] ** torch.arange(C,-1,-1, device=self.device)) * ( (1-t)[:,None] ** torch.arange(0,C+1, device=self.device) ) * self._bezier_binom_coeff # [f, C+1]
        
        return coeff
    
    def forward(self, t:torch.Tensor, idx:int):
        """
        ARGUMENTS
        ---------
        t: tensor [num_samples]
            float tensor in the range of [0,1]
        idx: curve idx.
        
        RETURNS
        -------
        sample_points: [num_samples, dimension]
        """
        if isinstance(idx, int):
            idx = torch.tensor([idx],device=self.device)
        
        sample_points = (self._get_bezier_coeff(t)[:,:,None] * self._control_points[idx]).sum(dim=1) # [f,c+1,1] * [c+1,d] = [f,c+1,d]--(sum)-->[f,d]
        
        return sample_points

    def __len__(self):
        return self._control_points.shape[0]

```

# prompts/papers/deblurgs/scene/cameras.py

``` py
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from scene.tonemapping import ToneMapping
import copy

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", depth=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
        self.original_depth = depth
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


def get_c2w(cam:Camera, want_numpy=True):
    """
    Get MVG convention c2w matrix from Camera object.
    
    ARGUMENTS
    ---------
    cam: camera object
    want_numpy: If True, returns numpy, otherwise tensor object.

    RETURNS
    -------
    c2w: (4,4) np array or [4,4] tensor, depending on your option.
    """
    if want_numpy:
        c2w = np.eye(4)
        c2w[:3,:3] = cam.world_view_transform[:3,:3].cpu().numpy()
        c2w[:3,3] = cam.camera_center.cpu().numpy()
    else:
        raise NotImplementedError
    
    return c2w

def c2w_to_cam(ref_cam:Camera, c2w):
    
    device = ref_cam.world_view_transform.device
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w)
    
    rot = c2w[:3,:3]
    trans = c2w[:3,3]

    world_view_transform = torch.eye(4, device=device)
    world_view_transform[:3,:3] = rot # NOTE rot.T.T 
    world_view_transform[3,:3] = -trans@rot # NOTE: not [:3,3] for world-view transform.
    
    
    cam = copy.deepcopy(ref_cam)
        
    cam.world_view_transform = world_view_transform
    cam.projection_matrix = getProjectionMatrix(znear=cam.znear, zfar=cam.zfar, fovX=cam.FoVx, fovY=cam.FoVy).transpose(0,1).cuda()
    cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
    cam.camera_center = cam.world_view_transform.inverse()[3, :3]
   
    return cam

```

# prompts/papers/deblurgs/scene/colmap_loader.py

``` py
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import collections
import struct

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1


    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                count += 1

    return xyzs, rgbs, errors

def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """


    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors

def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_colmap_bin_array(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_dense.py

    :param path: path to the colmap binary file.
    :return: nd array with the floating point values in the value
    """
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


```

# prompts/papers/deblurgs/scene/dataset_readers.py

``` py
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text

from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB, RGB2SH
from scene.gaussian_model import BasicPointCloud

from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

from scene.pcd_init import random_pcd_init

import copy
import open3d as o3d


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info, pcd):

    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3])
    cam_centers = np.stack(cam_centers) # (n,3)
    if pcd is not None:
        xyzs = pcd.points
        center = xyzs.mean(axis=0)
        dist = np.linalg.norm(cam_centers-center, axis=1)
        radius1 = np.percentile(dist, 10.0) # heuristic
    else:
        dist_matrix = np.linalg.norm(cam_centers - cam_centers[:,None,:] , axis=-1) # (n,n,3) -> (n,n)
        radius1 = np.percentile(dist_matrix,90)
        
    def get_center_and_diag(cam_centers):
        cam_centers = np.stack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=0, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=1)
        diagonal = np.max(dist)
        return center.flatten(), diagonal


    center, diagonal = get_center_and_diag(cam_centers)
    radius2 = diagonal * 1.1

    radius = min(radius1, radius2)
    print(f"pcd-cam radius : {radius1:.2f}")
    print(f"cam-center radius : {radius2:.2f}")
    print(f"Scene Radius = {radius:.2f}")

    return {"translate": None, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    
    permu_idx = [0 for _ in cam_extrinsics]

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        try:
            image_path = os.path.join(images_folder, os.path.basename(extr.name))
            image_name = os.path.basename(image_path).split(".")[0]
            image = Image.open(image_path)
        except FileNotFoundError:
            image_path = image_path[:-4]+".jpg"
            image = Image.open(image_path)
        
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, depth=None)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

# TODO move to somewhere else.    
def get_bds(cam_infos, pcd):
    """
    cam infos
    pcd: (n_pts,3)
    hwf: (3)
    
    RETURNS
    ------
    bds: (n_cam,2)
    """
    h = cam_infos[0].height
    w = cam_infos[0].width
    fx = fov2focal(cam_infos[0].FovX, w)
    fy = fov2focal(cam_infos[0].FovY, h)
    
    K = np.array([[fx,0.0,w/2],[0.0,fy,h/2],[0.0,0.0,1.0]])

    bds = []
    for cam_info in cam_infos:
        
        w2c = np.eye(4)
        w2c[:3,:3] = cam_info.R.T
        w2c[:3,3] = cam_info.T

        pcd_homog = np.pad(pcd,((0,0),(0,1)),mode='constant',constant_values=1.0) # (n,4)
        
        cam_coords = (pcd_homog @ w2c.T)[:,:3] # (n,3)

        depths = cam_coords[:,2] # (n)
        valid = depths>0.01 # (n)
        
        pixel_coords_homog = cam_coords @ np.linalg.inv(K) # (n,3)
        pixel_coords = pixel_coords_homog[:,:2] / pixel_coords_homog[:,2:] # (n,2)

        valid = np.logical_and( valid, pixel_coords[:,0] >= 0)
        valid = np.logical_and( valid, pixel_coords[:,0] < w)
        valid = np.logical_and( valid, pixel_coords[:,1] >= 0)
        valid = np.logical_and( valid, pixel_coords[:,1] < h)

        depths = depths[valid]
        
        near = np.percentile(depths, 0.1)
        far = np.percentile(depths, 99.9)
        bds.append([near,far])
    
    return np.array(bds)
        
def readColmapSceneInfo(args):
    path = args.source_path
    images = args.images
    eval = args.eval
    llffhold = args.llffhold

    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    # If llffhold is not specified, try locating "hold=n" file. If such file is detected, use it.
    if llffhold == 0:
        maybe_llff_file = [e for e in os.listdir(path) if "hold=" in e]
        assert len(maybe_llff_file) <= 1, "more than two llffhold indicator detected."
        if len(maybe_llff_file):
            llffhold = int( (maybe_llff_file[0].strip().split("="))[-1] )
            print(f"LLFF Hold is not specified, but we can detect indiactor file: llffhold={llffhold}")

    depths = None
    for i,cam_info in enumerate(cam_infos):
        image_id = int(''.join(c for c in cam_infos[i].image_name if c.isdigit())) # extract numeric part only.
        cam_infos[i] = cam_infos[i]._replace(depth=depths[image_id] if depths is not None else None)
    
    if eval and llffhold>0:
        train_cam_infos = [cam_info for idx, cam_info in enumerate(cam_infos) if int(cam_info.image_name) % llffhold != 0]
        test_cam_infos = [cam_info for idx, cam_info in enumerate(cam_infos) if int(cam_info.image_name) % llffhold == 0]
    else:
        if llffhold > 0 or eval:
            print("[ERROR] One of eval and llffhold is set, while the other is off. Check if something is wrong.")
            exit(1)
        train_cam_infos = cam_infos
        test_cam_infos = []


    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    # if not args.random_init:
    # if not os.path.exists(ply_path):
    print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    try:
        xyz, rgb, error = read_points3D_binary(bin_path)
    except:
        xyz, rgb, error = read_points3D_text(txt_path)

    # [Prune high error pcds]
    if args.num_initial_pcd > 0:

        error = error.reshape((-1,))
        percent = min( args.num_initial_pcd / xyz.shape[0] * 100, 100.0)
        error_filter_threshold = np.percentile(error, percent)
        valid_idx = error < error_filter_threshold
        
        xyz = xyz[valid_idx]
        rgb = rgb[valid_idx]
    
    storePly(ply_path, xyz, rgb)
    if args.random_init:
        ply_path = os.path.join(path, "sparse/0/points3D_random_init.ply")
        # if not os.path.exists(ply_path):
            # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        # xyz = np.random.random((num_pts, 3)) * radius * 2 - radius + center[None,:]
        bound_near = (args.z_far-args.z_near)*0.01
        bound_far = (args.z_far-args.z_near)*0.30
        bds = get_bds(train_cam_infos, xyz)
        xyz = random_pcd_init(train_cam_infos, near=args.z_near + bound_near, far=args.z_far - bound_far, num_pcd=num_pts, bds=bds)
        shs = RGB2SH(np.ones((num_pts, 3))*0.01, use_sigmoid=args.activation=="sigmoid")
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs, use_sigmoid=args.activation=="sigmoid"), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs, use_sigmoid=args.activation=="sigmoid") * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    nerf_normalization = getNerfppNorm(train_cam_infos, pcd=None if args.random_init else pcd)

    # filter_pcd(pcd, train_cam_infos)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],depth=None))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", center= np.array([0., 0. ,0.]), radius=1.3):
    # np.array([12.164,-4.05, 10.7]) np.array([0., 0. ,0.])
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []


    ply_path = os.path.join(path, "points3d.ply")

    
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        # xyz = np.random.random((num_pts, 3)) * radius * 2 - radius + center[None,:]
        xyz = random_pcd_init(train_cam_infos, near=2.0, far=8.0, num_pcd=num_pts)
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
        
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    
    nerf_normalization = getNerfppNorm(train_cam_infos, pcd=None)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


        

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}

```

# prompts/papers/deblurgs/scene/gaussian_activation.py

``` py

import torch
import torch.nn as nn

from utils.general_utils import inverse_sigmoid

class LowerBoundSigmoid(nn.Module):
    def __init__(self,lower_bound):
        super().__init__()
        self.lower_bound = lower_bound
    
    def forward(self, x):
        # alias.
        lb = self.lower_bound

        return torch.sigmoid(x) * (1.0 - lb) + lb
    
class InverseLowerBoundSigmoid(nn.Module):
    def __init__(self,lower_bound):
        super().__init__()
        self.lower_bound = lower_bound
    
    def forward(self, x):
        # alias.
        lb = self.lower_bound

        return inverse_sigmoid((x - lb) / (1.0 - lb) )
    
class Clamp(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        # alias.
        return x.clamp(0.0,1.0)
    
class InverseClamp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.clamp(0.0,1.0)

class LowerBoundExponent(nn.Module):
    def __init__(self,lower_bound):
        super().__init__()
        self.lower_bound = lower_bound
    
    def forward(self, x):
        # alias.
        lb = self.lower_bound

        return torch.exp(x)+lb

class LowerBoundLog(nn.Module):
    def __init__(self,lower_bound):
        super().__init__()
        self.lower_bound = lower_bound
        self.eps = 0.001

    def forward(self, x):
        # alias.
        lb = self.lower_bound

        return torch.log( (x-lb).clamp_min(self.eps) )

class BoundSigmoid(nn.Module):
    def __init__(self, lb, ub):
        super().__init__()
        self.lb, self.ub = lb, ub
    
    def forward(self, x):
        # alias.
        lb, ub = self.lb, self.ub

        return torch.sigmoid(x) / (ub-lb) + lb
    
class InverseBoundSigmoid(nn.Module):
    def __init__(self,lb, ub):
        super().__init__()
        self.lb, self.ub = lb, ub
        self.eps = (ub-lb)*0.001
    
    def forward(self, x):
        # alias.
        lb, ub = self.lb, self.ub

        return inverse_sigmoid( ((x-lb) * (ub-lb)).clamp(self.eps, 1.0-self.eps))

class InverseSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        ret = torch.zeros_like(x)
        ret[x>=20] = x[x>=20]
        ret[x<20] = torch.log(torch.expm1(x[x<20]))
        return ret


```

# prompts/papers/deblurgs/scene/gaussian_model.py

``` py
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from arguments import ModelParams
import torch.nn.functional as F
from scene.gaussian_activation import LowerBoundExponent, LowerBoundLog, BoundSigmoid, InverseBoundSigmoid, Clamp, InverseClamp

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        if self.scale_upper_bound < 0.0:
            # self.scaling_activation = F.softplus
            # self.scaling_inverse_activation = InverseSoftplus()
            self.scaling_activation = LowerBoundExponent(self.scale_lower_bound)
            self.scaling_inverse_activation = LowerBoundLog(self.scale_lower_bound)
        else:
            self.scaling_activation = BoundSigmoid(self.scale_lower_bound, self.scale_upper_bound)
            self.scaling_inverse_activation = InverseBoundSigmoid(self.scale_lower_bound, self.scale_upper_bound)
        self.covariance_activation = build_covariance_from_scaling_rotation


        self.opacity_activation = Clamp()
        self.inverse_opacity_activation = InverseClamp()

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, scene_args:ModelParams):
        self.active_sh_degree = 0
        self.max_sh_degree = scene_args.sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.z_near = scene_args.z_near
        self.z_far = scene_args.z_far

        self.alpha_lower_bound = scene_args.alpha_lower_bound
        self.scale_lower_bound = scene_args.scale_lb
        self.scale_upper_bound = scene_args.scale_ub

        self.use_isotrophic = scene_args.use_isotrophic
        self.use_sigmoid = scene_args.activation == "sigmoid"
        
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        if self.use_isotrophic:
            scaling = self._scaling[:,:1]
            return self.scaling_activation(scaling.expand(-1,3))
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH((inverse_sigmoid(torch.tensor(np.asarray(pcd.colors))) if self.use_sigmoid else torch.tensor(np.asarray(pcd.colors))).float().cuda(), self.use_sigmoid)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = self.scaling_inverse_activation(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        lb = self.alpha_lower_bound # Alias.
        
        opacities = self.inverse_opacity_activation( lb + (1.0-lb)*(0.1*torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda") ) )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"}, 
            # {'params': [self._features_rest], 'lr': 0.0, "name": "f_rest"}, # HARDCODING.
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    max_steps=training_args.iterations)
        

    def update_learning_rate(self, iteration, opt, alignment_lr=None):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # return lr
            elif param_group['name'] in ["curve_rot","curve_trans"] and iteration >= opt.curve_start_iter:
                param_group['lr'] = param_group['lr'] * (0.5)**(1/opt.curve_lr_half_iter)

            # HARDCODING
            elif alignment_lr is not None and param_group['name'] == "curve_alignment":
                param_group['lr'] = alignment_lr



    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacity_for_save = inverse_sigmoid(self.get_opacity)
        opacities = opacity_for_save.detach().cpu().numpy()
        scale_for_save = torch.log(self.get_scaling)
        scale = scale_for_save.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self, new_opacity:float = None):
        if new_opacity is None:
            lb = self.alpha_lower_bound 
            new_opacity = lb + (1-lb) * self.opacity_activation(torch.ones(1,device=self._opacity.device) * 0.1)
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*new_opacity))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]


    def load_ply(self, path):
        plydata = PlyData.read(path)
        # TODO implement for lowerbound scale load

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities_loaded = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        opacities = self.inverse_opacity_activation( torch.sigmoid(torch.from_numpy(opacities_loaded) ) )
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(opacities.float().cuda().requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "curve_" in group["name"]:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
    def light_prune(self,mask):

        self._xyz = self._xyz[mask]
        self._features_dc = self._features_dc[mask]
        self._features_rest = self._features_rest[mask]
        self._opacity = self._opacity[mask]
        self._scaling = self._scaling[mask]
        self._rotation = self._rotation[mask]
        
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] not in tensors_dict:
                continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, extent):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        min_opacity = self.alpha_lower_bound + (1 - self.alpha_lower_bound) * 0.005
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        self.prune_points(prune_mask)
    
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, denom_count):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += denom_count
    
    @torch.no_grad()
    def decay_opacity(self,r):
        self._opacity.data = self.inverse_opacity_activation(self.get_opacity*r)


```

# prompts/papers/deblurgs/scene/motion.py

``` py

import os

import torch
import torch.nn as nn
import roma
from scene.gaussian_model import GaussianModel
from scene.dataset_readers import CameraInfo
from scene.bezier import BezierModel
from arguments import ModelParams
import utils.pytorch3d_functions as torch3d
from scene.cameras import Camera, MiniCam
from utils.camera_utils import cameraList_from_camInfos
import gaussian_renderer
from scene.gaussian_activation import inverse_sigmoid

class CameraMotionModule:
    
    def __init__(self, cam_infos:list, args:ModelParams):
        
        """
        Aliases:
        C : curve order
        n : # imgs.
        f : # subframes.
        """
        print("Initializing Curve Model...")
        
        C = self.curve_order = args.curve_order
        f = self.n_subframes = args.num_subframes
        self.curve_type = args.curve_type
        self.curve_random_sample = args.curve_random_sample
        self.gaussians = None

        self.original_cam = cameraList_from_camInfos(cam_infos, 
                                                     resolution_scale=1, 
                                                     args=args)
        rotations = []
        translations = []
        
        for cam_info in cam_infos:
            cam_info: CameraInfo

            rotations.append(torch.from_numpy(cam_info.R)) # originally transposed
            translations.append(torch.from_numpy(-cam_info.T@cam_info.R.T)) # translation vec of c2w: cam location.

        rotations = torch.stack(rotations).cuda()
        translations = torch.stack(translations).cuda() 

        # Initial Parameters
        self._set_initial_parameters(rotations, translations) 

        # Alignment Parameters
        n = len(self)
        self._nu = nn.Parameter( inverse_sigmoid( torch.linspace(1/(f-1), 1.0-(1/(f-1)), f-2)[None,:].repeat(n,1).cuda()).contiguous().requires_grad_(True) )
    
    def link_gaussian(self, gaussians:GaussianModel):
        """
        Register GaussianModel Object.
        """
        self.gaussians = gaussians

    def add_training_setup(self, gaussians:GaussianModel, lr_dict:dict):
        """
        Extend gaussianmodel optimizer
        by adding pose_optimizer parameters.
        """
        for group in gaussians.optimizer.param_groups:
            if "curve_" in group['name'] and group['params'][0] in gaussians.optimizer.state:
                del gaussians.optimizer.state[group['params'][0]]
        gaussians.optimizer.param_groups = [e for e in gaussians.optimizer.param_groups if 'curve_' not in e['name']]
        
        gaussians.optimizer.add_param_group({'params': self._rot.parameters(),   'lr': lr_dict['curve_rot'], 'name': 'curve_rot'})
        if hasattr(self, "_trans"):
            gaussians.optimizer.add_param_group({'params': self._trans.parameters(),   'lr': lr_dict['curve_trans'], 'name': 'curve_trans'})
        gaussians.optimizer.add_param_group({'params': [self._nu],  'lr': lr_dict['curve_alignment'], 'name': 'curve_alignment'})

    def query(self, cam_idx:int, 
                    subframe_indice="all", 
                    post_process=None, 
                    background="random"):
        """
        Main query method.
        Render a blurry view, and retrieve additional queried data.

        ARGUMENTS
        ---------
        - cam_idx: int
            camera index

        - subframe_indice: "all", list[int] or int
            If "all" (Default), render all subframes.
            If list(or iterable) of int, this indicates subframe indice.
            If int, this indicates the number of subframes to be rendered; indice are evenly-spaced.
        
        - post_process: None or Callable.
            Postprocess (e.g. gamma-correction) for blurry view. Do nothing if None.
        
        - background: "random" or torch.tensor[3]
            background color. random or color in [0.0, 1.0]
        
        RETURNS
        -------
        - retrieved: dictionary
             dictionary of answered query, whose keys are
            - 'blurred': synthesized blurry view. Post_process will be applied here.
            - 'gt': gt observation. (Default)
            - 'subframes': all subframe renderings.
            - 'render_pkgs': list of render_pkgs from 3DGS render function.
            - 'depths': all subframe depth renderings.
        """ 

        assert hasattr(self, "gaussians") and isinstance(self.gaussians, GaussianModel)

        gaussians = self.gaussians
        
        # Configure background color.
        if background == "random":
            bg = torch.rand(3,device=gaussians._xyz.device)
        else:
            bg = background

        # Configure sub-frame cams.
        if subframe_indice == "all":
            subframe_cams = self.get_trajectory(cam_idx)
        else:
            nu = self._sample_nu_from_alignment(cam_idx)
            if isinstance(subframe_indice, int):
                if subframe_indice == 1:
                    subfr_idx = [len(self)//2]
                subfr_idx = torch.linspace(0,nu.shape[0]-1, subframe_indice, device=nu.device).long()
            else:
                subfr_idx = subframe_indice
            nu = nu[subfr_idx]
            subframe_cams = self.get_trajectory(cam_idx, nu)
        
            
        # Main code for render.
        render_pkg_subframes = []
        
        for cam in subframe_cams:
            render_pkg = gaussian_renderer.render(cam, gaussians, bg)
            render_pkg_subframes.append(render_pkg)        
        
        render_subframes = torch.stack([render_pkg['render'] for render_pkg in render_pkg_subframes]) # [f,3,h,w], f is num_subframes.

        # Return Values
        retrieved_dic = {}

        blurred = render_subframes.mean(dim=0) # [3,h,w]
        if post_process is not None:
            blurred = post_process(blurred)
        
        retrieved_dic['blurred'] = blurred
        retrieved_dic['gt'] = self.get_gt_image(cam_idx) # [3,h,w]
        retrieved_dic['subframes'] = render_subframes
        retrieved_dic['depths'] = torch.stack([render_pkg['depth'] for render_pkg in render_pkg_subframes])
        retrieved_dic['render_pkgs'] = render_pkg_subframes
    
        return retrieved_dic

    def get_trajectory(self, idx, t=None):
        """
        idx: int
        t: None or torch.tensor of size [f (#_of_frames)]. 
           (tensor of) position on the trajectory in the range of [0,1].
           if None, sample from alignment parameter "t" of this model.
        RETURN
        ------
        list of MiniCam type objects (which can be used in rasterization later.)
        corresponding to camera idx.
        """

        # sample subframe c2w_rotations, c2w_translations.
        rot_interp, trans_interp = self._sample_c2w_from_nu(idx, t)

        # Convert to list of Minicam objects, and returns.
        return self._c2w_to_minicam(rot_interp, trans_interp, self.original_cam[0])
    
    def _set_initial_parameters(self, rotations, translations):
        """
        set initial parameters.

        ARGUMENTS
        ---------
        rotations: rotation part of c2w matrix [n, 3, 3]
        translations: camera origin (or equivalently, translation part of c2w matrix [n,3])
        """
        n = rotations.shape[0]

        if self.curve_type == "quarternion_cartesian":
            rot_params = roma.rotmat_to_unitquat(rotations) # [n,4]
            self._rot = BezierModel(rot_params, self.curve_order)
            self._trans = BezierModel(translations, self.curve_order, initial_noise=0.01)

        elif self.curve_type == "se3":
            # NOTE: transpose for torch3d convention
            c2w = torch.zeros(n,4,4).cuda()
            c2w[:,:3,:3] = rotations.transpose(-2,-1)
            c2w[:,3,:3] = translations
            c2w[:,3,3] = 1.0

            params = torch3d.se3_log_map(c2w) 
            self._rot = BezierModel(params[:,3:], self.curve_order)
            self._trans = BezierModel(params[:,:3], self.curve_order)
        else:
            raise NotImplementedError
        
    def _sample_nu_from_alignment(self, idx):
        
        device = self._nu.device
        
        nu_mid = torch.sigmoid(self._nu[idx]) # [f-2]
        if self.curve_random_sample:
            nu_mid = nu_mid + torch.rand_like(nu_mid) / self.n_subframes - (1/(2*self.n_subframes)) # add some "uncertainty"
     

        # return nu_mid.sort().values # HARDCODING.
        return torch.cat([torch.zeros(1, device=device), nu_mid, torch.ones(1, device=device)]).clamp(0.0, 1.0).sort().values # [f]

    def _sample_c2w_from_nu(self, idx, nu=None):
        """
        ARGUMENTS
        ---------
        idx: curve index.
        t: Tensor of shape [num_subframes,], ranging in [0.0,1.0] 

        RETURNS
        -------
        c2w_rotations: Tensor of shape [num_subframes, 3, 3] 
        c2w_translations: Tensor of shape [num_subframes, 3]
        """
        
        if nu is None:
            nu = self._sample_nu_from_alignment(idx)
        elif torch.is_tensor(nu):
            nu = nu.to(self.device)
        else:
            raise NotImplementedError

        
        if self.curve_type == "quarternion_cartesian":
            rot_quaternion = self._rot(nu,idx) # [f,4]
            rot_quaternion = rot_quaternion / rot_quaternion.norm(dim=1, keepdim=True) # [f,4]
            c2w_rotations = roma.unitquat_to_rotmat(rot_quaternion) # [f,3,3]
            c2w_translations = self._trans(nu,idx) # [f,3]

        elif self.curve_type == "se3":
            se3 = torch.cat([self._trans(nu,idx), self._rot(nu, idx)], dim=1) # [f,6]
            c2w = torch3d.se3_exp_map(se3) # [f,4,4]
            c2w_rotations = c2w[:,:3,:3].transpose(-2,-1)
            c2w_translations = c2w[:,3,:3]

        else:
            raise NotImplementedError
        return c2w_rotations, c2w_translations
        
    def _c2w_to_minicam(self, rots, transes, ref_cam:Camera):
        """
        given batch of rotation and translation in c2w poses,
        returns minicam object.

        ARGUMENTS
        ---------
        rots: [b,3,3]
        transes: [b,3]
        ref_cam: Camera or Minicam object. 
                 Additional attributes (znear, zfar, fov, etc...) will be duplicated from here.
        RETURNS
        -------
        list of minicam objects
        """

        minicam_list = []
        for i, (rot,trans) in enumerate(zip(rots,transes)): # c2w
            
            world_view_transform = torch.eye(4, device=self.device)
            world_view_transform[:3,:3] = rot # NOTE rot.T.T 
            world_view_transform[3,:3] = -trans@rot # NOTE: not [:3,3] for world-view transform.
            
            projection_matrix = ref_cam.projection_matrix
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            minicam_list.append(
                MiniCam(width=ref_cam.image_width,
                        height=ref_cam.image_height,
                        fovy=ref_cam.FoVy,
                        fovx=ref_cam.FoVx,
                        znear=ref_cam.znear,
                        zfar=ref_cam.zfar,
                        world_view_transform=world_view_transform,
                        full_proj_transform=full_proj_transform,)
            )
        
        return minicam_list
    
    def get_gt_image(self, idx):
        return self.original_cam[idx].original_image
        
    def get_depth(self, idx):
        return self.original_cam[idx].original_depth
    
    def __len__(self):
        return len(self._rot)

    @property
    def device(self):
        return self._rot.device
    
    def is_optimizing(self):
        return self._rot._control_points.requires_grad
    
    def alternate_optimization(self):
        """
        Stop optimizing if it was doing. Start optimizing if optimizing process was stopped.
        """
        new_state = not self.is_optimizing()
        
        print("Curve gradient:" , "[On]" if new_state else "[Off]")
        for optimizable in [self._rot, self._trans, self._nu]:
            optimizable.requires_grad_(new_state)
        
    @torch.no_grad()
    def get_middle_cams(self):
        """
        get list of "middle" from the trajectory.
        """
        cams = []
        for i in range(len(self)):
            nu = self._sample_nu_from_alignment(i)
            mid_idx = nu.shape[0]//2
            nu_mid = nu[mid_idx: mid_idx+1]
            cam = self.get_trajectory(i,nu_mid)[0]
            cams.append(cam)
        return cams
    
    
    def save(self, state_dict_path:str):
        """
        Save camera motion parameters.
        """
        
        assert(state_dict_path.endswith(".pth"))
        
        sdict = {"rot": self._rot.state_dict(),
                 "trans": self._trans.state_dict(),
                 "nu": self._nu}

        torch.save(sdict, state_dict_path)
        print("[SAVED] Camera Motion")

    def load(self, path:str):
        """
        Load camera motion parameters.
        """

        if path.endswith(".pth"):
            state_dict_path = path
        else:
            state_dict_path = os.path.join(path,"cm.pth")

        sdict = torch.load(state_dict_path)
        self._rot.load_state_dict(sdict["rot"])
        self._trans.load_state_dict(sdict["trans"])
        self._nu = sdict['nu']
        print("[LOADED] Camera Motion")


```

# prompts/papers/deblurgs/scene/pcd_init.py

``` py

import numpy as np
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.mvg_utils import  build_K, get_normalized_coords, normalized_coords_to_cam_coords, cam_to_world_coords

def random_pcd_init(cam_infos, near=0.0, far=8.0, num_pcd=100_000,bds=None):
    """
    generate pcd along to camera frustrum

    """
    all_xyz = []
    d = 50 # num of points per ray
    num_pcd_per_cam = num_pcd // (max(len(cam_infos)-5,1)) + 2
    if bds is not None:
        print(bds.mean(axis=0))  
    for i, cam_info in enumerate(cam_infos):
        rot = cam_info.R.T # NOTE: rotation is transposed for glm-library in CUDA.
        trans = cam_info.T
        
        w2c = np.eye(4)
        w2c[:3,:3] = rot
        w2c[:3,3] = trans
        c2w = np.linalg.inv(w2c)

        w = cam_info.width
        h = cam_info.height
        fx = fov2focal(cam_info.FovX, w)
        fy = fov2focal(cam_info.FovY, h)
        K = build_K(fx*0.8, fy*0.8, w/2, h/2) # spread little bit wider area than original field of view.

        stride_coeff = num_pcd_per_cam**(-1/3)
        stride_h = int(h*stride_coeff)
        stride_w = int(w*stride_coeff)
        # stride_d = int(d*stride_coeff)

        pixel_coords = np.stack( np.meshgrid(np.linspace(0,w-1,w), np.linspace(0,h-1,h), indexing="xy"),axis=-1)
        pixel_coords = pixel_coords[::stride_h, ::stride_w]
        pixel_coords = pixel_coords.reshape((-1,2))
        
        norm_coords = get_normalized_coords(pixel_coords, K)
        
        norm_coords = np.tile(norm_coords, (d*2,1))
        
        cam_near = max(near, bds[i,0] if bds is not None else 0.0)
        cam_far = min(far, bds[i,1] if bds is not None else 999999999.9)

        depth = np.random.random((norm_coords.shape[0]))*(cam_far-cam_near)+cam_near
        cam_coords = normalized_coords_to_cam_coords(norm_coords, depth)[:num_pcd_per_cam]
        
        xyz_world = cam_to_world_coords(cam_coords, c2w)
        all_xyz.append(xyz_world)

    return np.concatenate(all_xyz,axis=0)[:num_pcd]

```

# prompts/papers/deblurgs/scene/tonemapping.py

``` py

import torch
import torch.nn as nn

class ToneMapping(nn.Module):

    # TODO 
    # somehow make this code smarter (than if elses...)
    def __init__(self, tone_mapping_type:str, eps=1e-8, bound=0):
        """
        Tone Mapping (CRF).
        currently only support: x^(1/2.2)
        """
        self.tone_mapping_type = tone_mapping_type
        self.eps = eps
        self.bound = bound
        super().__init__()
    
    def forward(self, x):
        if self.tone_mapping_type == "gamma":
            return ((x-self.bound) / (1.0-2.0*self.bound)).clamp_min(self.eps)  ** (1/2.2)
        elif self.tone_mapping_type == "reverse_gamma":
            return x.clamp_min(self.eps) ** (2.2) * (1.0-2.0*self.bound) + self.bound
        elif self.tone_mapping_type in ["identity", "reverse_identity"]:
            return x
        else:
            raise NotImplementedError("Unknown tone mapping type.")
    
    def inverse(self):
        if "reverse" in self.tone_mapping_type:
            return ToneMapping(self.tone_mapping_type[:8])
        else:
            return ToneMapping("reverse_"+self.tone_mapping_type)
        

```

# prompts/papers/deblurgs/scripts/colmap_visualization.py

``` py

import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import open3d as o3d

import argparse
import math
import cv2
import shutil

sys.path.append(os.getcwd())
from utils.system_utils import do_system
from utils.mvg_utils import build_K, to_w2c

def read_intrinsic(camera_path):

    with open(camera_path, 'r') as f:
        lines = f.readlines()

    content = lines[3]
    elements = content.strip().split()
    print(*elements)

    type_cam = elements[1]
    print(f"CAM_TYPE: {type_cam}")

    if type_cam == "PINHOLE":
        fx, fy, cx, cy = map(float, elements[-4:] )
    elif type_cam == "SIMPLE_PINHOLE":
        fx, cx, cy = map(float, elements[-3:] )
        fy = fx
    else:
        raise NotImplementedError
    
    return fx, fy, cx, cy

def read_poses(images_path):
    if not os.path.exists(images_path):
        raise Exception(f"No such file : {images_path}")

    with open(images_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise Exception(f"Invalid cameras.txt file : {images_path}")

    comments = lines[:4]
    contents = lines[4:]

    data = []
    
    for i, content in enumerate(contents[::2]):
        content_items = content.split(' ')
        q_xyzw = np.array(content_items[2:5] + content_items[1:2], dtype=np.float32) # colmap uses wxyz
        t_xyz = np.array(content_items[5:8], dtype=np.float32)
        img_name = content_items[9]

        R = Rot.from_quat(q_xyzw).as_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, -1] = t_xyz

        data.append((img_name, T))
        
    data.sort(key=lambda x:int( ''.join(c for c in x[0] if c.isdigit())) )
    poses = np.stack([pose for img_name, pose in data])
    return poses

def read_point3d_txt(point3d_path):
    if not os.path.exists(point3d_path):
        raise Exception(f"No such file : {point3d_path}")

    with open(point3d_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise Exception(f"Invalid cameras.txt file : {point3d_path}")

    comments = lines[:3]
    contents = lines[3:]

    XYZs = []
    RGBs = []
    candidate_ids = {}

    for pt_idx, content in enumerate(contents):
        content_items = content.split(' ')
        pt_id = content_items[0]
        XYZ = content_items[1:4]
        RGB = content_items[4:7]
        error = content_items[7],
        candidate_id = content_items[8::2]
        XYZs.append(np.array(XYZ, dtype=np.float32).reshape(1,3))
        RGBs.append(np.array(RGB, dtype=np.float32).reshape(1, 3) / 255.0)
        candidate_ids[pt_id] = candidate_id
    XYZs = np.concatenate(XYZs, axis=0)
    RGBs = np.concatenate(RGBs, axis=0)

    return XYZs, RGBs, candidate_ids


def write_pose(poses, out_path, stride=5):
    """
    poses: [n,3,4] or [n,4,4] cam2world matrix
    stride: # of camera group per saving (just pass large number like 1000000 to make it one file)
    """
    n = poses.shape[0]
    for i in range(0,n,stride):

        poses_partial = poses[i:i+stride]
        m_cam = None

        for j,pose in enumerate(poses_partial):
            m = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
            m.transform(pose)
            if m_cam is None:
                m_cam = m
            else:
                m_cam += m

        o3d.io.write_triangle_mesh(os.path.join(out_path, f"cam_{i:03d}.ply"), m_cam)
    # Save the camera coordinate frames as meshes for visualization
    # o3d.io.write_triangle_mesh(filename, m_cam)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./davis_preprocess")
    parser.add_argument("--scale", type=float, default=0.1)
    args = parser.parse_args()


    fx, fy, cx, cy = read_intrinsic(os.path.join(args.path, "sparse_txt", "cameras.txt"))

    poses = read_poses(os.path.join(args.path, "sparse_txt", "images.txt"))

    xyz, rgb, _ = read_point3d_txt(os.path.join(args.path, "sparse_txt", "points3D.txt"))
    print("points", xyz.shape)

    # SCALE
    xyz *= args.scale
    poses[:,:3,3] = poses[:,:3,3] * args.scale
        
    # VISUALIZATION - PCD
    vis_path = os.path.join(args.path, "visualization")

    if os.path.exists(vis_path):
        shutil.rmtree(vis_path)
    os.makedirs(vis_path, exist_ok=True)
    
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    path = os.path.join(vis_path, "pcd.ply")
    o3d.io.write_point_cloud(path, pcd)

    # FOR MY VISUALIZATION - CAMS
    write_pose(np.stack([np.linalg.inv(w2c) for w2c in poses]), vis_path, stride=1)


```

# prompts/papers/deblurgs/scripts/run_colmap.py

``` py

import cv2
import argparse
import os
import sys

import numpy as np
import shutil
from pathlib import Path

sys.path.append(os.getcwd())
from utils.system_utils import do_system
from scene.dataset_readers import read_intrinsics_text

def get_parser()->argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--result_path", type=str, required=True)

    parser.add_argument("--image_path", type=str, default=None, 
                        help="Data directory where images are located.")
    parser.add_argument("--video_path", type=str, default=None, 
                        help="Video path. Ignored if --image_path is given.")
    parser.add_argument("--mask_path", type=str, default=None,
                        help="Path for mask. Optional.")
    parser.add_argument("--reverse_mask", action="store_true",
                        help="Reverse mask (1 and 0)")
    parser.add_argument("--resize_factor", type=float, default=None,
                        help="Resize factor")

    parser.add_argument("--video_frame_min", type=int, default=None,
                        help="[Optional] frame clipping min (inclusive).")
    parser.add_argument("--video_frame_max", type=int, default=None,
                        help="[Optional] frame clipping max (exclusive).")
    parser.add_argument("--video_skip", type=int, default=1,
                        help="[Optional] frame skip rate.")
    # parser.add_argument("--mask_path", type=str, default=None,
    #                     help="Optional: motion mask for colmap.")
    
    parser.add_argument("--no_colmap", action="store_true",
                        help="not running colmap")
    parser.add_argument("--keep_image_name", action="store_true",
                        help="disable re-labeling image filename (default: renaming from 0 to n-1.jpg)")

    parser.add_argument("--forceful", action="store_true",
                        help="disable to ask whether the system want to overwrite result directory.")
    # colmap.
    parser.add_argument("--colmap_matcher", default="exhaustive", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], help="select which matcher colmap should use. sequential for videos, exhaustive for adhoc images")
    parser.add_argument("--intrinsic_path", default=None, type=str, help="Reference '[COLMAP workspace]/cameras.txt' for intrinsic information. Prior 1: providing this will ignore all intrinsic arguments below.")
    parser.add_argument("--focal_length", nargs="+", default=None, type=float, help="[Optional] provide focal length will fix it and no longer optimize.")
    parser.add_argument("--principal_points", nargs="+", default=None, type=float, help="[Optional] provide principal points will fix it and no longer optimize.")
    
    parser.add_argument("--radial", nargs="+", type=float, default=None, help="[Optional] providing radial parameter (k1, k2, p1, p2) will fix it and no longer optimize.")

    parser.add_argument("--camera_model", default="OPENCV", help="COLMAP camera model.")

    return parser

def maybe_resize(img, args):
    if args.resize_factor is not None:
        w = int(img.shape[1] / args.resize_factor)
        h = int(img.shape[0] / args.resize_factor)
        
        img = cv2.resize(img, (w,h))
    return img

def get_images(args):
    """
    Get images from path.
    """

    imgs = []
    if args.image_path is not None:
        print("Loading images from", args.image_path)
        for i, filename in enumerate(sorted([e for e in os.listdir(args.image_path) if ".jpg" in e or ".png" in e])):
            full_path = os.path.join(args.image_path, filename)
            print(f"{i:03d}")
            print(f"processing: {full_path}")

            img = cv2.imread(full_path)
            img = maybe_resize(img, args)
            imgs.append(img)

        imgs = np.stack(imgs)
        
    elif args.video_path is not None:
        
        assert not args.keep_image_name

        print("Loading video from", args.video_path)
        
        vidcap = cv2.VideoCapture(args.video_path)
        
        success = 1
        while success:
            success, img = vidcap.read()
            if success: 
                img = maybe_resize(img, args)
                imgs.append(img)
        imgs = np.stack(imgs)
        
    else:
        raise Exception("At least one of --image_path or --video_path required.")
    clipping_min = args.video_frame_min if args.video_frame_min is not None else 0
    clipping_max = args.video_frame_max if args.video_frame_max is not None else len(imgs)
    
    imgs = imgs[clipping_min:clipping_max:args.video_skip]

    return imgs

def write_images(args, imgs, folder='images', ext="png"):
    
    image_write_path = os.path.join(args.result_path, folder)
    shutil.rmtree(image_write_path, ignore_errors=True)
    os.makedirs(image_write_path, exist_ok=True)
    
    original_filenames = os.listdir(args.image_path)
    original_filenames.sort()

    for i, img in enumerate(imgs):
        filename = original_filenames[i] if args.keep_image_name else f"{i:05d}.{ext}"
        full_path = os.path.join(image_write_path, filename)
        
        print(f"{i:03d}")
        print(f"writing: {full_path}")

        cv2.imwrite(full_path, img)

def read_sparse_txt(path):
    with open(path, "r") as fp:
        while True:
            line = fp.readline()
            if not line:
                break
            line = line.strip()
            if line[0] == "#":
                continue

            tokens = line.split()

            params = tokens[4:]


def get_camera_param_format(args, imgs):
    
    if args.intrinsic_path is not None:
    
        raise NotImplementedError
        return
    assert args.focal_length is not None
    
    if len( args.focal_length ) == 1:
        fx = fy = args.focal_length[0]
    elif len( args.focal_length) == 2:
        fx, fy = args.focal_length

    if args.principal_points is None:
        h, w = imgs.shape[1:3]
        cx = w/2
        cy = h/2
    else:
        assert len(args.principal_points) == 2
        cx, cy = args.principal_points
    
    if args.radial is not None:
        k1, k2, p1, p2 = args.radial
    else:
        k1, k2, p1, p2 = 0.0, 0.0, 0.0, 0.0

    
    # NOTE
    # Colmap parameter: 
    # SIMPLE_PINHOLE: f cx cy
    # PINHOLE: fx fy cx cy
    # SIMPLE_RADIAL: f cx cy k1
    # RADIAL: f cx cy k1 k2
    # OPENCV: fx fy cx cy k1 k2 p1 p2

    if args.camera_model == "SIMPLE_PINHOLE":
        return f"{fx:.10f},{cx:.10f},{cy:.10f} "
    
    elif args.camera_model == "PINHOLE":
        return f"{fx:.10f},{fy:.10f},{cx:.10f},{cy:.10f} "
    
    elif args.camera_model == "SIMPLE_RADIAL":
        return f"{fx:.10f},{cx:.10f},{cy:.10f},{k1:.10f} "
    
    elif args.camera_model == "RADIAL":
        return f"{fx:.10f},{cx:.10f},{cy:.10f},{k1:.10f},{k2:.10f} "
    
    elif args.camera_model == "OPENCV":
        return f"{fx:.10f},{fy:.10f},{cx:.10f},{cy:.10f},{k1:.10f},{k2:.10f},{p1:.10f},{p2:.10f} "
    


def run_colmap(args, imgs):
    db = os.path.join(args.result_path, "database.db")
    images = os.path.join(args.result_path, "images")
    text = os.path.join(args.result_path, "sparse_txt")
    text_distortion = os.path.join(args.result_path, "sparse_distortion_txt")
    
    mask = os.path.join(args.result_path, "colmap_masks")
    undistortion_tmpdir = os.path.join(args.result_path, "dense")
    cam_model = args.camera_model

    flag_EAS = 1
    is_refining_focal = int(args.focal_length is None)
    is_refining_extra_params = int('PINHOLE' not in cam_model and args.radial is None) 
    is_refining_principal = 0 #int('PINHOLE' not in cam_model and args.focal_length is None)
    
    sparse = os.path.join(args.result_path, "sparse")

    print(f"running colmap with:\n\tdb={db}\n\timages={images}\n\tsparse={sparse}\n\ttext={text}")
    print(f"warning! folders '{sparse}' and '{text}' will be deleted/replaced. continue? (Y/n) Y")
    # if (input(f"warning! folders '{sparse}' and '{text}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
    #     sys.exit(1)
    if os.path.exists(db):
        os.remove(db)

    if os.path.exists(mask):
        fextract_additional_command = f" --ImageReader.mask_path {mask}"
    else:
        fextract_additional_command = ""

    if args.focal_length is not None:    
        camera_param_format = get_camera_param_format(args, imgs)
        fextract_additional_command += f" --ImageReader.camera_params {camera_param_format}"
    
    fextract_additional_command += " --SiftExtraction.use_gpu 0 "
    do_system( f"colmap feature_extractor "
               f"--ImageReader.camera_model {cam_model} "
               f"--SiftExtraction.estimate_affine_shape {flag_EAS} "
               f"--SiftExtraction.domain_size_pooling {flag_EAS} "
               f"--ImageReader.single_camera 1 "
               f"--database_path {db} "
               f"--image_path {images} "
                "--SiftExtraction.max_num_features 8192 "
               f"{fextract_additional_command}")
    
    do_system(f"colmap {args.colmap_matcher}_matcher --SiftMatching.guided_matching {flag_EAS} --database_path {db} --SiftMatching.use_gpu 0")
    
    shutil.rmtree(sparse, ignore_errors=True)

    do_system(f"mkdir {sparse}")
    do_system(f"colmap mapper --database_path {db} --image_path {images} --output_path {sparse} "
                "--Mapper.abs_pose_max_error 20 " # 12
                "--Mapper.init_max_error 12 " # 4
                "--Mapper.filter_max_reproj_error 8 " # 4
                "--Mapper.init_max_reg_trials 5 "
                "--Mapper.max_reg_trials 5 "
                "--Mapper.min_num_matches 5 "
                "--Mapper.init_min_num_inliers 30 " # 100
                "--Mapper.abs_pose_min_num_inliers 15 " # 30
                "--Mapper.abs_pose_min_inlier_ratio 0.12 " # 0.25
                "--Mapper.tri_ignore_two_view_tracks 1 "
                "--Mapper.ba_local_max_num_iterations 100 "
                "--Mapper.ba_global_max_num_iterations 100 "
               f"--Mapper.ba_refine_focal_length {is_refining_focal} "
               f"--Mapper.ba_refine_principal_point {is_refining_principal} "
               f"--Mapper.ba_refine_extra_params {is_refining_extra_params} ")
    
    do_system(f"colmap bundle_adjuster --input_path {sparse}/0 --output_path {sparse}/0 "
              f"--BundleAdjustment.refine_principal_point {is_refining_principal} "
              f"--BundleAdjustment.refine_extra_params {is_refining_extra_params} "
              f"--BundleAdjustment.refine_focal_length {is_refining_focal}")
    try:
        shutil.rmtree(text)
    except:
        pass


    # Undistortion if needed
    if "PINHOLE" not in cam_model:
        # Save Distortion Parameters.
        do_system(f"mkdir {text_distortion}")
        do_system(f"colmap model_converter --input_path {sparse}/0 --output_path {text_distortion} --output_type TXT")
        
        os.makedirs(undistortion_tmpdir)

        do_system(f"colmap image_undistorter --image_path {images} --input_path {sparse}/0 --output_path {undistortion_tmpdir}")

        # Remove distorted images.
        do_system(f"rm -rf {images}")
        do_system(f"rm -rf {sparse}")
        
        os.makedirs(sparse)
        
        do_system(f"mv {undistortion_tmpdir}/images {args.result_path}")
        do_system(f"mv {undistortion_tmpdir}/sparse {sparse}")
        do_system(f"mv {sparse}/sparse {sparse}/0")        
        
        do_system(f"rm -rf {undistortion_tmpdir}")
        

    do_system(f"mkdir {text}")
    do_system(f"colmap model_converter --input_path {sparse}/0 --output_path {text} --output_type TXT")
def move_mask(args, imgs):
    """
    Move mask images from path to result.
    """

    mask_write_path = os.path.join(args.result_path, "masks")
    os.makedirs(mask_write_path, exist_ok=True)
    
    if not args.no_colmap:
        colmap_mask_path = os.path.join(args.result_path, "colmap_masks")
        os.makedirs(colmap_mask_path, exist_ok=True)
        
    if args.mask_path is not None:
        print("Loading masks from", args.mask_path)
        
        for i, filename in enumerate(sorted(os.listdir(args.mask_path))):
            full_path = os.path.join(args.mask_path, filename)
            print(f"{i:03d}")
            print(f"processing: {full_path}")

            mask = cv2.imread(full_path)
            # mask = maybe_resize(mask, args)
        
            full_write_path = os.path.join(mask_write_path, filename)        
            print(f"writing: {full_write_path}")

            cv2.imwrite(full_write_path, mask)
            if not args.no_colmap:
                mask = (np.sum(mask,axis=-1)==0).astype(float)
                mask = cv2.resize(mask,(imgs.shape[2],imgs.shape[1]) )
                
                mask = (mask!=0).astype(int)*255
                h,w = mask.shape
                # mask = np.broadcast_to(mask.reshape((h,w,1)),((h,w,3)))

                filename_jpg = f"{i:05d}.jpg"
                full_colmap_mask_path = os.path.join(colmap_mask_path, f"{filename_jpg}.png")
                if args.reverse_mask:
                    mask = 255 - mask

                print(f"writing: {full_colmap_mask_path}")
                
                cv2.imwrite(full_colmap_mask_path, mask )


def warn_and_overwrite(path, forceful=False):
    
    if not os.path.exists(path):
        return
    if forceful:
        print(f"Forceful mode. Overwriting {path} ...")
    else:
        image_path = os.path.join(path, "images")
        if os.path.exists(image_path):
            print("Found previous COLMAP workspace. Overwriting...")
        else:
            user_answer = input(f"{path} does not look like COLMAP workspace. Do you want to Overwrite (y/N)? ").strip().lower()
            
            if user_answer != "y":
                print(f"Not overwriting {path}. Halting...")
                exit(0)

    shutil.rmtree(path)
    os.makedirs(path)

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    warn_and_overwrite(args.result_path, forceful=args.forceful)
    imgs = get_images(args)

    write_images(args, imgs)
    
    if args.mask_path is not None:
        move_mask(args, imgs)
    
    if not args.no_colmap:
        run_colmap(args, imgs)
        do_system(f"python scripts/colmap_visualization.py --path {args.result_path}")


```

# prompts/papers/deblurgs/scripts/triangulation.py

``` py

import os
import sys
sys.path.append(os.getcwd())
from utils.system_utils import do_system

import numpy as np
from scene.cameras import Camera, MiniCam
from typing import Iterable, Union
import torchvision.utils
import argparse
from arguments import ModelParams
from scene.gaussian_model import GaussianModel
from scene import Scene
import shutil
from utils.camera_utils import fov2focal
import sqlite3
from scene.cameras import get_c2w
from scene.colmap_loader import rotmat2qvec, qvec2rotmat

def read_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT * FROM images")
    images_tuples = c.fetchall()

    c.execute("SELECT * FROM cameras")
    cameras_tuples = c.fetchall()

    return cameras_tuples, images_tuples


def triangulate(cams:Iterable[Camera], output_path:str):
    """
    Using information in cams, run colmap triangulation and create a new colmap workspace.
    i.e. fix camera intrinsic&extrinsic and add point cloud.
    Useful for converting blender (and possibly llff) to colmap.
    
    ARGUMENTS
    ---------
    cams: list of Camera
      - note that it should be Camera, not Minicam object, as this script uses ground-truth image information to begin with.
    output_path
      - path for new workspace of COLMAP.
    """    

    # Workspace.
    image_path = os.path.join(output_path, "images")
    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(image_path)

    # Save Images.
    for cam in cams:
        print(cam.image_name)
        image_path_file = os.path.join(image_path, f"{cam.image_name}.png")
        torchvision.utils.save_image(cam.original_image, image_path_file)

    # Configuration.
    db_path = os.path.join(output_path, "database.db")
    sparse_path = os.path.join(output_path, "sparse_txt_tmp")
    shutil.rmtree(sparse_path, ignore_errors=True)
    os.makedirs(sparse_path)
    
    fx = fov2focal(cams[0].FoVx, cams[0].image_width)
    fy = fov2focal(cams[0].FoVy, cams[0].image_height)
    cx, cy = cams[0].image_width/2.0 , cams[0].image_height/2.0
    flag_EAS = 1

    # Feature Extract & Matching.
    do_system("colmap feature_extractor "
              f"--database_path {db_path} " 
              f"--image_path {image_path} "
              f"--SiftExtraction.estimate_affine_shape {flag_EAS} "
              f"--SiftExtraction.domain_size_pooling {flag_EAS} "
              f"--ImageReader.single_camera 1 "
              f"--ImageReader.camera_model PINHOLE "
              f"--SiftExtraction.use_gpu 0 "
              f'''--ImageReader.camera_params "{fx},{fy},{cx},{cy}" ''')
    
    do_system(f"colmap exhaustive_matcher "
              f"--database_path {db_path} "
              f"--SiftMatching.guided_matching {flag_EAS} "
              f"--SiftMatching.use_gpu 0 ")  

    # Save intrinsic in COLMAP convention.
    with open(os.path.join(sparse_path, "cameras.txt"), "w") as fp:
        print("# \n"*3, end='', file=fp)
        cam = cams[0]
        w = cam.image_width
        h = cam.image_height

        # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        print(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}", file=fp)        

    # Create Empty pointcloud file.
    with open(os.path.join(sparse_path, "points3D.txt"), "w") as fp:
        pass

    # Save Extrinsic.
    with open(os.path.join(sparse_path, "images.txt"), "w") as fp:
        print("# \n"*4, end='', file=fp)
        extr_dic = {}
        for i, cam in enumerate(cams):            
            # Render and Save.
            render_filename = f"{cam.image_name}.png"

            # Save pose in COLMAP convention.
            c2w = get_c2w(cam)
            w2c = np.linalg.inv(c2w)
            qvec = rotmat2qvec(w2c[:3,:3])
            tvec = w2c[:3,3]
            
            extr_dic[render_filename] = (qvec,tvec)
        _, image_tuples = read_db(db_path=db_path)

        # Follow Database order.
        for i, image_tuple in enumerate(image_tuples):
            render_filename = image_tuple[1]
            qvec, tvec = extr_dic[render_filename]
            # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            print(i+1, *qvec, *tvec, 1, render_filename, end="\n\n", file=fp)

    # Triangulation. (get PCD)
    sparse_result_path = os.path.join(output_path, "sparse", "0")
    shutil.rmtree(sparse_result_path, ignore_errors=True)
    os.makedirs(sparse_result_path)

    do_system(f"colmap point_triangulator "
              f"--database_path {db_path} "
              f"--image_path {image_path} "
              f"--input_path {sparse_path} "
              f"--output_path {sparse_result_path}")
    
    # Remove pointcloud-less sparse path.
    shutil.rmtree(sparse_path)
    sparse_path = sparse_result_path


    sparse_txt_path = os.path.join(output_path,"sparse_txt")
    shutil.rmtree(sparse_txt_path, ignore_errors=True)
    os.makedirs(sparse_txt_path)

    do_system(f"colmap model_converter "
              f"--input_path {sparse_path} "
              f"--output_path {sparse_txt_path} "
              f"--output_type TXT")
    
    do_system(f"python scripts/colmap_visualization.py --path {output_path} ")
      
    print("[DONE]")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Triangulation script parameters")
    model_params = ModelParams(parser, sentinel=False)
    parser.add_argument("--result_path", type=str, required=True, help="new colmap directory.")

    args = parser.parse_args()

    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, shuffle=False)
    triangulate(scene.getTrainCameras(),output_path=args.result_path)


```

# prompts/papers/deblurgs/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py

``` py
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    viewmatrix,
    projmatrix,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        viewmatrix,
        projmatrix,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        viewmatrix,
        projmatrix,
        raster_settings,
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            viewmatrix,
            projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.z_near,
            raster_settings.z_far,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.use_sigmoid,
            raster_settings.debug
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, depth, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, viewmatrix, projmatrix)
        
        return color, depth, radii

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_depth, _):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, viewmatrix, projmatrix = ctx.saved_tensors
        
        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                viewmatrix, 
                projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                raster_settings.z_near,
                raster_settings.z_far,
                grad_out_color, 
                grad_out_depth,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.use_sigmoid,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_viewmatrix, grad_projmatrix = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_viewmatrix, grad_projmatrix = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            grad_viewmatrix, 
            grad_projmatrix,
            None,
        )
        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    z_near : float
    z_far : float
    use_sigmoid:bool
    # viewmatrix : torch.Tensor
    # projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None,
                viewmatrix=None, projmatrix=None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            viewmatrix,
            projmatrix,
            raster_settings,
        )



```

# prompts/papers/deblurgs/submodules/diff-gaussian-rasterization/setup.py

``` py
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)


```

# prompts/papers/deblurgs/submodules/simple-knn/setup.py

``` py
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cxx_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

setup(
    name="simple_knn",
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
            "spatial.cu", 
            "simple_knn.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": [], "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)


```

# prompts/papers/deblurgs/test.py

``` py

import os
import shutil

from scene import Scene, GaussianModel
from scene.cameras import Camera, MiniCam
from scene.cameras import get_c2w
from utils.graphics_utils import getProjectionMatrix
from gaussian_renderer import render
from arguments import ModelParams, get_combined_args

from utils.image_utils import psnr
from utils.loss_utils import ssim, l1_loss
from lpipsPyTorch import lpips
from scene.colmap_loader import rotmat2qvec, qvec2rotmat
from utils.system_utils import do_system
from utils.graphics_utils import fov2focal
from utils.graphics_utils import getWorld2View, getProjectionMatrix

import torch
import torch.nn as nn
import torch.optim as optim
import roma

from tqdm import tqdm
import copy
import argparse

import random
import math
from PIL import Image
import utils.general_utils 
import utils.colorize
import torchvision.utils
import numpy as np
from scripts.colmap_visualization import read_poses
from utils.colmap_reoder import read_db

class OptimPoseModel(nn.Module):

    def __init__(self, cams:list):
        
        """
        
        Aliases:
        C : curve order
        n : # imgs.
        f : # subframes.
        """
        super().__init__()

        print("Optim Pose Model...")
        
        rots = []
        transes = []
        
        self.cams = cams
        for cam in cams:
            cam:Camera    
            rots.append(torch.from_numpy(cam.R).cuda())
            transes.append(torch.from_numpy(cam.T).cuda())

        rots = torch.stack(rots)
        transes = torch.stack(transes)

        rots_unitquat = roma.rotmat_to_unitquat(rots)
        
        self._rot = nn.Parameter(rots_unitquat.float().clone().contiguous().requires_grad_(True)) # [n,4]
        self._trans = nn.Parameter(transes.float().clone().contiguous().requires_grad_(True)) # [n,3]

        
    def forward(self,idx):
        cam: Camera
        cam = copy.deepcopy(self.cams[idx])
        
        quat = self._rot[idx] + 1e-8# [4]
        
        unitquat = quat / quat.norm() # [4]
        rotmat = roma.unitquat_to_rotmat(unitquat[None,:]).squeeze() # [3,3]
        trans = self._trans[idx] # [3]

        cam.world_view_transform = torch.eye(4).cuda()
        cam.world_view_transform[:3,:3] = rotmat.T
        cam.world_view_transform[:3, 3] = trans
        cam.world_view_transform = cam.world_view_transform.transpose(0,1)

        cam.projection_matrix = getProjectionMatrix(znear=cam.znear, zfar=cam.zfar, fovX=cam.FoVx, fovY=cam.FoVy).transpose(0,1).cuda()
        cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
        cam.camera_center = cam.world_view_transform.inverse()[3, :3]

        return cam

@torch.no_grad()
def evaluate(cams:list, scene: Scene, gaussians:GaussianModel,bg_color:torch.Tensor, vis_dir:str=None):
    """
    Evaluation using test cams.

    RETURNS
    -------
    psnr, ssim, lpips: (float each) metric of current settings.

    """

    if vis_dir is not None:
        vis_path = os.path.join(scene.model_path, vis_dir)
        shutil.rmtree(vis_path, ignore_errors=True)
        os.makedirs(vis_path)
    else:
        vis_path = None
    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    n = len(cams)
    for i, cam in enumerate(cams):
        
        gt_image = cam.original_image
        image = scene.tone_mapping(render(cam, gaussians, bg_color)["render"] )
        psnr_test += psnr(image, gt_image).mean().item()
        ssim_test += ssim(image, gt_image).mean().item()
        lpips_test += lpips(image, gt_image, net_type='alex').mean().item()

        if vis_path is not None:
            errormap = utils.colorize.colorize(torch.abs(gt_image - image).permute(1,2,0).mean(dim=-1)).permute(2,0,1)
            torchvision.utils.save_image(gt_image, os.path.join(vis_path, f"{i:03d}_gt.png"))
            torchvision.utils.save_image(image, os.path.join(vis_path, f"{i:03d}_render.png"))
            torchvision.utils.save_image(errormap, os.path.join(vis_path, f"{i:03d}_error.png"))
                    
    
    return psnr_test/n, ssim_test/n, lpips_test/n

def optimize_test_pose(scene: Scene, gaussians:GaussianModel, bg_color:torch.Tensor, num_iter_per_view:int=2000):
    """
    Run iNeRF-like pose optimization for test veiws.
    Note that test camera pose is not accurate for curve-optimized 3DGS scene, so this process is essential. 
    
    RETURNS
    -------
    optimized_cams: list of Camera object, fit to current scene.
    """
    torch.cuda.empty_cache()

    test_cameras = scene.getTestCameras()
    n = len(test_cameras)

    optim_model = OptimPoseModel(test_cameras)
    optim_param_group = [{"params":[optim_model._rot],   'lr': 5e-5, 'name':"rot"},
                         {"params":[optim_model._trans], 'lr': 5e-4, 'name':"trans"}]

    optimizer = optim.Adam(optim_param_group, lr=5e-4, eps=1e-15)

    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=num_iter_per_view//20,gamma=0.9)
    
    pbar = tqdm(range(num_iter_per_view), desc="Optimizing...")

    l2_error_ema = 0.0
    
    for iteration in pbar:
        
        idx_list = list(range(n))
        random.shuffle(idx_list)
        
        # Run 1 Epoch.
        while len(idx_list) > 0:        
            # Choose one test view.
            idx = idx_list.pop()
            viewpoint_cam = optim_model(idx)
            
            # Loss.
            gt_image = viewpoint_cam.original_image
            image = render(viewpoint_cam, gaussians, bg_color)["render"]
            image = scene.tone_mapping(image).clamp(0.0,1.0)
            loss = l1_loss(image, gt_image)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                l2_error_ema = l2_error_ema * 0.6 + ((gt_image-image)**2).mean().item() * 0.4
    
        lr_scheduler.step()
        if iteration % 20 == 0:
            with torch.no_grad():
                current_psnr = 20 * math.log10(1.0 / math.sqrt(l2_error_ema))
                pbar.set_description(f"Optimizing...PSNR ={current_psnr:6.2f} lr = {optimizer.param_groups[0]['lr']:10.6f}")
                
    return [optim_model(i) for i in range(n)]

@torch.no_grad()
def initialize_test_pose(args:ModelParams, scene: Scene, gaussians:GaussianModel, bg_color:torch.Tensor,exclude = [], old_version=False):
    """
    Only functions when testing without known pose.
    (i.e.) not llffhold-style.
    """
    source_path = args.source_path
    model_path = args.model_path
    if len(scene.getTestCameras()) > 0:
        return
    
    print("Not LLFFHOLD style dataset... Looking for test image without poses.")
    test_image_dir = os.path.join(source_path, "test_images")
    if not os.path.exists(test_image_dir):
        print("No test image detected... Exiting")
        exit()

    # Prepare temporary colmap workspace.
    tmp_colmap_workspace = os.path.join(model_path, "render_colmap")
    shutil.rmtree(tmp_colmap_workspace, ignore_errors=True)
    os.makedirs(tmp_colmap_workspace)

    db_path = os.path.join(tmp_colmap_workspace, "database.db")

    tmp_images = os.path.join(tmp_colmap_workspace, "images_rendered")
    os.makedirs(tmp_images)

    tmp_sparse = os.path.join(tmp_colmap_workspace, "sparse", "1")
    os.makedirs(tmp_sparse)

    flag_EAS = 1

    # Load cams.
    scene.camera_motion_module.load(os.path.join(model_path, "cm.pth"))
    cams = [cam for i,cam in enumerate(scene.camera_motion_module.get_middle_cams()) if i not in exclude]
    
    # Render from train view, save.
    print("Rendering from training view...")
    for i, cam in enumerate(cams):
        cam: MiniCam
        
        # Render and Save.
        render_filename = f"{i:03d}_render.png"
        rendered = render(cam, gaussians, bg_color)["render"]
        rendered = scene.tone_mapping(rendered)
        torchvision.utils.save_image(rendered, os.path.join(tmp_images, render_filename))

    # Save extrinsic => we will do later to keep track with database order.
    # Save intrinsic in COLMAP convention.
    with open(os.path.join(tmp_sparse, "cameras.txt"), "w") as fp:
        print("# \n"*3, end='', file=fp)
        cam = cams[0]
        w = cam.image_width
        h = cam.image_height
        fx = fov2focal(cam.FoVx, w)
        fy = fov2focal(cam.FoVy, h)
        cx = w/2
        cy = h/2

        # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        print(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}", file=fp)        

    # Create Empty pointcloud file.
    with open(os.path.join(tmp_sparse, "points3D.txt"), "w") as fp:
        pass

    # Run colmap with rendered images only.
    do_system("colmap feature_extractor "
              f"--database_path {db_path} " 
              f"--image_path {tmp_images} "
              f"--SiftExtraction.estimate_affine_shape {flag_EAS} "
              f"--SiftExtraction.domain_size_pooling {flag_EAS} "
              f"--ImageReader.single_camera 1 "
              f"--ImageReader.camera_model PINHOLE "
              f"--SiftExtraction.use_gpu 0 "
              f'''--ImageReader.camera_params "{fx},{fy},{cx},{cy}" ''')
    
    do_system(f"colmap exhaustive_matcher "
              f"--database_path {db_path} "
              f"--SiftMatching.guided_matching {flag_EAS} "
              f"--SiftMatching.use_gpu 0 ")
    
    tmp_sparse_pcd = os.path.join(tmp_colmap_workspace,"sparse","2")
    os.makedirs(tmp_sparse_pcd, exist_ok=True)

    # Save Extrinsic.
    with open(os.path.join(tmp_sparse, "images.txt"), "w") as fp:
        print("# \n"*4, end='', file=fp)
        extr_dic = {}
        for i, cam in enumerate(cams):
            cam: MiniCam
            
            # Render and Save.
            render_filename = f"{i:03d}_render.png"

            # Save pose in COLMAP convention.
            c2w = get_c2w(cam)
            w2c = np.linalg.inv(c2w)
            qvec = rotmat2qvec(w2c[:3,:3])
            tvec = w2c[:3,3]
            
            extr_dic[render_filename] = (qvec,tvec)
        _, image_tuples = read_db(db_path=db_path)

        # Follow Database order.
        for i, image_tuple in enumerate(image_tuples):
            render_filename = image_tuple[1]
            qvec, tvec = extr_dic[render_filename]
            # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            print(i+1, *qvec, *tvec, 1, render_filename, end="\n\n", file=fp)

    # Triangulation. (get PCD)
    inputpath_arg = "--input_path" if not old_version else "--import_path"
    outputpath_arg = "--output_path" if not old_version else "--export_path"
    cmd = f"colmap point_triangulator " + \
              f"--database_path {db_path} " + \
              f"--image_path {tmp_images} " + \
              f"{inputpath_arg} {tmp_sparse} " + \
              f"{outputpath_arg} {tmp_sparse_pcd}"
    do_system(cmd)
    # Prepare test images
    test_image_files = os.listdir(test_image_dir)
    test_image_files.sort()

    tmp_test_images = os.path.join(tmp_colmap_workspace, "test_images")
    shutil.rmtree(tmp_test_images, ignore_errors=True)
    os.makedirs(tmp_test_images)

    for i, test_image_file in enumerate(test_image_files):
        test_image_path = os.path.join(test_image_dir, test_image_file)
        img_pil = Image.open(test_image_path)
        img_pil.save(os.path.join(tmp_test_images,f"{i:03d}.png"))
        print("[DONE]", test_image_path)
    
    # feature extraction and match.
    do_system("colmap feature_extractor "
              f"--database_path {db_path} " 
              f"--image_path {tmp_test_images} "
              f"--SiftExtraction.estimate_affine_shape {flag_EAS} "
              f"--SiftExtraction.domain_size_pooling {flag_EAS} "
              f"--ImageReader.single_camera 1 "
              f"--ImageReader.camera_model PINHOLE "
              f"--SiftExtraction.use_gpu 0 "
              f'''--ImageReader.camera_params "{fx},{fy},{cx},{cy}" ''')
    
    do_system(f"colmap exhaustive_matcher "
              f"--database_path {db_path} "
              f"--SiftMatching.guided_matching {flag_EAS} "
              f"--SiftMatching.use_gpu 0 ")
    

    tmp_sparse_final = os.path.join(tmp_colmap_workspace,"sparse","0")
    shutil.rmtree(tmp_sparse_final, ignore_errors=True)
    os.makedirs(tmp_sparse_final)

    do_system(f"colmap image_registrator "
              f"--database_path {db_path} "
              f"{inputpath_arg} {tmp_sparse_pcd} "
              f"{outputpath_arg} {tmp_sparse_final}")

    tmp_sparse_txt = os.path.join(tmp_colmap_workspace,"sparse_txt")
    shutil.rmtree(tmp_sparse_txt, ignore_errors=True)
    os.makedirs(tmp_sparse_txt)

    do_system(f"colmap model_converter "
              f"--input_path {tmp_sparse_final} "
              f"--output_path {tmp_sparse_txt} "
              f"--output_type TXT")

    do_system(f"python scripts/colmap_visualization.py --path {tmp_colmap_workspace} ")
        
    # Get test images and poses.
    image_txtfile = os.path.join(tmp_sparse_txt, "images.txt")
    
    with open(image_txtfile, 'r') as f:
        lines = f.readlines()

    lines = lines[4:]
    lines = lines[::2]

    test_cams = []

    one_cam = scene.getTrainCameras()[0]
    
    for line in lines:
        tokens = line.strip().split()
        img_name = tokens[-1]
        if "render" in img_name:
            continue

        test_image_path = os.path.join(tmp_test_images, img_name)
        img_pil = Image.open(test_image_path)
        img = utils.general_utils.PILtoTorch(img_pil, img_pil.size).cuda()

        qvec = np.array(list(map(float, tokens[1:5])))
        tvec = np.array(list(map(float, tokens[5:8])))

        R = qvec2rotmat(qvec).T
        T = np.array(tvec)

        
        # world_view_transform = torch.tensor(getWorld2View(R, T)).transpose(0, 1).cuda()
        # projection_matrix = getProjectionMatrix(znear=one_cam.znear, zfar=one_cam.zfar, fovX=one_cam.FoVx, fovY=one_cam.FoVy).transpose(0,1).cuda()
        # full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        # camera_center = world_view_transform.inverse()[3, :3]
        
        test_cam = Camera(1, R, T, one_cam.FoVx, one_cam.FoVy, img, None, img_name, 1)
        # test_cam.original_image = img
        test_cams.append(test_cam)
    
    scene.test_cameras[1.0] = test_cams

    
if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Testing script parameters")
    model_params = ModelParams(parser, sentinel=False)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--exclude_indice", nargs="+", default=[] , type=int)
    parser.add_argument("--colmap_old_ver", action="store_true")
    args = get_combined_args(parser)
    print("Evaluating...")

    bg_color = torch.ones(3).cuda()
    
    scene_args = model_params.extract(args)

    # [HARDCODING] If hold exists, forcefully turn on the eval mode.
    data_path = scene_args.source_path
    if len( [e for e in os.listdir(data_path) if "hold" in e] ) == 1:
        scene_args.eval= True

    gaussians = GaussianModel(scene_args)
    scene = Scene(scene_args, gaussians, load_iteration=args.iteration, shuffle=False)
    initialize_test_pose(scene_args, scene, gaussians, bg_color, 
                         exclude=args.exclude_indice, 
                         old_version=args.colmap_old_ver)

    # Before fitting test pose.
    before_psnr, before_ssim, before_lpips = evaluate(scene.getTestCameras(), scene, gaussians, bg_color, vis_dir="eval_before")
    print(f"!!! (Unfit) PSNR: {before_psnr:.2f} SSIM: {before_ssim:.3f} LPIPS: {before_lpips:.3f}")

    fit_cams = optimize_test_pose(scene, gaussians, bg_color=torch.ones(3).cuda())

    after_psnr, after_ssim, after_lpips = evaluate(fit_cams, scene, gaussians, bg_color, vis_dir="eval_after")
    print(f"!!! (Fit) PSNR: {after_psnr:.2f} SSIM: {after_ssim:.3f} LPIPS: {after_lpips:.3f}")

    with open(os.path.join(scene_args.model_path, "eval.txt"), "w") as fp:
        print(f"PSNR: {after_psnr:.2f}", file=fp)
        print(f"SSIM: {after_ssim:.3f}", file=fp)
        print(f"LPIPS: {after_lpips:.3f}", file=fp)
        

```

# prompts/papers/deblurgs/train.py

``` py
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import numpy as np
from utils.loss_utils import l1_loss, batchwise_smoothness_loss , hinge_l2, tv_loss
import sys
from scene import Scene, GaussianModel
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, OptimizationParams
from utils.visualization import Visualizer
import utils.general_utils as general_utils
from utils.system_utils import do_system
from utils.logger import Logger
import time

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


        
def training(dataset:ModelParams, opt:OptimizationParams, args):
  
    saving_iterations = args.save_iterations
    checkpoint_iterations = args.checkpoint_iterations
    checkpoint = args.start_checkpoint
    
    render_iterations = args.render_iterations
    is_visualizing_curve = not args.disable_curve_visualize
    
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians, load_path=args.load_path)
    cam_motion_module = scene.camera_motion_module

    gaussians.training_setup(opt)
    
    # Load camera motion parameters if path is given.
    if args.load_camera_motion_path is not None: 
        cam_motion_module.load(args.load_camera_motion_path)


    # Add pose params to the optimizer.
    cam_motion_module.add_training_setup(gaussians=gaussians, lr_dict={'curve_rot':opt.curve_rotation_lr,
                                                                       'curve_trans':opt.curve_controlpoints_lr,
                                                                       'curve_alignment':opt.curve_alignment_lr})
    # Gaussian Loader.
    first_iter = 0
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    first_iter += 1
    training_time_sec = 0.0

    # Background color is for Visualizer only.
    # (We provide random background color for training as we want the influence of background to be 0.)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") 
    
    # Prepare logger, visualizer.
    scene_name = [e for e in args.source_path.split("/") if len(e.strip())>0][-1]
    progress_bar = tqdm(range(first_iter, opt.iterations+1), desc=scene_name, ncols=200)
    logger = Logger(progress_bar, ema_weight=0.6)
    visualizer = Visualizer(opt, scene, gaussians, background, vis_cam_idx=args.vis_cam_idx)
    
    # Schedulers
    densify_threshold_func = general_utils.get_expon_lr_func(opt.densify_grad_threshold_init,
                                                             opt.densify_grad_threshold_final,
                                                             max_steps=opt.densify_annealing_until)

    lambda_t_smooth_func = general_utils.get_expon_lr_func( opt.lambda_t_smooth_init,
                                                            opt.lambda_t_smooth_final,
                                                            max_steps=opt.iterations)
    noise_func = general_utils.get_expon_lr_func(opt.noise_init,
                                                 opt.noise_final,
                                                 max_steps=opt.iterations)
    
    alignment_func = general_utils.get_scheduler(lr_init=opt.curve_alignment_lr,
                                                 lr_final=1e-7,
                                                 warmup_ratio=0.0,
                                                 step_warmup=opt.curve_alignment_start,
                                                 step_final=opt.iterations)
    # alignment_func = get_expon_lr_func(opt.curve_alignment_lr,
    #                                    0.0,
    #                                    lr_delay_steps=int(opt.densify_until_iter*opt.drop_alignment),
    #                                    max_steps=opt.iterations)
    
    # alignment_func = get
    # Turn off camera motion optimizer.
    cam_motion_module.alternate_optimization() 

    for iteration in range(first_iter, opt.iterations + 1):        
        
        t0 = time.time()

        # Update scheduled hyperparameters.
        gaussians.update_learning_rate(iteration, opt,alignment_lr=alignment_func(iteration))
        densification_threshold = densify_threshold_func(iteration) if args.flag != 1 else opt.densify_grad_threshold_init
        lambda_t_smooth = lambda_t_smooth_func(iteration)
        
        # Turn on/off camera motion optimizer. 
        if iteration == opt.curve_start_iter or iteration == opt.curve_end_iter:    
            cam_motion_module.alternate_optimization()
        
        # Turn off random sampling pose on the camera motion.
        if iteration == opt.random_sample_until:    
            cam_motion_module.curve_random_sample = False 

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
         
        # Get camera idx and sub-frame indice.        
        cam_idx = scene.get_random_cam_idx()
        if iteration>=opt.curve_start_iter:
            subframe_indice = "all"
        else:
            subframe_indice = 1

        # Render
        retrieved = cam_motion_module.query(cam_idx=cam_idx, 
                                            subframe_indice=subframe_indice)
        
        blur = retrieved['blurred']
        subframes = retrieved['subframes']
        subframe_depths = retrieved['depths']
        gt = retrieved['gt']
        render_pkgs = retrieved['render_pkgs']
        
        # (Optional) Add noise to the GT if desired.
        noise = noise_func(iteration)
        gt = scene.tone_mapping.inverse()(gt) + torch.randn_like(gt)*noise
        
        # [========== Loss =========] #
        Ll1 = l1_loss(blur, gt)
        L_t_smooth = batchwise_smoothness_loss(subframes)
        
        # Depth Smoothness (Optional). Not written in the paper.
        if opt.lambda_depth_tv>0.0:
            L_depth_tv = tv_loss(subframe_depths[:,None,:,:])
        else:
            L_depth_tv = 0.0
            
        # Penalize opacity and t out-of-range (not written in the paper.)
        # (We have replaced opacity activation from sigmoid to identity.)
        if opt.lambda_hinge > 0.0:
            L_hinge = hinge_l2(gaussians._opacity) # + hinge_l2(scene.camera_motion_module._nu)
        loss = Ll1 + \
               lambda_t_smooth * L_t_smooth + \
               opt.lambda_depth_tv * L_depth_tv + \
               opt.lambda_hinge * L_hinge
            
        loss.backward()

        
        with torch.no_grad():
            # Progress bar
            logger.update( {"l1":(Ll1,"ema",".5f"),
                            "smooth":(L_t_smooth,"ema",".7f"),
                            "hinge":(L_hinge,"ema",".7f"),
                            "vel":(alignment_func(iteration),"update",".4f"),
                            "#pts":(gaussians._xyz.shape[0],"update","7d")})
            if iteration % 10 == 0:
                logger.show()

            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                scene.camera_motion_module.save(os.path.join(scene.model_path, "cm.pth"))

            # Densification
            if iteration < opt.densify_until_iter:
                # Now that we have more than 1 image in a single training iter, iterate over all viewpoint tensors.
                for render_pkg in render_pkgs:
                    viewspace_point_tensor, visibility_filter, radii = render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
                    gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, 1.0/len(render_pkgs))

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    gaussians.densify_and_prune(densification_threshold, scene.cameras_extent)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
       

            # Optimizer step
            if iteration < opt.iterations:
                if opt.clip_grad>0.0:
                    torch.nn.utils.clip_grad_value_([e['params'][0] for e in gaussians.optimizer.param_groups],opt.clip_grad)

                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            # 
            time_sec = time.time() - t0
            training_time_sec = training_time_sec + time_sec
            # Save and visualize.
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
            if iteration in render_iterations:
                visualizer.traj_render(iteration)
            
            if is_visualizing_curve:
                visualizer.run(iteration)

    if is_visualizing_curve:
        visualizer.save_video()
        
    with open(os.path.join(args.model_path,"time.txt") ,"w") as fp:
        print(f"Training Time = {training_time_sec:7.5f}sec" , file=fp)
        
    for rendercode in ["render_spiral", "render_trainview"]:
        do_system(f"python {rendercode}.py --model {args.model_path} --source {args.source_path} "
                f"--resolution {args.resolution} --tone_mapping {args.tone_mapping_type} "
                f"--sh_degree {args.sh_degree} --activation {args.activation}")


def print_args(args):
    path = os.path.join(args.model_path, "args.txt")
    with open(path, "w") as fp:
        for k,v in args.__dict__.items():
            print(k,":",v,file=fp)

def set_output_folder(args):    
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    
if __name__ == "__main__":

    torch.set_printoptions(precision=4, sci_mode=False)
    np.set_printoptions(precision=4, suppress=True)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
     
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[150_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[50_000, 100_000, 150_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--render_iterations", nargs="+", type=int, default=[25_000, 50_000, 75_000, 100_000, 125_000, 150_000])
    
    parser.add_argument("--disable_curve_visualize", action="store_true", help="Do not use visualizer.")
    parser.add_argument("--vis_cam_idx", type=int, default=None, help="visualizer will focus on [VIS_CAM_IDX]-th camera rendering, instead of overall view.")
    parser.add_argument("--load_camera_motion_path", type=str, default=None, help="Load motion parameters, either .pth file or the workspace directory.")
    parser.add_argument("--load_path", type=str, default=None, help="Load gaussian from.")
    
    parser.add_argument("--flag", type=int, default=None, help="custom flag for hard-coding experiment.")
    args = parser.parse_args(sys.argv[1:])
        
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    os.makedirs(args.model_path, exist_ok=True)

    print_args(args)
    set_output_folder(args)

    training(lp.extract(args), op.extract(args), args )

    # All done
    print("\nTraining complete.")


```

# prompts/papers/deblurgs/utils/camera_utils.py

``` py
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import cv2
import torch
WARNED = False
from scene.tonemapping import ToneMapping

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    if cam_info.depth is not None:
        resized_depth = cv2.resize(cam_info.depth, resolution) # [12/8] TODO fix here: use better resize. 
        resized_depth = torch.from_numpy(resized_depth).cuda()
    else:
        resized_depth = None    
    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, depth=resized_depth)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


```

# prompts/papers/deblurgs/utils/colmap_reoder.py

``` py
import sqlite3
import os
import argparse

def read_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT * FROM images")
    images_tuples = c.fetchall()

    c.execute("SELECT * FROM cameras")
    cameras_tuples = c.fetchall()

    return cameras_tuples, images_tuples

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str)
    parser.add_argument("--database_filename", type=str, default="database.db")
    parser.add_argument("")
    args = parser.parse_args()

    dir_db = os.path.join(args.datadir, 'db.db')
    file_images = os.path.join(args.datadir, 'sparse_learned', 'images.txt')

    # Read db file to dictionary.
    cam_dict, img_dict = read_db(dir_db)

    # Parse images.txt file line-by-line.
    image_list = []
    with open(file_images, 'r') as f:
        lines = f.readlines()
        #import pdb; pdb.set_trace()
        for line in lines:
            if line != '\n':
                image_list.append(line)
                #print(line.split(' ')[-1].replace('\n', ''))
    os.system("mv {} {}".format(file_images, file_images.replace('images.txt', 'images_sorted.txt')))
    
    with open(file_images, 'w') as f:
    #with open(os.path.join(args.datadir, 'sparse_learned', 'debug.txt'), 'w') as f:
        for data in img_dict:
            print(data[1])
            for img_data in image_list:
                img_name = img_data.split(' ')[-1].replace('\n', '')

                if img_name == data[1]:
                    idx_split = img_data.find(' ', 1)
                    img_data_new = str(data[0]) + ' ' + img_data[idx_split+1:]
                    f.write(img_data_new)
                    f.write('\n')
                    


```

# prompts/papers/deblurgs/utils/colorize.py

``` py

import torch
import numpy as np
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib as mpl
from matplotlib import cm

TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision

def get_vertical_colorbar(h, vmin, vmax, cmap_name='jet', label=None, cbar_precision=2):
    '''
    :param w: pixels
    :param h: pixels
    :param vmin: min value
    :param vmax: max value
    :param cmap_name:
    :param label
    :return:
    '''
    fig = Figure(figsize=(2, 8), dpi=100)
    fig.subplots_adjust(right=1.5)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting.
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    tick_cnt = 6
    tick_loc = np.linspace(vmin, vmax, tick_cnt)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    ticks=tick_loc,
                                    orientation='vertical')

    tick_label = [str(np.round(x, cbar_precision)) for x in tick_loc]
    if cbar_precision == 0:
        tick_label = [x[:-2] for x in tick_label]

    cb1.set_ticklabels(tick_label)

    cb1.ax.tick_params(labelsize=18, rotation=0)

    if label is not None:
        cb1.set_label(label)

    fig.tight_layout()

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    im = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    im = im[:, :, :3].astype(np.float32) / 255.
    if h != im.shape[0]:
        w = int(im.shape[1] / im.shape[0] * h)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

    return im

def colorize_np(x, cmap_name='jet', mask=None, range=None, append_cbar=False, cbar_in_image=False, cbar_precision=2):
    '''
    turn a grayscale image into a color image
    :param x: input grayscale, [H, W]
    :param cmap_name: the colorization method
    :param mask: the mask image, [H, W]
    :param range: the range for scaling, automatic if None, [min, max]
    :param append_cbar: if append the color bar
    :param cbar_in_image: put the color bar inside the image to keep the output image the same size as the input image
    :return: colorized image, [H, W]
    '''
    if range is not None:
        vmin, vmax = range
    elif mask is not None:
        # vmin, vmax = np.percentile(x[mask], (2, 100))
        vmin = np.min(x[mask][np.nonzero(x[mask])])
        vmax = np.max(x[mask])
        # vmin = vmin - np.abs(vmin) * 0.01
        x[np.logical_not(mask)] = vmin
        # print(vmin, vmax)
    else:
        vmin, vmax = np.percentile(x, (1, 100))
        vmax += TINY_NUMBER

    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin)
    # x = np.clip(x, 0., 1.)

    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]

    if mask is not None:
        mask = np.float32(mask[:, :, np.newaxis])
        x_new = x_new * mask + np.ones_like(x_new) * (1. - mask)

    cbar = get_vertical_colorbar(h=x.shape[0], vmin=vmin, vmax=vmax, cmap_name=cmap_name, cbar_precision=cbar_precision)

    if append_cbar:
        if cbar_in_image:
            x_new[:, -cbar.shape[1]:, :] = cbar
        else:
            x_new = np.concatenate((x_new, np.zeros_like(x_new[:, :5, :]), cbar), axis=1)
        return x_new
    else:
        return x_new


# tensor
@torch.no_grad()
def colorize(x, cmap_name='jet', mask=None, range=None, append_cbar=False, cbar_in_image=False):
    device = x.device
    x = x.cpu().numpy()
    if mask is not None:
        mask = mask.cpu().numpy() > 0.99
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    x = colorize_np(x, cmap_name, mask, range, append_cbar, cbar_in_image)
    x = torch.from_numpy(x).to(device)
    return x



```

# prompts/papers/deblurgs/utils/export_utils.py

``` py

import os
import sys

import math
import imageio

import numpy as np
import torch

sys.path.append(os.getcwd())

from gaussian_renderer import render

from scene import Scene
from scene.cameras import get_c2w as cam_to_numpy_c2w
from scene.cameras import c2w_to_cam as numpy_c2w_to_cam

import matplotlib
from utils.mvg_utils import mean_camera_pose, get_c2w_from_eye

from scene.cameras import Camera, MiniCam


def depth_colorize_with_mask(depthlist, background=(0,0,0), dmindmax=None):
    """ depth: (H,W) - [0 ~ 1] / mask: (H,W) - [0 or 1]  -> colorized depth (H,W,3) [0 ~ 1] """
    print("Depth colorizing...", end="")
    batch, vx, vy = np.where(depthlist!=0)
    if dmindmax is None:
        valid_depth = depthlist[batch, vx, vy]
        dmin, dmax = valid_depth.min(), valid_depth.max()
    else:
        dmin, dmax = dmindmax
    norm_dth = np.ones_like(depthlist)*dmax # [B, H, W]
    norm_dth[batch, vx, vy] = (depthlist[batch, vx, vy]-dmin)/(dmax-dmin)
    
    final_depth = np.ones(depthlist.shape + (3,)) * np.array(background).reshape(1,1,1,3) # [B, H, W, 3]
    cmapper = matplotlib.cm.get_cmap('jet_r')
    final_depth[batch, vx, vy] = cmapper(norm_dth)[batch,vx,vy,:3]
    print(" [DONE]")

    return final_depth

@torch.no_grad()
def depth_colorize(depths:torch.Tensor, z_near:float=0.01, z_far:float=100, clip_percentage:float=1.000):
    """
    ARGUMENTS
    ---------
    depths: tensor [b,h,w]
    z_near, z_far: decides "absolute scale"
    
    RETURNS
    -------
    final_depth: uint8 ndarray [b,h,w,3]
    """
    z_near = max(z_near, depths.min().item())
    z_far = min(z_far, depths.max().item() , depths.reshape((-1,)).sort().values[int((depths.numel()-1)*clip_percentage)].item())
    depths = ( depths - z_near ) / (z_far - z_near)
    depths = depths.clip(0.0, 1.0)

    depths_npy = depths.cpu().numpy()
    cmapper = matplotlib.cm.get_cmap('jet_r')
    final_depth = cmapper(depths_npy)

    return (final_depth * 255).astype(np.uint8)

def depths_to_ndc_z(depths:torch.Tensor, z_near:float, z_far:float):
    """
    ARGUMENTS
    ---------
    depths: tensor [b,h,w]
    z_near, z_far: decides "absolute scale"
    
    RETURNS
    -------
    ndc_depth: tensor [b,h,w]
        ndc_depth. 
    """
    z_near = max(z_near, depths.min().item())
    z_far = min(z_far, depths.max().item())
    depths = depths.clamp(z_near, z_far)

    return ( ( z_far * depths - z_far*z_near ) / (z_far-z_near) ) / depths


@torch.no_grad()
def get_render_path(scene:Scene, spin_angle=5.0, n_frames=50 , spin_for=2):

    """
    Let view-vector to be avg. of cameras->lookat vector,
    get spiral camera path around the view-vector.

    Arguments
    ---------
    scene: 3DGS Scene object.
    spin_angle: the angle between view-vector and cameras.
    n_frames: the number of cameras.
    spin_for: the number of spinning.
    """

    # Deg->Radian
    spin_angle = spin_angle*np.pi / 180.0

    # Camera Objects.
    cameras = scene.camera_motion_module.get_middle_cams()
    
    # Reference for metadata copy.
    ref_camera:Camera = cameras[0]

    # Convert to np c2w matrices, get mean c2w.
    cam_c2ws = np.stack([cam_to_numpy_c2w(camera) for camera in cameras]) # (n,4,4)
    mean_c2w = mean_camera_pose(cam_c2ws) 

    # Define pivot c2w matrix.
    up = mean_c2w[:3,1]
    eye = mean_c2w[:3,3]
    # c2w_pivot = get_c2w_from_eye(eye, lookat, up)
    c2w_pivot = mean_c2w.copy()

    # Define "look-at" by average depth of center-cropped rendering.
    camera_pivot = numpy_c2w_to_cam(ref_cam=ref_camera, c2w=c2w_pivot)
    depth_pivot = render(camera_pivot, scene.gaussians, torch.zeros(3).cuda())['depth']
    _,H,W = depth_pivot.shape
    lookat_z = depth_pivot[:, H//4:H*3//4 , W//4:W*3//4].mean().cpu().numpy()
    lookat = eye + lookat_z * c2w_pivot[:3,2]

    # Length between eye and lookat
    l = np.linalg.norm(eye-lookat)

    # get "circle"
    radius_x = math.tan(spin_angle) * l
    radius_y = math.tan(spin_angle) * l

    # make it array.
    radius_x = np.linspace(radius_x/spin_for, radius_x, n_frames * spin_for)
    radius_y = np.linspace(radius_y/spin_for, radius_y, n_frames * spin_for)
    
    x_pivot_coords = np.tile(np.cos(np.linspace(0.0, 2.0*np.pi, n_frames)),spin_for) * radius_x
    y_pivot_coords = np.tile(np.sin(np.linspace(0.0, 2.0*np.pi, n_frames)),spin_for) * radius_y
    z_pivot_coords = np.zeros(n_frames*spin_for)
    
    pivot_coords = np.stack([x_pivot_coords,y_pivot_coords,z_pivot_coords,np.ones_like(z_pivot_coords)],axis=0) #[4,n_frames]
    
    eyes_circle = ((c2w_pivot@pivot_coords).T)[:,:3] # [n_frames, 3]
    c2ws =  np.stack([get_c2w_from_eye(eye_cam, lookat, up) for eye_cam in eyes_circle]) #[n_frames, 3, 3]
    
    result_cams = []

    for c2w in c2ws:
        result_cams.append(numpy_c2w_to_cam(ref_cam=ref_camera, c2w=c2w))

    return result_cams
def make_video(imgs, path, fps=32):

    writer = imageio.get_writer(path , fps=fps)
    
    for img in imgs:
        writer.append_data(img)    
    writer.close()

def center_crop_with_ratio(x, ratio):
    """
    ARUMENTS
    --------
    x: np.ndarray (b,h,w,c) or [h,w,c]

    RETURNS
    -------
    cropped img (b,h',w',c) or (h',w',c)
    """
    assert 3 <= len(x.shape) <= 4
    is_batched = len(x.shape) == 4
    if not is_batched:
        x = x[None,...]
    
    H,W = x.shape[1:3]
    crop_ch, crop_cw = H/2, W/2
    crop_lenh, crop_lenw = H*ratio, W*ratio

    h1 = int(crop_ch - crop_lenh/2)
    h2 = int(crop_ch + crop_lenh/2)

    w1 = int(crop_cw - crop_lenw/2)
    w2 = int(crop_cw + crop_lenw/2)

    x = x[:,h1:h2,w1:w2,:]
    
    if not is_batched:
        x = x[0]
    
    return x

```

# prompts/papers/deblurgs/utils/general_utils.py

``` py
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random
import math


def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        """
        Changed Behavior for DeblurGS
        """
        nonlocal lr_init, lr_final, lr_delay_steps, max_steps
        step -= lr_delay_steps
        max_steps -= lr_delay_steps
        if step < 0:
            return lr_init
        elif step > max_steps:
            return lr_final
        elif lr_init <= 0.0:
            return 0.0
        elif lr_init <= lr_final:
            return lr_init
        if lr_final <= 0.0:
            lr_final = 1e-6

        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return log_lerp

    return helper

def get_scheduler(lr_init, lr_final, warmup_ratio, step_warmup, step_final):
    """Return a scheduler function that handles exponential growth during warmup and exponential decay."""
    
    def get_lr(step):
        """Calculate the learning rate for a given step."""
        if step < 1:
            raise ValueError("Step must be greater than 0")
        
        lr_start = lr_init * warmup_ratio

        # Exponential warmup phase
        if step <= step_warmup:
            lr = 0.0
            # warmup_rate = math.log(lr_init / lr_start) / (step_warmup - 1)
            # lr = lr_start * math.exp(warmup_rate * (step - 1))
        
        # Exponential decay phase
        elif step <= step_final:
            if lr_init <= 1e-8:
                return 0.0
            decay_rate = math.log(lr_final / lr_init) / (step_final - step_warmup)
            lr = lr_init * math.exp(decay_rate * (step - step_warmup))
        
        else:
            lr = lr_final
        
        return lr
    
    return get_lr

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


```

# prompts/papers/deblurgs/utils/graphics_utils.py

``` py
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

```

# prompts/papers/deblurgs/utils/image_utils.py

``` py
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


```

# prompts/papers/deblurgs/utils/logger.py

``` py

import tqdm
import torch
class Logger:
    
    def __init__(self, progress_bar:tqdm, ema_weight:float=0.6):
        self.pbar = progress_bar
        self.ema_weight = ema_weight

        self.log_dic = {}

    def update(self, display_dict:dict):
        
        for key, (newval,logtype,fmt) in display_dict.items():
            if torch.is_tensor(newval):
                newval = newval.item()
            if logtype.strip().lower() == "ema":    
                self.log_dic[key] = (self.log_dic.get(key,(0,fmt))[0] * self.ema_weight + newval * (1.0-self.ema_weight),fmt)
            elif logtype.strip().lower() == "update":
                self.log_dic[key] = (newval,fmt)
            else:
                raise NotImplementedError

    def show(self):
        self.pbar.set_postfix({k:format(val,fmt) for k,(val,fmt) in self.log_dic.items()})
        self.pbar.update(10)

```

# prompts/papers/deblurgs/utils/loss_utils.py

``` py
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def tv_loss(x:torch.Tensor):
    """
    ARGUMENTS
    ---------
    x: torch.Tensor [b,c,h,w]

    RETURNS
    -------
    smoothness_loss torch.tensor (float) of size(1)
    """
    horizontal_loss = l2_loss(x[: , : , :-1 , :] , x[ : , : , 1 : , :])
    vertical_loss   = l2_loss(x[: , :,  : , :-1] , x[ : , : , : , 1: ])
    return horizontal_loss + vertical_loss

def batchwise_smoothness_loss(x:torch.Tensor):
    """
    ARGUMENTS
    ---------
    x: torch.Tensor [b,3,h,w]

    RETURNS
    -------
    smoothness_loss (batch-side tv loss) torch.tensor (float) of size(1)
    """
    device= x.device
    if x.shape[0]==1:
        return torch.zeros(1, device=device)
    return l1_loss( x[1:], x[:-1] )


def hinge_l2(x:torch.Tensor):
    """
    hinge loss
    """
    loss = torch.zeros_like(x)
    
    loss[x<=0.0] = x[x<=0.0] ** 2
    loss[x>=1.0] = (x[x>=1.0] - 1.0) ** 2
    return loss.mean()


```

# prompts/papers/deblurgs/utils/mvg_utils.py

``` py


import numpy as np
from scipy.spatial.transform import Rotation as R

def build_K(fx, fy, cx, cy):
    K = np.array([[fx, 0., cx,],
                  [0., fy, cy,],
                  [0., 0., 1.,]]).astype(np.float64)
    return K

def get_normalized_coords(pixel_coords, K):
    
    pixel_coords_homog = np.pad(pixel_coords, ((0,0),(0,1)), 'constant', constant_values=1.0)    
    
    K_inv = np.linalg.inv(K)
    norm_coords = K_inv @ pixel_coords_homog.T

    return (norm_coords.T)[...,:2] 


def normalized_coords_to_cam_coords(normalized_coords, depth):
    normalized_coords_homog = np.pad(normalized_coords, ((0,0),(0,1)), 'constant', constant_values=1.0)  # (hw, 3)
    return normalized_coords_homog * depth.reshape(-1,1)

def cam_to_world_coords(cam_coords, c2w):
    """
    cam_coords: (hw, 3)
    c2w: (4,4)
    """
    cam_coords_homog = np.pad(cam_coords, ((0,0),(0,1)), 'constant', constant_values=1.0)  # (hw,4)
    world_coords = c2w @ cam_coords_homog.T

    return (world_coords.T)[...,:3]



def to_w2c(c2w):
    """
    build w2c 4x4 matrix from c2w ((3x4) or (4x4))
    """

    w2c = np.eye(4,4)
    
    cam_loc = c2w[:3,3]
    rot_c2w = c2w[:3,:3]

    rot_w2c = rot_c2w.T
    trans_vec = -rot_w2c@cam_loc

    w2c[:3, :3] = rot_w2c
    w2c[:3,  3] = trans_vec
    w2c = w2c.astype(np.float64)
    return w2c

def mean_camera_pose(c2ws):
    """
    Compute the mean camera pose from a list of SE(3) matrices.

    Parameters:
    - se3_matrices (numpy.ndarray): Array of SE(3) matrices of shape (n, 4, 4), where n is the number of matrices.

    Returns:
    - numpy.ndarray: Mean SE(3) matrix representing the averaged camera pose.
    """
    translations = c2ws[:, :3, 3]  # Extract translation vectors
    rotations = R.from_matrix(c2ws[:, :3, :3])  # Extract rotation matrices

    # Compute mean translation
    mean_translation = np.mean(translations, axis=0)

    # Compute mean rotation
    mean_rotation = rotations.mean().as_matrix()

    # Construct mean SE(3) matrix
    mean_se3_matrix = np.eye(4)
    mean_se3_matrix[:3, :3] = mean_rotation
    mean_se3_matrix[:3, 3] = mean_translation

    return mean_se3_matrix


def get_c2w_from_eye(eye, lookat, up):
    # get c2w matrix for pivot camera.
    z_vec = lookat-eye
    x_vec = np.cross(up,z_vec) 
    y_vec = np.cross(z_vec,x_vec)

    x_vec = x_vec/np.linalg.norm(x_vec)
    y_vec = y_vec/np.linalg.norm(y_vec)
    z_vec = z_vec/np.linalg.norm(z_vec)
    
    rot_pivot = np.stack([x_vec,y_vec,z_vec],axis=0).T
    
    c2w = np.eye(4)
    c2w[:3,:3] = rot_pivot
    c2w[:3,3] = eye
    return c2w

```

# prompts/papers/deblurgs/utils/pytorch3d_functions.py

``` py

"""
Copyright: pytorch3d.
As installing pytorch3d usually causes version error, 
some codes are imported manually.
"""

import warnings
from typing import Tuple
import torch
import math

DEFAULT_ACOS_BOUND: float = 1.0 - 1e-4


def _dacos_dx(x: float) -> float:
    """
    Calculates the derivative of `arccos(x)` w.r.t. `x`.
    """
    return (-1.0) / math.sqrt(1.0 - x * x)
def _acos_linear_approximation(x: torch.Tensor, x0: float) -> torch.Tensor:
    """
    Calculates the 1st order Taylor expansion of `arccos(x)` around `x0`.
    """
    return (x - x0) * _dacos_dx(x0) + math.acos(x0)
def acos_linear_extrapolation(
    x: torch.Tensor,
    bounds: Tuple[float, float] = (-DEFAULT_ACOS_BOUND, DEFAULT_ACOS_BOUND),
) -> torch.Tensor:
    """
    Implements `arccos(x)` which is linearly extrapolated outside `x`'s original
    domain of `(-1, 1)`. This allows for stable backpropagation in case `x`
    is not guaranteed to be strictly within `(-1, 1)`.

    More specifically::

        bounds=(lower_bound, upper_bound)
        if lower_bound <= x <= upper_bound:
            acos_linear_extrapolation(x) = acos(x)
        elif x <= lower_bound: # 1st order Taylor approximation
            acos_linear_extrapolation(x)
                = acos(lower_bound) + dacos/dx(lower_bound) * (x - lower_bound)
        else:  # x >= upper_bound
            acos_linear_extrapolation(x)
                = acos(upper_bound) + dacos/dx(upper_bound) * (x - upper_bound)

    Args:
        x: Input `Tensor`.
        bounds: A float 2-tuple defining the region for the
            linear extrapolation of `acos`.
            The first/second element of `bound`
            describes the lower/upper bound that defines the lower/upper
            extrapolation region, i.e. the region where
            `x <= bound[0]`/`bound[1] <= x`.
            Note that all elements of `bound` have to be within (-1, 1).
    Returns:
        acos_linear_extrapolation: `Tensor` containing the extrapolated `arccos(x)`.
    """

    lower_bound, upper_bound = bounds

    if lower_bound > upper_bound:
        raise ValueError("lower bound has to be smaller or equal to upper bound.")

    if lower_bound <= -1.0 or upper_bound >= 1.0:
        raise ValueError("Both lower bound and upper bound have to be within (-1, 1).")

    # init an empty tensor and define the domain sets
    acos_extrap = torch.empty_like(x)
    x_upper = x >= upper_bound
    x_lower = x <= lower_bound
    x_mid = (~x_upper) & (~x_lower)

    # acos calculation for upper_bound < x < lower_bound
    acos_extrap[x_mid] = torch.acos(x[x_mid])
    # the linear extrapolation for x >= upper_bound
    acos_extrap[x_upper] = _acos_linear_approximation(x[x_upper], upper_bound)
    # the linear extrapolation for x <= lower_bound
    acos_extrap[x_lower] = _acos_linear_approximation(x[x_lower], lower_bound)

    return acos_extrap
def so3_relative_angle(
    R1: torch.Tensor,
    R2: torch.Tensor,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Calculates the relative angle (in radians) between pairs of
    rotation matrices `R1` and `R2` with `angle = acos(0.5 * (Trace(R1 R2^T)-1))`

    .. note::
        This corresponds to a geodesic distance on the 3D manifold of rotation
        matrices.

    Args:
        R1: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        R2: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        cos_angle: If==True return cosine of the relative angle rather than
            the angle itself. This can avoid the unstable calculation of `acos`.
        cos_bound: Clamps the cosine of the relative rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.
        eps: Tolerance for the valid trace check of the relative rotation matrix
            in `so3_rotation_angle`.
    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R1` or `R2` is of incorrect shape.
        ValueError if `R1` or `R2` has an unexpected trace.
    """
    R12 = torch.bmm(R1, R2.permute(0, 2, 1))
    return so3_rotation_angle(R12, cos_angle=cos_angle, cos_bound=cos_bound, eps=eps)



def so3_rotation_angle(
    R: torch.Tensor,
    eps: float = 1e-4,
    cos_angle: bool = False,
    cos_bound: float = 1e-4,
) -> torch.Tensor:
    """
    Calculates angles (in radians) of a batch of rotation matrices `R` with
    `angle = acos(0.5 * (Trace(R)-1))`. The trace of the
    input matrices is checked to be in the valid range `[-1-eps,3+eps]`.
    The `eps` argument is a small constant that allows for small errors
    caused by limited machine precision.

    Args:
        R: Batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: Tolerance for the valid trace check.
        cos_angle: If==True return cosine of the rotation angles rather than
            the angle itself. This can avoid the unstable
            calculation of `acos`.
        cos_bound: Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call. Note that the non-finite outputs/gradients
            are returned when the angle is requested (i.e. `cos_angle==False`)
            and the rotation angle is close to 0 or π.

    Returns:
        Corresponding rotation angles of shape `(minibatch,)`.
        If `cos_angle==True`, returns the cosine of the angles.

    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    if ((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
        raise ValueError("A matrix has trace outside valid range [-1-eps,3+eps].")

    # phi ... rotation angle
    phi_cos = (rot_trace - 1.0) * 0.5

    if cos_angle:
        return phi_cos
    else:
        if cos_bound > 0.0:
            bound = 1.0 - cos_bound
            return acos_linear_extrapolation(phi_cos, (-bound, bound))
        else:
            return torch.acos(phi_cos)


def so3_exp_map(log_rot: torch.Tensor, eps: float = 0.0001) -> torch.Tensor:
    """
    Convert a batch of logarithmic representations of rotation matrices `log_rot`
    to a batch of 3x3 rotation matrices using Rodrigues formula [1].

    In the logarithmic representation, each rotation matrix is represented as
    a 3-dimensional vector (`log_rot`) who's l2-norm and direction correspond
    to the magnitude of the rotation angle and the axis of rotation respectively.

    The conversion has a singularity around `log(R) = 0`
    which is handled by clamping controlled with the `eps` argument.

    Args:
        log_rot: Batch of vectors of shape `(minibatch, 3)`.
        eps: A float constant handling the conversion singularity.

    Returns:
        Batch of rotation matrices of shape `(minibatch, 3, 3)`.

    Raises:
        ValueError if `log_rot` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    return _so3_exp_map(log_rot, eps=eps)[0]


def so3_exponential_map(log_rot: torch.Tensor, eps: float = 0.0001) -> torch.Tensor:
    warnings.warn(
        """so3_exponential_map is deprecated,
        Use so3_exp_map instead.
        so3_exponential_map will be removed in future releases.""",
        PendingDeprecationWarning,
    )

    return so3_exp_map(log_rot, eps)




def _so3_exp_map(
    log_rot: torch.Tensor, eps: float = 0.0001
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A helper function that computes the so3 exponential map and,
    apart from the rotation matrix, also returns intermediate variables
    that can be re-used in other functions.
    """
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)

    R = (
        fac1[:, None, None] * skews
        # pyre-fixme[16]: `float` has no attribute `__getitem__`.
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    return R, rot_angles, skews, skews_square


def so3_log_map(
    R: torch.Tensor, eps: float = 0.0001, cos_bound: float = 1e-4
) -> torch.Tensor:
    """
    Convert a batch of 3x3 rotation matrices `R`
    to a batch of 3-dimensional matrix logarithms of rotation matrices
    The conversion has a singularity around `(R=I)` which is handled
    by clamping controlled with the `eps` and `cos_bound` arguments.

    Args:
        R: batch of rotation matrices of shape `(minibatch, 3, 3)`.
        eps: A float constant handling the conversion singularity.
        cos_bound: Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 1 - cos_bound] to avoid non-finite outputs/gradients
            of the `acos` call when computing `so3_rotation_angle`.
            Note that the non-finite outputs/gradients are returned when
            the rotation angle is close to 0 or π.

    Returns:
        Batch of logarithms of input rotation matrices
        of shape `(minibatch, 3)`.

    Raises:
        ValueError if `R` is of incorrect shape.
        ValueError if `R` has an unexpected trace.
    """

    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    phi = so3_rotation_angle(R, cos_bound=cos_bound, eps=eps)

    phi_sin = torch.sin(phi)

    # We want to avoid a tiny denominator of phi_factor = phi / (2.0 * phi_sin).
    # Hence, for phi_sin.abs() <= 0.5 * eps, we approximate phi_factor with
    # 2nd order Taylor expansion: phi_factor = 0.5 + (1.0 / 12) * phi**2
    phi_factor = torch.empty_like(phi)
    ok_denom = phi_sin.abs() > (0.5 * eps)
    # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
    phi_factor[~ok_denom] = 0.5 + (phi[~ok_denom] ** 2) * (1.0 / 12)
    phi_factor[ok_denom] = phi[ok_denom] / (2.0 * phi_sin[ok_denom])

    log_rot_hat = phi_factor[:, None, None] * (R - R.permute(0, 2, 1))

    log_rot = hat_inv(log_rot_hat)

    return log_rot


def hat_inv(h: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse Hat operator [1] of a batch of 3x3 matrices.

    Args:
        h: Batch of skew-symmetric matrices of shape `(minibatch, 3, 3)`.

    Returns:
        Batch of 3d vectors of shape `(minibatch, 3, 3)`.

    Raises:
        ValueError if `h` is of incorrect shape.
        ValueError if `h` not skew-symmetric.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim1, dim2 = h.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    ss_diff = torch.abs(h + h.permute(0, 2, 1)).max()

    HAT_INV_SKEW_SYMMETRIC_TOL = 1e-5
    if float(ss_diff) > HAT_INV_SKEW_SYMMETRIC_TOL:
        raise ValueError("One of input matrices is not skew-symmetric.")

    x = h[:, 2, 1]
    y = h[:, 0, 2]
    z = h[:, 1, 0]

    v = torch.stack((x, y, z), dim=1)

    return v


def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.

    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`

    Raises:
        ValueError if `v` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h
def se3_exp_map(log_transform: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """
    Convert a batch of logarithmic representations of SE(3) matrices `log_transform`
    to a batch of 4x4 SE(3) matrices using the exponential map.
    See e.g. [1], Sec 9.4.2. for more detailed description.

    A SE(3) matrix has the following form:
        ```
        [ R 0 ]
        [ T 1 ] ,
        ```
    where `R` is a 3x3 rotation matrix and `T` is a 3-D translation vector.
    SE(3) matrices are commonly used to represent rigid motions or camera extrinsics.

    In the SE(3) logarithmic representation SE(3) matrices are
    represented as 6-dimensional vectors `[log_translation | log_rotation]`,
    i.e. a concatenation of two 3D vectors `log_translation` and `log_rotation`.

    The conversion from the 6D representation to a 4x4 SE(3) matrix `transform`
    is done as follows:
        ```
        transform = exp( [ hat(log_rotation) 0 ]
                         [   log_translation 1 ] ) ,
        ```
    where `exp` is the matrix exponential and `hat` is the Hat operator [2].

    Note that for any `log_transform` with `0 <= ||log_rotation|| < 2pi`
    (i.e. the rotation angle is between 0 and 2pi), the following identity holds:
    ```
    se3_log_map(se3_exponential_map(log_transform)) == log_transform
    ```

    The conversion has a singularity around `||log(transform)|| = 0`
    which is handled by clamping controlled with the `eps` argument.

    Args:
        log_transform: Batch of vectors of shape `(minibatch, 6)`.
        eps: A threshold for clipping the squared norm of the rotation logarithm
            to avoid unstable gradients in the singular case.

    Returns:
        Batch of transformation matrices of shape `(minibatch, 4, 4)`.

    Raises:
        ValueError if `log_transform` is of incorrect shape.

    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    [2] https://en.wikipedia.org/wiki/Hat_operator
    """

    if log_transform.ndim != 2 or log_transform.shape[1] != 6:
        raise ValueError("Expected input to be of shape (N, 6).")

    N, _ = log_transform.shape

    log_translation = log_transform[..., :3]
    log_rotation = log_transform[..., 3:]

    # rotation is an exponential map of log_rotation
    (
        R,
        rotation_angles,
        log_rotation_hat,
        log_rotation_hat_square,
    ) = _so3_exp_map(log_rotation, eps=eps)

    # translation is V @ T
    V = _se3_V_matrix(
        log_rotation,
        log_rotation_hat,
        log_rotation_hat_square,
        rotation_angles,
        eps=eps,
    )
    T = torch.bmm(V, log_translation[:, :, None])[:, :, 0]

    transform = torch.zeros(
        N, 4, 4, dtype=log_transform.dtype, device=log_transform.device
    )

    transform[:, :3, :3] = R
    transform[:, :3, 3] = T
    transform[:, 3, 3] = 1.0

    return transform.permute(0, 2, 1)




def se3_log_map(
    transform: torch.Tensor, eps: float = 1e-4, cos_bound: float = 1e-4
) -> torch.Tensor:
    """
    Convert a batch of 4x4 transformation matrices `transform`
    to a batch of 6-dimensional SE(3) logarithms of the SE(3) matrices.
    See e.g. [1], Sec 9.4.2. for more detailed description.

    A SE(3) matrix has the following form:
        ```
        [ R 0 ]
        [ T 1 ] ,
        ```
    where `R` is an orthonormal 3x3 rotation matrix and `T` is a 3-D translation vector.
    SE(3) matrices are commonly used to represent rigid motions or camera extrinsics.

    In the SE(3) logarithmic representation SE(3) matrices are
    represented as 6-dimensional vectors `[log_translation | log_rotation]`,
    i.e. a concatenation of two 3D vectors `log_translation` and `log_rotation`.

    The conversion from the 4x4 SE(3) matrix `transform` to the
    6D representation `log_transform = [log_translation | log_rotation]`
    is done as follows:
        ```
        log_transform = log(transform)
        log_translation = log_transform[3, :3]
        log_rotation = inv_hat(log_transform[:3, :3])
        ```
    where `log` is the matrix logarithm
    and `inv_hat` is the inverse of the Hat operator [2].

    Note that for any valid 4x4 `transform` matrix, the following identity holds:
    ```
    se3_exp_map(se3_log_map(transform)) == transform
    ```

    The conversion has a singularity around `(transform=I)` which is handled
    by clamping controlled with the `eps` and `cos_bound` arguments.

    Args:
        transform: batch of SE(3) matrices of shape `(minibatch, 4, 4)`.
        eps: A threshold for clipping the squared norm of the rotation logarithm
            to avoid division by zero in the singular case.
        cos_bound: Clamps the cosine of the rotation angle to
            [-1 + cos_bound, 3 - cos_bound] to avoid non-finite outputs.
            The non-finite outputs can be caused by passing small rotation angles
            to the `acos` function in `so3_rotation_angle` of `so3_log_map`.

    Returns:
        Batch of logarithms of input SE(3) matrices
        of shape `(minibatch, 6)`.

    Raises:
        ValueError if `transform` is of incorrect shape.
        ValueError if `R` has an unexpected trace.

    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    [2] https://en.wikipedia.org/wiki/Hat_operator
    """

    if transform.ndim != 3:
        raise ValueError("Input tensor shape has to be (N, 4, 4).")

    N, dim1, dim2 = transform.shape
    if dim1 != 4 or dim2 != 4:
        raise ValueError("Input tensor shape has to be (N, 4, 4).")

    if not torch.allclose(transform[:, :3, 3], torch.zeros_like(transform[:, :3, 3])):
        raise ValueError("All elements of `transform[:, :3, 3]` should be 0.")

    # log_rot is just so3_log_map of the upper left 3x3 block
    R = transform[:, :3, :3].permute(0, 2, 1)
    log_rotation = so3_log_map(R, eps=eps, cos_bound=cos_bound)

    # log_translation is V^-1 @ T
    T = transform[:, 3, :3]
    V = _se3_V_matrix(*_get_se3_V_input(log_rotation), eps=eps)
    log_translation = torch.linalg.solve(V, T[:, :, None])[:, :, 0]

    return torch.cat((log_translation, log_rotation), dim=1)




def _se3_V_matrix(
    log_rotation: torch.Tensor,
    log_rotation_hat: torch.Tensor,
    log_rotation_hat_square: torch.Tensor,
    rotation_angles: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    A helper function that computes the "V" matrix from [1], Sec 9.4.2.
    [1] https://jinyongjeong.github.io/Download/SE3/jlblanco2010geometry3d_techrep.pdf
    """

    V = (
        torch.eye(3, dtype=log_rotation.dtype, device=log_rotation.device)[None]
        + log_rotation_hat
        # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
        * ((1 - torch.cos(rotation_angles)) / (rotation_angles**2))[:, None, None]
        + (
            log_rotation_hat_square
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            * ((rotation_angles - torch.sin(rotation_angles)) / (rotation_angles**3))[
                :, None, None
            ]
        )
    )

    return V


def _get_se3_V_input(log_rotation: torch.Tensor, eps: float = 1e-4):
    """
    A helper function that computes the input variables to the `_se3_V_matrix`
    function.
    """
    # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and `int`.
    nrms = (log_rotation**2).sum(-1)
    rotation_angles = torch.clamp(nrms, eps).sqrt()
    log_rotation_hat = hat(log_rotation)
    log_rotation_hat_square = torch.bmm(log_rotation_hat, log_rotation_hat)
    return log_rotation, log_rotation_hat, log_rotation_hat_square, rotation_angles

```

# prompts/papers/deblurgs/utils/sh_utils.py

``` py
#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result

def RGB2SH(rgb, use_sigmoid=False):
    return (rgb) / C0 if use_sigmoid else (rgb - 0.5) / C0

def SH2RGB(sh, use_sigmoid=False):
    return sh * C0 if use_sigmoid else sh * C0 + 0.5

```

# prompts/papers/deblurgs/utils/system_utils.py

``` py
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from errno import EEXIST
from os import makedirs, path
import os
import sys

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

def do_system(arg):
    print(f"==== running: {arg}")
    err = os.system(arg)
    if err:
        print("FATAL: command failed")
        sys.exit(err)


```

# prompts/papers/deblurgs/utils/visualization.py

``` py

import os
from scene.cameras import Camera, get_c2w, c2w_to_cam
from scene import Scene
from scene.gaussian_model import GaussianModel
from arguments import OptimizationParams

from scene.motion import CameraMotionModule
import torch
import torchvision
import torch.nn as nn
import numpy as np
from gaussian_renderer import render
import math
import cv2
import matplotlib.pyplot as plt
import render_spiral
import shutil
import utils.colorize
import open3d as o3d
import utils.mvg_utils


class Visualizer:
    
    def __init__(self, opt:OptimizationParams, scene: Scene, gaussians: GaussianModel, 
                 bg_color, 
                 alignment_vis_folder="vis_alignment", traj_vis_folder = "vis_traj",
                 n_visualize_shots:int=200,
                 exponential:float=1.7,
                 vis_cam_idx = None):
        """
        Visualization class.
        
        ARGUMENTS
        ---------
        scene: Scene class
        gaussians: Gaussian Splatting model.
        n_visualization_shots:
            number of visualization shots.
        exponential:
            visualization iteration will be decided by shape of function f(x)={x^exponential}
        """

        print("Initializing Visualizer...")
        self.gaussians = gaussians
        self.scene = scene
        self.bg_color = bg_color
        self.draw_camera = True
        self.num_visualize_subframes = 3

        self.vis_iters = self._get_vis_iteration(n_visualize_shots=n_visualize_shots, alpha=exponential, n_iters=opt.iterations)

        # Prepare Alignment Visualizer.
        n = len(scene.camera_motion_module)
        possible_indice = np.arange(n)
        self.selected_indice = np.random.choice(possible_indice, size=9, replace=False)
        self.selected_indice.sort()
        
        # Prepare Trajectory Visualizer.
        self.ref_camera = self._get_visualization_camera(scene, gaussians, vis_cam_idx)
        self.cam_scale = self._get_camera_scale(scene, gaussians)

        # Prepare directory.
        self.alignment_path = os.path.join(scene.model_path, alignment_vis_folder)
        self.traj_vis_path = os.path.join(scene.model_path, traj_vis_folder)

        shutil.rmtree(self.alignment_path, ignore_errors=True)
        os.makedirs(self.alignment_path)
        
        shutil.rmtree(self.traj_vis_path, ignore_errors=True)
        os.makedirs(self.traj_vis_path)
        print("[Done] Initializing Visualizer.")
        

    def _get_vis_iteration(self, n_visualize_shots, alpha, n_iters=30000):
        a = n_iters / n_visualize_shots ** (alpha)

        visualize_iters = a * (np.arange(1, n_visualize_shots+1).astype(float))**alpha
        visualize_iters = visualize_iters.astype(int)
        return visualize_iters
        
    def _get_visualization_camera(self, scene:Scene, gaussians:GaussianModel, vis_cam_idx=None, threshold = 0.5):
        """
        obtain "reasonable" camera to watch observation process.
        """
        
        if vis_cam_idx is not None:
            self.draw_camera = False
            return self.sample_subframe_cams(idx=vis_cam_idx, num_subframes=1)[0]
        
        print(" ==> searching for reasonable camera")
        
        lookat = gaussians._xyz.detach().cpu().numpy().mean(axis=0)
        pts = np.stack([cam.camera_center.cpu().numpy() for cam in scene.getTrainCameras()]) # (n,3)
        
        # Binary search for the lowest zoom which can see all cameras.

        zoom_lb, zoom_ub = 1.5, 100.0

        while zoom_ub - zoom_lb >= 1e-3:
            zoom = (zoom_lb+zoom_ub) / 2.0
            c2ws = np.stack([get_c2w(cam) for cam in scene.getTrainCameras()])
            mean_c2w = utils.mvg_utils.mean_camera_pose(c2ws)
            eye = mean_c2w[:3,3]
            up = mean_c2w[:3,1]
            zoomout_eye = lookat + zoom * (eye-lookat)
            zoomout_c2w = utils.mvg_utils.get_c2w_from_eye(zoomout_eye,lookat,up)
            zoomout_cam = c2w_to_cam(ref_cam=scene.getTrainCameras()[0], c2w=zoomout_c2w)
            W,H  = zoomout_cam.image_width, zoomout_cam.image_height

            pts_hom = np.pad(pts, ((0,0),(0,1)), 'constant', constant_values=1.0) # (n,4)
            pts_cam_hom = pts_hom @ zoomout_cam.world_view_transform.cpu().numpy() # (n,4)
            pts_cam = pts_cam_hom[:,:3] / pts_cam_hom[:,3:] # (n,3)
            pts_cheirality = pts_cam[:,2] >= 0.1 # (n,)

            pts_ndc_hom = pts_hom @ zoomout_cam.full_proj_transform.cpu().numpy() # (n,4)
            pts_ndc = pts_ndc_hom[:,:3] / pts_ndc_hom[:,3:] # (n,3)
            
            pts_pix = (( pts_ndc[:,:2] + 1.0) * np.array([zoomout_cam.image_width,zoomout_cam.image_height]).astype(float) -1.0) * 0.5 # (n,2)

            pts_inside = np.logical_and( np.logical_and( pts_pix[:,0] >= -threshold*W , pts_pix[:,0] <= (1.0+threshold)*W) , 
                                         np.logical_and( pts_pix[:,1] >= -threshold*H , pts_pix[:,1] <= (1.0+threshold)*H) )
            
            pts_good = np.logical_and(pts_inside, pts_cheirality)

            if pts_good.all():
                zoom_ub = zoom
            else:
                zoom_lb = zoom
        
        return zoomout_cam

    def _get_camera_scale(self, scene, gaussians):
        return 0.5   

    @torch.no_grad()
    def draw_cone_on_render_img(self, cam_render: Camera, rendered_img:np.ndarray, cams_for_draw:list, scale=1.0, color=np.array([0,0,255])):
        """
        Draw camera cone on the rendered_img, which is rendered from cam.
        
        ARGUMENTS
        ---------
        cam_render: Camera object used for render 'rendered_img'
        rendered_img: np.array (H,W,3)
        cam_for_draw: Camera object to be painted. 
        scale: float. Decides how large the cone is.
        color: RGB format
        RETURNS
        -------
        rendered_img with cam cone.
        """
        if not self.draw_camera:
            return rendered_img
        
        color = np.ascontiguousarray(color[::-1]) # to BGR format for cv2
        H,W,_ = rendered_img.shape
        
        c2ws_draw = np.stack([get_c2w(cam) for cam in cams_for_draw])
        for cam_draw, c2w_draw in zip(cams_for_draw, c2ws_draw):
            cone_x, cone_y = math.tan(cam_draw.FoVx/2), math.tan(cam_draw.FoVy/2)

            cone_camera_draw_space_homog = np.pad(np.array([[ 0.0   ,   0.0  , 0.0],
                                                            [ cone_x,  cone_y, 1.0],
                                                            [ cone_x, -cone_y, 1.0], 
                                                            [-cone_x, -cone_y, 1.0],
                                                            [-cone_x,  cone_y, 1.0]]) * scale , 
                                                ((0,0),(0,1)),
                                                'constant',
                                                constant_values=1.0) # (5,4)
            
            cone_world_space_homog = cone_camera_draw_space_homog @ c2w_draw.T # (5,4)
            
            cam_hom = cone_camera_draw_space_homog @ cam_render.world_view_transform.cpu().numpy() # (5,4)
            if (cam_hom[:,2]/cam_hom[:,3] < 0.1).any():
                continue
            
            ndc_hom = cone_world_space_homog @ cam_render.full_proj_transform.cpu().numpy() # (5,4)
            ndc = ndc_hom[:,:3] / ndc_hom[:,3:] # [5,3]

            pix = (( ndc[:,:2] + 1.0) * np.array([W,H]).astype(float) -1.0) * 0.5 # [5,2]
            connectivity = [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)]
            for i,j in connectivity:
                try:
                    rendered_img = cv2.line(rendered_img, pix[i].astype(int).tolist(), pix[j].astype(int).tolist(), color.tolist(), thickness=1)
                except Exception as e:
                    pass # TODO do something later
                    # print("[ERROR]" ,e)
        return rendered_img
            
    @torch.no_grad()
    def render_gaussian_and_cams(self, iteration):
        
        tonemapping = self.scene.tone_mapping

        rendered = tonemapping(render(self.ref_camera, self.gaussians, self.bg_color)["render"]).permute(1,2,0).cpu().numpy()
        rendered = np.ascontiguousarray((rendered * 255.0).clip(0.0,255.0).astype(np.uint8)[:,:,::-1])
        
        color1 = np.array([0,255,255])
        color2 = np.array([255,255,0])
        t = np.linspace(0, 1, len(self.scene.camera_motion_module))[:,None]
        colors = ((1-t)*color1 + t*color2).astype(np.uint8)
        
        for i ,color in enumerate(colors):
            subframe_cams = self.sample_subframe_cams(i, num_subframes=5)
            rendered = self.draw_cone_on_render_img(self.ref_camera, rendered, subframe_cams ,scale=self.cam_scale,color=color)

        cv2.imwrite(os.path.join(self.traj_vis_path, f"{iteration:05d}.png" ), rendered) # RGB2BGR

    def sample_subframe_cams(self, idx, num_subframes=None):
        t = self.scene.camera_motion_module._sample_nu_from_alignment(idx)
        if num_subframes is not None:
            subfr_idx = torch.linspace(0,t.shape[0]-1, num_subframes, device=t.device).long()
            t = t[subfr_idx]
        traj = self.scene.camera_motion_module.get_trajectory(idx, t)
        
        return traj
    @torch.no_grad()
    def visualize_alignment(self, iteration):
        
        # Create a 3x3 grid of subplots
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 10))

        # Generate and plot data in each subplot
        for i in range(3):
            for j in range(3):

                # get camera index for visualization.
                idx = self.selected_indice[i*3+j]
                
                nu = self.scene.camera_motion_module._sample_nu_from_alignment(idx).detach().cpu().numpy()
                nu_pivot = torch.sigmoid(self.scene.camera_motion_module._nu[idx].sort().values).detach().cpu().numpy()
                y = np.linspace(0.0,1.0,nu.shape[0])
                # Plotting the histogram in the current subplot
                
                axes[i,j].plot(nu, y, 'o', markersize=2)

                y = np.linspace(0.0, 1.0,nu_pivot.shape[0])
                axes[i,j].plot(nu_pivot, y, 'o', color="red", markersize=3)
                
                # Adding labels and title to each subplot
                axes[i, j].set_xlabel('nu')
                axes[i, j].set_title(f'Idx {idx}')

                axes[i, j].set_ylim(-0.1,1.1)

        # Adjust layout to prevent overlapping
        plt.tight_layout()

        # Save the entire figure as an image file 
        plt.savefig(os.path.join( self.alignment_path, f"{iteration:05d}.jpg") )

        plt.close()

    @torch.no_grad()
    def run(self, current_iter):
        if current_iter in self.vis_iters:
            self.render_gaussian_and_cams(current_iter)
            self.visualize_alignment(current_iter)


    @torch.no_grad()
    def traj_render(self, current_iter):

        render_path = f"{self.scene.model_path}/traj_render_{current_iter:05d}" 
        shutil.rmtree(render_path, ignore_errors=True)
        os.makedirs(render_path)
        
        print(f"Iter {current_iter} : render traj.")
        cmm:CameraMotionModule = self.scene.camera_motion_module
        
        for i in range(len(self.scene.camera_motion_module)):
            
            blur_retrieve:dict = cmm.query(cam_idx=i, 
                                           subframe_indice="all", 
                                           post_process=self.scene.tone_mapping)
            blurred = blur_retrieve["blurred"]
            gt = blur_retrieve["gt"]

            subframe_renders = cmm.query(cam_idx=i, 
                                         subframe_indice=self.num_visualize_subframes, 
                                         post_process=self.scene.tone_mapping)["subframes"]
            for j, subframe_render in enumerate(subframe_renders):
                torchvision.utils.save_image(self.scene.tone_mapping(subframe_render).clamp(0.0,1.0), 
                                             os.path.join(render_path, f"{i:03d}_{j:02d}.png"))
            
            torchvision.utils.save_image(blurred.clamp(0.0,1.0), os.path.join(render_path, f"{i:03d}_blur.png"))
            torchvision.utils.save_image(gt, os.path.join(render_path, f"{i:03d}_gt.png"))

            errormap = utils.colorize.colorize(torch.abs(blurred - gt).permute(1,2,0).mean(dim=-1)).permute(2,0,1)
            torchvision.utils.save_image(errormap, os.path.join(render_path, f"{i:03d}_l1.png"))


            

    def save_video(self):
        for video_frame_path in [self.alignment_path, self.traj_vis_path]:
            files = [e for e in os.listdir(video_frame_path) if e.endswith(".png") or e.endswith(".jpg")]
            files.sort(key=lambda x:int(x.split(".")[0]))
            imgs = []
            for file in files:
                full_path = os.path.join(video_frame_path, file)            
                img = cv2.imread(full_path)[:,:,::-1]
                imgs.append(img)
            
            imgs = np.stack(imgs)
            video_name = video_frame_path.split("/")[-1].strip()
            
            render_spiral.make_video(imgs, os.path.join(self.scene.model_path,f"{video_name}.mp4"),fps=20)



```

