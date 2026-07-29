# prompts/papers/IBRNet/config.py

``` py
# Copyright 2020 Google LLC
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


import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()
    # general
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--rootdir', type=str,
                        default='/home/qw246/S7/code/IBRNet/',
                        help='the path to the project root directory. Replace this path with yours!')
    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument('--distributed', action='store_true', help='if use distributed training')
    parser.add_argument("--local_rank", type=int, default=0, help='rank for distributed training')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')

    ########## dataset options ##########
    ## train and eval dataset
    parser.add_argument("--train_dataset", type=str, default='ibrnet_collected',
                        help='the training dataset, should either be a single dataset, '
                             'or multiple datasets connected with "+", for example, ibrnet_collected+llff+spaces')
    parser.add_argument("--dataset_weights", nargs='+', type=float, default=[],
                        help='the weights for training datasets, valid when multiple datasets are used.')
    parser.add_argument("--train_scenes", nargs='+', default=[],
                        help='optional, specify a subset of training scenes from training dataset')
    parser.add_argument('--eval_dataset', type=str, default='llff_test', help='the dataset to evaluate')
    parser.add_argument('--eval_scenes', nargs='+', default=[],
                        help='optional, specify a subset of scenes from eval_dataset to evaluate')
    ## others
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, '
                             'useful for large datasets like deepvoxels or nerf_synthetic')

    ########## model options ##########
    ## ray sampling options
    parser.add_argument('--sample_mode', type=str, default='uniform',
                        help='how to sample pixels from images for training:'
                             'uniform|center')
    parser.add_argument('--center_ratio', type=float, default=0.8, help='the ratio of center crop to keep')
    parser.add_argument("--N_rand", type=int, default=32 * 16,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--chunk_size", type=int, default=1024 * 4,
                        help='number of rays processed in parallel, decrease if running out of memory')

    ## model options
    parser.add_argument('--coarse_feat_dim', type=int, default=32, help="2D feature dimension for coarse level")
    parser.add_argument('--fine_feat_dim', type=int, default=32, help="2D feature dimension for fine level")
    parser.add_argument('--num_source_views', type=int, default=10,
                        help='the number of input source views for each target view')
    parser.add_argument('--rectify_inplane_rotation', action='store_true', help='if rectify inplane rotation')
    parser.add_argument('--coarse_only', action='store_true', help='use coarse network only')
    parser.add_argument("--anti_alias_pooling", type=int, default=1, help='if use anti-alias pooling')

    ########## checkpoints ##########
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ckpt_path", type=str, default="",
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--no_load_opt", action='store_true',
                        help='do not load optimizer when reloading')
    parser.add_argument("--no_load_scheduler", action='store_true',
                        help='do not load scheduler when reloading')

    ########### iterations & learning rate options ##########
    parser.add_argument("--n_iters", type=int, default=250000, help='num of iterations')
    parser.add_argument("--lrate_feature", type=float, default=1e-3, help='learning rate for feature extractor')
    parser.add_argument("--lrate_mlp", type=float, default=5e-4, help='learning rate for mlp')
    parser.add_argument("--lrate_decay_factor", type=float, default=0.5,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument("--lrate_decay_steps", type=int, default=50000,
                        help='decay learning rate by a factor every specified number of steps')

    ########## rendering options ##########
    parser.add_argument("--N_samples", type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=64, help='number of important samples per ray')
    parser.add_argument("--inv_uniform", action='store_true',
                        help='if True, will uniformly sample inverse depths')
    parser.add_argument("--det", action='store_true', help='deterministic sampling for coarse and fine samples')
    parser.add_argument("--white_bkgd", action='store_true',
                        help='apply the trick to avoid fitting to white background')
    parser.add_argument("--render_stride", type=int, default=1,
                        help='render with large stride for validation to save time')

    ########## logging/saving options ##########
    parser.add_argument("--i_print", type=int, default=100, help='frequency of terminal printout')
    parser.add_argument("--i_img", type=int, default=500, help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')

    ########## evaluation options ##########
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    return parser


```

# prompts/papers/IBRNet/eval/eval.py

``` py
# Copyright 2020 Google LLC
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


import sys
sys.path.append('../')
import imageio
from config import config_parser
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.render_image import render_single_image
from ibrnet.model import IBRNetModel
from utils import *
from ibrnet.projection import Projector
from ibrnet.data_loaders import dataset_dict
import tensorflow as tf
from lpips_tensorflow import lpips_tf
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"]="0"


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    args.distributed = False

    # Create IBRNet model
    model = IBRNetModel(args, load_scheduler=False, load_opt=False)
    eval_dataset_name = args.eval_dataset
    extra_out_dir = '{}/{}'.format(eval_dataset_name, args.expname)
    print("saving results to eval/{}...".format(extra_out_dir))
    os.makedirs(extra_out_dir, exist_ok=True)

    projector = Projector(device='cuda:0')

    assert len(args.eval_scenes) == 1, "only accept single scene"
    scene_name = args.eval_scenes[0]
    out_scene_dir = os.path.join(extra_out_dir, '{}_{:06d}'.format(scene_name, model.start_step))
    os.makedirs(out_scene_dir, exist_ok=True)

    test_dataset = dataset_dict[args.eval_dataset](args, 'test', scenes=args.eval_scenes)
    save_prefix = scene_name
    test_loader = DataLoader(test_dataset, batch_size=1)
    total_num = len(test_loader)
    results_dict = {scene_name: {}}
    sum_coarse_psnr = 0
    sum_fine_psnr = 0
    running_mean_coarse_psnr = 0
    running_mean_fine_psnr = 0
    sum_coarse_lpips = 0
    sum_fine_lpips = 0
    running_mean_coarse_lpips = 0
    running_mean_fine_lpips = 0
    sum_coarse_ssim = 0
    sum_fine_ssim = 0
    running_mean_coarse_ssim = 0
    running_mean_fine_ssim = 0

    pred_ph = tf.placeholder(tf.float32)
    gt_ph = tf.placeholder(tf.float32)
    distance_t = lpips_tf.lpips(pred_ph, gt_ph, model='net-lin', net='vgg')
    ssim_tf = tf.image.ssim(pred_ph, gt_ph, max_val=1.)
    psnr_tf = tf.image.psnr(pred_ph, gt_ph, max_val=1.)

    for i, data in enumerate(test_loader):
        rgb_path = data['rgb_path'][0]
        file_id = os.path.basename(rgb_path).split('.')[0]
        src_rgbs = data['src_rgbs'][0].cpu().numpy()

        averaged_img = (np.mean(src_rgbs, axis=0) * 255.).astype(np.uint8)
        imageio.imwrite(os.path.join(out_scene_dir, '{}_average.png'.format(file_id)),
                        averaged_img)

        model.switch_to_eval()
        with torch.no_grad():
            ray_sampler = RaySamplerSingleImage(data, device='cuda:0')
            ray_batch = ray_sampler.get_all()
            featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))

            ret = render_single_image(ray_sampler=ray_sampler,
                                      ray_batch=ray_batch,
                                      model=model,
                                      projector=projector,
                                      chunk_size=args.chunk_size,
                                      det=True,
                                      N_samples=args.N_samples,
                                      inv_uniform=args.inv_uniform,
                                      N_importance=args.N_importance,
                                      white_bkgd=args.white_bkgd,
                                      featmaps=featmaps)

            gt_rgb = data['rgb'][0]
            coarse_pred_rgb = ret['outputs_coarse']['rgb'].detach().cpu()
            coarse_err_map = torch.sum((coarse_pred_rgb - gt_rgb) ** 2, dim=-1).numpy()
            coarse_err_map_colored = (colorize_np(coarse_err_map, range=(0., 1.)) * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(out_scene_dir, '{}_err_map_coarse.png'.format(file_id)),
                            coarse_err_map_colored)
            coarse_pred_rgb_np = np.clip(coarse_pred_rgb.numpy()[None, ...], a_min=0., a_max=1.)
            gt_rgb_np = gt_rgb.numpy()[None, ...]

            # different implementation of the ssim and psnr metrics can be different.
            # we use the tf implementation for evaluating ssim and psnr to match the setup of NeRF paper.
            with tf.Session() as session:
                coarse_lpips = session.run(distance_t, feed_dict={pred_ph: coarse_pred_rgb_np, gt_ph: gt_rgb_np})[0]
                coarse_ssim = session.run(ssim_tf, feed_dict={pred_ph: coarse_pred_rgb_np, gt_ph: gt_rgb_np})[0]
                coarse_psnr = session.run(psnr_tf, feed_dict={pred_ph: coarse_pred_rgb_np, gt_ph: gt_rgb_np})[0]

            # saving outputs ...
            coarse_pred_rgb = (255 * np.clip(coarse_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
            imageio.imwrite(os.path.join(out_scene_dir, '{}_pred_coarse.png'.format(file_id)), coarse_pred_rgb)

            gt_rgb_np_uint8 = (255 * np.clip(gt_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
            imageio.imwrite(os.path.join(out_scene_dir, '{}_gt_rgb.png'.format(file_id)), gt_rgb_np_uint8)

            coarse_pred_depth = ret['outputs_coarse']['depth'].detach().cpu()
            imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_coarse.png'.format(file_id)),
                            (coarse_pred_depth.numpy().squeeze() * 1000.).astype(np.uint16))
            coarse_pred_depth_colored = colorize_np(coarse_pred_depth,
                                                    range=tuple(data['depth_range'].squeeze().cpu().numpy()))
            imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_vis_coarse.png'.format(file_id)),
                            (255 * coarse_pred_depth_colored).astype(np.uint8))
            coarse_acc_map = torch.sum(ret['outputs_coarse']['weights'], dim=-1).detach().cpu()
            coarse_acc_map_colored = (colorize_np(coarse_acc_map, range=(0., 1.)) * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(out_scene_dir, '{}_acc_map_coarse.png'.format(file_id)),
                            coarse_acc_map_colored)

            sum_coarse_psnr += coarse_psnr
            running_mean_coarse_psnr = sum_coarse_psnr / (i + 1)
            sum_coarse_lpips += coarse_lpips
            running_mean_coarse_lpips = sum_coarse_lpips / (i + 1)
            sum_coarse_ssim += coarse_ssim
            running_mean_coarse_ssim = sum_coarse_ssim / (i + 1)

            if ret['outputs_fine'] is not None:
                fine_pred_rgb = ret['outputs_fine']['rgb'].detach().cpu()
                fine_pred_rgb_np = np.clip(fine_pred_rgb.numpy()[None, ...], a_min=0., a_max=1.)

                with tf.Session() as session:
                    fine_lpips = session.run(distance_t, feed_dict={pred_ph: fine_pred_rgb_np, gt_ph: gt_rgb_np})[0]
                    fine_ssim = session.run(ssim_tf, feed_dict={pred_ph: fine_pred_rgb_np, gt_ph: gt_rgb_np})[0]
                    fine_psnr = session.run(psnr_tf, feed_dict={pred_ph: fine_pred_rgb_np, gt_ph: gt_rgb_np})[0]

                fine_err_map = torch.sum((fine_pred_rgb - gt_rgb) ** 2, dim=-1).numpy()
                fine_err_map_colored = (colorize_np(fine_err_map, range=(0., 1.)) * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(out_scene_dir, '{}_err_map_fine.png'.format(file_id)),
                                fine_err_map_colored)

                fine_pred_rgb = (255 * np.clip(fine_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
                imageio.imwrite(os.path.join(out_scene_dir, '{}_pred_fine.png'.format(file_id)), fine_pred_rgb)
                fine_pred_depth = ret['outputs_fine']['depth'].detach().cpu()
                imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_fine.png'.format(file_id)),
                                (fine_pred_depth.numpy().squeeze() * 1000.).astype(np.uint16))
                fine_pred_depth_colored = colorize_np(fine_pred_depth,
                                                      range=tuple(data['depth_range'].squeeze().cpu().numpy()))
                imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_vis_fine.png'.format(file_id)),
                                (255 * fine_pred_depth_colored).astype(np.uint8))
                fine_acc_map = torch.sum(ret['outputs_fine']['weights'], dim=-1).detach().cpu()
                fine_acc_map_colored = (colorize_np(fine_acc_map, range=(0., 1.)) * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(out_scene_dir, '{}_acc_map_fine.png'.format(file_id)),
                                fine_acc_map_colored)
            else:
                fine_ssim = fine_lpips = fine_psnr = 0.

            sum_fine_psnr += fine_psnr
            running_mean_fine_psnr = sum_fine_psnr / (i + 1)
            sum_fine_lpips += fine_lpips
            running_mean_fine_lpips = sum_fine_lpips / (i + 1)
            sum_fine_ssim += fine_ssim
            running_mean_fine_ssim = sum_fine_ssim / (i + 1)

            print("==================\n"
                  "{}, curr_id: {} \n"
                  "current coarse psnr: {:03f}, current fine psnr: {:03f} \n"
                  "running mean coarse psnr: {:03f}, running mean fine psnr: {:03f} \n"
                  "current coarse ssim: {:03f}, current fine ssim: {:03f} \n"
                  "running mean coarse ssim: {:03f}, running mean fine ssim: {:03f} \n" 
                  "current coarse lpips: {:03f}, current fine lpips: {:03f} \n"
                  "running mean coarse lpips: {:03f}, running mean fine lpips: {:03f} \n"
                  "===================\n"
                  .format(scene_name, file_id,
                          coarse_psnr, fine_psnr,
                          running_mean_coarse_psnr, running_mean_fine_psnr,
                          coarse_ssim, fine_ssim,
                          running_mean_coarse_ssim, running_mean_fine_ssim,
                          coarse_lpips, fine_lpips,
                          running_mean_coarse_lpips, running_mean_fine_lpips
                          ))

            results_dict[scene_name][file_id] = {'coarse_psnr': coarse_psnr,
                                                 'fine_psnr': fine_psnr,
                                                 'coarse_ssim': coarse_ssim,
                                                 'fine_ssim': fine_ssim,
                                                 'coarse_lpips': coarse_lpips,
                                                 'fine_lpips': fine_lpips,
                                                 }

    mean_coarse_psnr = sum_coarse_psnr / total_num
    mean_fine_psnr = sum_fine_psnr / total_num
    mean_coarse_lpips = sum_coarse_lpips / total_num
    mean_fine_lpips = sum_fine_lpips / total_num
    mean_coarse_ssim = sum_coarse_ssim / total_num
    mean_fine_ssim = sum_fine_ssim / total_num

    print('------{}-------\n'
          'final coarse psnr: {}, final fine psnr: {}\n'
          'fine coarse ssim: {}, final fine ssim: {} \n'
          'final coarse lpips: {}, fine fine lpips: {} \n'
          .format(scene_name, mean_coarse_psnr, mean_fine_psnr,
                  mean_coarse_ssim, mean_fine_ssim,
                  mean_coarse_lpips, mean_fine_lpips,
                  ))

    results_dict[scene_name]['coarse_mean_psnr'] = mean_coarse_psnr
    results_dict[scene_name]['fine_mean_psnr'] = mean_fine_psnr
    results_dict[scene_name]['coarse_mean_ssim'] = mean_coarse_ssim
    results_dict[scene_name]['fine_mean_ssim'] = mean_fine_ssim
    results_dict[scene_name]['coarse_mean_lpips'] = mean_coarse_lpips
    results_dict[scene_name]['fine_mean_lpips'] = mean_fine_lpips

    f = open("{}/psnr_{}_{}.txt".format(extra_out_dir, save_prefix, model.start_step), "w")
    f.write(str(results_dict))
    f.close()



```

# prompts/papers/IBRNet/eval/render_llff_video.py

``` py
# Copyright 2020 Google LLC
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


from torch.utils.data import Dataset
import sys
sys.path.append('../')
from torch.utils.data import DataLoader
import imageio
from config import config_parser
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.render_image import render_single_image
from ibrnet.model import IBRNetModel
from utils import *
from ibrnet.projection import Projector
from ibrnet.data_loaders import get_nearest_pose_ids
from ibrnet.data_loaders.llff_data_utils import load_llff_data, batch_parse_llff_poses
import time


class LLFFRenderDataset(Dataset):
    def __init__(self, args,
                 scenes='fern',  # 'fern', 'flower', 'fortress', 'horns', 'leaves', 'orchids', 'room', 'trex'
                 **kwargs):

        self.folder_path = os.path.join(args.rootdir, 'data/nerf_llff_data/')
        self.num_source_views = args.num_source_views

        print("loading {} for rendering".format(scenes))

        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []
        self.h = []
        self.w = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []

        for i, scene in enumerate(scenes):
            scene_path = os.path.join(self.folder_path, scene)
            _, poses, bds, render_poses, i_test, rgb_files = load_llff_data(scene_path, load_imgs=False, factor=4)
            near_depth = np.min(bds)
            far_depth = np.max(bds)
            intrinsics, c2w_mats = batch_parse_llff_poses(poses)
            h, w = poses[0][:2, -1]
            render_intrinsics, render_c2w_mats = batch_parse_llff_poses(render_poses)

            i_test = [i_test]
            i_val = i_test
            i_train = np.array([i for i in np.arange(len(rgb_files)) if
                                (i not in i_test and i not in i_val)])

            self.train_intrinsics.append(intrinsics[i_train])
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            num_render = len(render_intrinsics)
            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in render_intrinsics])
            self.render_poses.extend([c2w_mat for c2w_mat in render_c2w_mats])
            self.render_depth_range.extend([[near_depth, far_depth]]*num_render)
            self.render_train_set_ids.extend([i]*num_render)
            self.h.extend([int(h)]*num_render)
            self.w.extend([int(w)]*num_render)

    def __len__(self):
        return len(self.render_poses)

    def __getitem__(self, idx):
        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        h, w = self.h[idx], self.w[idx]
        camera = np.concatenate(([h, w], intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        id_render = -1
        nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                train_poses,
                                                self.num_source_views,
                                                tar_id=id_render,
                                                angular_dist_method='dist')

        src_rgbs = []
        src_cameras = []
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(),
                                         train_pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)
        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.5])

        return {'camera': torch.from_numpy(camera),
                'rgb_path': '',
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]),
                'src_cameras': torch.from_numpy(src_cameras),
                'depth_range': depth_range
                }


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    args.distributed = False

    # Create ibrnet model
    model = IBRNetModel(args, load_scheduler=False, load_opt=False)
    eval_dataset_name = args.eval_dataset
    extra_out_dir = '{}/{}'.format(eval_dataset_name, args.expname)
    print('saving results to {}...'.format(extra_out_dir))
    os.makedirs(extra_out_dir, exist_ok=True)

    projector = Projector(device='cuda:0')

    assert len(args.eval_scenes) == 1, "only accept single scene"
    scene_name = args.eval_scenes[0]
    out_scene_dir = os.path.join(extra_out_dir, '{}_{:06d}'.format(scene_name, model.start_step), 'videos')
    print('saving results to {}'.format(out_scene_dir))

    os.makedirs(out_scene_dir, exist_ok=True)

    test_dataset = LLFFRenderDataset(args, scenes=args.eval_scenes)
    save_prefix = scene_name
    test_loader = DataLoader(test_dataset, batch_size=1)
    total_num = len(test_loader)
    out_frames = []
    crop_ratio = 0.075

    for i, data in enumerate(test_loader):
        start = time.time()
        src_rgbs = data['src_rgbs'][0].cpu().numpy()
        averaged_img = (np.mean(src_rgbs, axis=0) * 255.).astype(np.uint8)
        imageio.imwrite(os.path.join(out_scene_dir, '{}_average.png'.format(i)), averaged_img)

        model.switch_to_eval()
        with torch.no_grad():
            ray_sampler = RaySamplerSingleImage(data, device='cuda:0')
            ray_batch = ray_sampler.get_all()
            featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))

            ret = render_single_image(ray_sampler=ray_sampler,
                                      ray_batch=ray_batch,
                                      model=model,
                                      projector=projector,
                                      chunk_size=args.chunk_size,
                                      det=True,
                                      N_samples=args.N_samples,
                                      inv_uniform=args.inv_uniform,
                                      N_importance=args.N_importance,
                                      white_bkgd=args.white_bkgd,
                                      featmaps=featmaps)
            torch.cuda.empty_cache()

        coarse_pred_rgb = ret['outputs_coarse']['rgb'].detach().cpu()
        coarse_pred_rgb = (255 * np.clip(coarse_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
        imageio.imwrite(os.path.join(out_scene_dir, '{}_pred_coarse.png'.format(i)), coarse_pred_rgb)

        coarse_pred_depth = ret['outputs_coarse']['depth'].detach().cpu()
        imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_coarse.png'.format(i)),
                        (coarse_pred_depth.numpy().squeeze() * 1000.).astype(np.uint16))
        coarse_pred_depth_colored = colorize_np(coarse_pred_depth,
                                                range=tuple(data['depth_range'].squeeze().numpy()))
        imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_vis_coarse.png'.format(i)),
                        (255 * coarse_pred_depth_colored).astype(np.uint8))

        coarse_acc_map = torch.sum(ret['outputs_coarse']['weights'].detach().cpu(), dim=-1)
        coarse_acc_map_colored = (colorize_np(coarse_acc_map, range=(0., 1.)) * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(out_scene_dir, '{}_acc_map_coarse.png'.format(i)),
                        coarse_acc_map_colored)

        if ret['outputs_fine'] is not None:
            fine_pred_rgb = ret['outputs_fine']['rgb'].detach().cpu()
            fine_pred_rgb = (255 * np.clip(fine_pred_rgb.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
            imageio.imwrite(os.path.join(out_scene_dir, '{}_pred_fine.png'.format(i)), fine_pred_rgb)
            fine_pred_depth = ret['outputs_fine']['depth'].detach().cpu()
            imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_fine.png'.format(i)),
                            (fine_pred_depth.numpy().squeeze() * 1000.).astype(np.uint16))
            fine_pred_depth_colored = colorize_np(fine_pred_depth,
                                                  range=tuple(data['depth_range'].squeeze().cpu().numpy()))
            imageio.imwrite(os.path.join(out_scene_dir, '{}_depth_vis_fine.png'.format(i)),
                            (255 * fine_pred_depth_colored).astype(np.uint8))
            fine_acc_map = torch.sum(ret['outputs_fine']['weights'].detach().cpu(), dim=-1)
            fine_acc_map_colored = (colorize_np(fine_acc_map, range=(0., 1.)) * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(out_scene_dir, '{}_acc_map_fine.png'.format(i)),
                            fine_acc_map_colored)
        else:
            fine_pred_rgb = None

        out_frame = fine_pred_rgb if fine_pred_rgb is not None else coarse_pred_rgb
        h, w = averaged_img.shape[:2]
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
        # crop out image boundaries
        out_frame = out_frame[crop_h:h - crop_h, crop_w:w - crop_w, :]
        out_frames.append(out_frame)

        print('frame {} completed, {}'.format(i, time.time() - start))

    imageio.mimwrite(os.path.join(extra_out_dir, '{}.mp4'.format(scene_name)), out_frames, fps=30, quality=8)


```

# prompts/papers/IBRNet/ibrnet/criterion.py

``` py
# Copyright 2020 Google LLC
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

import torch.nn as nn
from utils import img2mse


class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, ray_batch, scalars_to_log):
        '''
        training criterion
        '''
        pred_rgb = outputs['rgb']
        pred_mask = outputs['mask'].float()
        gt_rgb = ray_batch['rgb']

        loss = img2mse(pred_rgb, gt_rgb, pred_mask)

        return loss, scalars_to_log



```

# prompts/papers/IBRNet/ibrnet/data_loaders/__init__.py

``` py
# Copyright 2020 Google LLC
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


from .google_scanned_objects import *
from .realestate import *
from .deepvoxels import *
from .realestate import *
from .llff import *
from .llff_test import *
from .ibrnet_collected import *
from .realestate import *
from .spaces_dataset import *
from .nerf_synthetic import *

dataset_dict = {
    'spaces': SpacesFreeDataset,
    'google_scanned': GoogleScannedDataset,
    'realestate': RealEstateDataset,
    'deepvoxels': DeepVoxelsDataset,
    'nerf_synthetic': NerfSyntheticDataset,
    'llff': LLFFDataset,
    'ibrnet_collected': IBRNetCollectedDataset,
    'llff_test': LLFFTestDataset
}

```

# prompts/papers/IBRNet/ibrnet/data_loaders/colmap_read_model.py

``` py
# Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Johannes L. Schoenberger (jsch at inf.ethz.ch)

import os
import sys
import collections
import numpy as np
import struct


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


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
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) \
                         for camera_model in CAMERA_MODELS])


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


def read_cameras_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
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
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras


def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for camera_line_index in range(num_cameras):
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


def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
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


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
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


def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb,
                                               error=error, image_ids=image_ids,
                                               point2D_idxs=point2D_idxs)
    return points3D


def read_points3d_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb,
                error=error, image_ids=image_ids,
                point2D_idxs=point2D_idxs)
    return points3D


def read_model(path, ext):
    if ext == ".txt":
        cameras = read_cameras_text(os.path.join(path, "cameras" + ext))
        images = read_images_text(os.path.join(path, "images" + ext))
        points3D = read_points3D_text(os.path.join(path, "points3D") + ext)
    else:
        cameras = read_cameras_binary(os.path.join(path, "cameras" + ext))
        images = read_images_binary(os.path.join(path, "images" + ext))
        points3D = read_points3d_binary(os.path.join(path, "points3D") + ext)
    return cameras, images, points3D


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


def main():
    if len(sys.argv) != 3:
        print("Usage: python read_model.py path/to/model/folder [.txt,.bin]")
        return

    cameras, images, points3D = read_model(path=sys.argv[1], ext=sys.argv[2])

    print("num_cameras:", len(cameras))
    print("num_images:", len(images))
    print("num_points3D:", len(points3D))


if __name__ == "__main__":
    main()


```

# prompts/papers/IBRNet/ibrnet/data_loaders/create_training_dataset.py

``` py
# Copyright 2020 Google LLC
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


import numpy as np
from . import dataset_dict
from torch.utils.data import Dataset, Sampler
from torch.utils.data import DistributedSampler, WeightedRandomSampler
from typing import Optional
from operator import itemgetter
import torch


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self):
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


def create_training_dataset(args):
    # parse args.train_dataset, "+" indicates that multiple datasets are used, for example "ibrnet_collect+llff+spaces"
    # otherwise only one dataset is used
    # args.dataset_weights should be a list representing the resampling rate for each dataset, and should sum up to 1

    print('training dataset: {}'.format(args.train_dataset))
    mode = 'train'
    if '+' not in args.train_dataset:
        train_dataset = dataset_dict[args.train_dataset](args, mode,
                                                         scenes=args.train_scenes
                                                         )
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    else:
        train_dataset_names = args.train_dataset.split('+')
        weights = args.dataset_weights
        assert len(train_dataset_names) == len(weights)
        assert np.abs(np.sum(weights) - 1.) < 1e-6
        print('weights:{}'.format(weights))
        train_datasets = []
        train_weights_samples = []
        for training_dataset_name, weight in zip(train_dataset_names, weights):
            train_dataset = dataset_dict[training_dataset_name](args, mode,
                                                                scenes=args.train_scenes,
                                                                )
            train_datasets.append(train_dataset)
            num_samples = len(train_dataset)
            weight_each_sample = weight / num_samples
            train_weights_samples.extend([weight_each_sample]*num_samples)

        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        train_weights = torch.from_numpy(np.array(train_weights_samples))
        sampler = WeightedRandomSampler(train_weights, len(train_weights))
        train_sampler = DistributedSamplerWrapper(sampler) if args.distributed else sampler

    return train_dataset, train_sampler





```

# prompts/papers/IBRNet/ibrnet/data_loaders/data_utils.py

``` py
# Copyright 2020 Google LLC
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

import numpy as np
import math
from PIL import Image
import torchvision.transforms as transforms
import torch
from scipy.spatial.transform import Rotation as R
import cv2

rng = np.random.RandomState(234)
_EPS = np.finfo(float).eps * 4.0
TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision


def vector_norm(data, axis=None, out=None):
    """Return length, i.e. eucledian norm, of ndarray along axis.
    """
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    else:
        data *= data
        np.sum(data, axis=axis, out=out)
        np.sqrt(out, out)


def quaternion_about_axis(angle, axis):
    """Return quaternion for rotation about axis.
    """
    quaternion = np.zeros((4, ), dtype=np.float64)
    quaternion[:3] = axis[:3]
    qlen = vector_norm(quaternion)
    if qlen > _EPS:
        quaternion *= math.sin(angle/2.0) / qlen
    quaternion[3] = math.cos(angle/2.0)
    return quaternion


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.
    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)


def rectify_inplane_rotation(src_pose, tar_pose, src_img, th=40):
    relative = np.linalg.inv(tar_pose).dot(src_pose)
    relative_rot = relative[:3, :3]
    r = R.from_matrix(relative_rot)
    euler = r.as_euler('zxy', degrees=True)
    euler_z = euler[0]
    if np.abs(euler_z) < th:
        return src_pose, src_img

    R_rectify = R.from_euler('z', -euler_z, degrees=True).as_matrix()
    src_R_rectified = src_pose[:3, :3].dot(R_rectify)
    out_pose = np.eye(4)
    out_pose[:3, :3] = src_R_rectified
    out_pose[:3, 3:4] = src_pose[:3, 3:4]
    h, w = src_img.shape[:2]
    center = ((w - 1.) / 2., (h - 1.) / 2.)
    M = cv2.getRotationMatrix2D(center, -euler_z, 1)
    src_img = np.clip((255*src_img).astype(np.uint8), a_max=255, a_min=0)
    rotated = cv2.warpAffine(src_img, M, (w, h), borderValue=(255, 255, 255), flags=cv2.INTER_LANCZOS4)
    rotated = rotated.astype(np.float32) / 255.
    return out_pose, rotated


def random_crop(rgb, camera, src_rgbs, src_cameras, size=(400, 600), center=None):
    h, w = rgb.shape[:2]
    out_h, out_w = size[0], size[1]
    if out_w >= w or out_h >= h:
        return rgb, camera, src_rgbs, src_cameras

    if center is not None:
        center_h, center_w = center
    else:
        center_h = np.random.randint(low=out_h // 2 + 1, high=h - out_h // 2 - 1)
        center_w = np.random.randint(low=out_w // 2 + 1, high=w - out_w // 2 - 1)

    rgb_out = rgb[center_h - out_h // 2:center_h + out_h // 2, center_w - out_w // 2:center_w + out_w // 2, :]
    src_rgbs = np.array(src_rgbs)
    src_rgbs = src_rgbs[:, center_h - out_h // 2:center_h + out_h // 2,
               center_w - out_w // 2:center_w + out_w // 2, :]
    camera[0] = out_h
    camera[1] = out_w
    camera[4] -= center_w - out_w // 2
    camera[8] -= center_h - out_h // 2
    src_cameras[:, 4] -= center_w - out_w // 2
    src_cameras[:, 8] -= center_h - out_h // 2
    src_cameras[:, 0] = out_h
    src_cameras[:, 1] = out_w
    return rgb_out, camera, src_rgbs, src_cameras


def random_flip(rgb, camera, src_rgbs, src_cameras):
    h, w = rgb.shape[:2]
    h_r, w_r = src_rgbs.shape[1:3]
    rgb_out = np.flip(rgb, axis=1).copy()
    src_rgbs = np.flip(src_rgbs, axis=-2).copy()
    camera[2] *= -1
    camera[4] = w - 1. - camera[4]
    src_cameras[:, 2] *= -1
    src_cameras[:, 4] = w_r - 1. - src_cameras[:, 4]
    return rgb_out, camera, src_rgbs, src_cameras


def get_color_jitter_params(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
    color_jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    transform = transforms.ColorJitter.get_params(color_jitter.brightness,
                                                  color_jitter.contrast,
                                                  color_jitter.saturation,
                                                  color_jitter.hue)
    return transform


def color_jitter(img, transform):
    '''
    Args:
        img: np.float32 [h, w, 3]
        transform:
    Returns: transformed np.float32
    '''
    img = Image.fromarray((255.*img).astype(np.uint8))
    img_trans = transform(img)
    img_trans = np.array(img_trans).astype(np.float32) / 255.
    return img_trans


def color_jitter_all_rgbs(rgb, ref_rgbs, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
    transform = get_color_jitter_params(brightness, contrast, saturation, hue)
    rgb_trans = color_jitter(rgb, transform)
    ref_rgbs_trans = []
    for ref_rgb in ref_rgbs:
        ref_rgbs_trans.append(color_jitter(ref_rgb, transform))

    ref_rgbs_trans = np.array(ref_rgbs_trans)
    return rgb_trans, ref_rgbs_trans


def deepvoxels_parse_intrinsics(filepath, trgt_sidelength, invert_y=False):
    # Get camera intrinsics
    with open(filepath, 'r') as file:
        f, cx, cy = list(map(float, file.readline().split()))[:3]
        grid_barycenter = torch.Tensor(list(map(float, file.readline().split())))
        near_plane = float(file.readline())
        scale = float(file.readline())
        height, width = map(float, file.readline().split())

        try:
            world2cam_poses = int(file.readline())
        except ValueError:
            world2cam_poses = None

    if world2cam_poses is None:
        world2cam_poses = False

    world2cam_poses = bool(world2cam_poses)

    cx = cx / width * trgt_sidelength
    cy = cy / height * trgt_sidelength
    f = trgt_sidelength / height * f

    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    # Build the intrinsic matrices
    full_intrinsic = np.array([[fx, 0., cx, 0.],
                               [0., fy, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])

    return full_intrinsic, grid_barycenter, scale, near_plane, world2cam_poses


def angular_dist_between_2_vectors(vec1, vec2):
    vec1_unit = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + TINY_NUMBER)
    vec2_unit = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + TINY_NUMBER)
    angular_dists = np.arccos(np.clip(np.sum(vec1_unit*vec2_unit, axis=-1), -1.0, 1.0))
    return angular_dists


def batched_angular_dist_rot_matrix(R1, R2):
    '''
    calculate the angular distance between two rotation matrices (batched)
    :param R1: the first rotation matrix [N, 3, 3]
    :param R2: the second rotation matrix [N, 3, 3]
    :return: angular distance in radiance [N, ]
    '''
    assert R1.shape[-1] == 3 and R2.shape[-1] == 3 and R1.shape[-2] == 3 and R2.shape[-2] == 3
    return np.arccos(np.clip((np.trace(np.matmul(R2.transpose(0, 2, 1), R1), axis1=1, axis2=2) - 1) / 2.,
                             a_min=-1 + TINY_NUMBER, a_max=1 - TINY_NUMBER))


def get_nearest_pose_ids(tar_pose, ref_poses, num_select, tar_id=-1, angular_dist_method='vector',
                         scene_center=(0, 0, 0)):
    '''
    Args:
        tar_pose: target pose [3, 3]
        ref_poses: reference poses [N, 3, 3]
        num_select: the number of nearest views to select
    Returns: the selected indices
    '''
    num_cams = len(ref_poses)
    num_select = min(num_select, num_cams-1)
    batched_tar_pose = tar_pose[None, ...].repeat(num_cams, 0)

    if angular_dist_method == 'matrix':
        dists = batched_angular_dist_rot_matrix(batched_tar_pose[:, :3, :3], ref_poses[:, :3, :3])
    elif angular_dist_method == 'vector':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        scene_center = np.array(scene_center)[None, ...]
        tar_vectors = tar_cam_locs - scene_center
        ref_vectors = ref_cam_locs - scene_center
        dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
    elif angular_dist_method == 'dist':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)
    else:
        raise Exception('unknown angular distance calculation method!')

    if tar_id >= 0:
        assert tar_id < num_cams
        dists[tar_id] = 1e3  # make sure not to select the target id itself

    sorted_ids = np.argsort(dists)
    selected_ids = sorted_ids[:num_select]
    # print(angular_dists[selected_ids] * 180 / np.pi)
    return selected_ids


```

# prompts/papers/IBRNet/ibrnet/data_loaders/data_verifier.py

``` py
# Copyright 2020 Google LLC
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


'''
This file verifies if the camera poses are correct by looking at the epipolar geometry.
Given a pair of images and their relative pose, we sample a bunch of discriminative points in the first image,
we draw its corresponding epipolar line in the other image. If the camera pose is correct,
the epipolar line should pass the ground truth correspondence location.
'''


import cv2
import numpy as np


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def two_view_geometry(intrinsics1, extrinsics1, intrinsics2, extrinsics2):
    '''
    :param intrinsics1: 4 by 4 matrix
    :param extrinsics1: 4 by 4 W2C matrix
    :param intrinsics2: 4 by 4 matrix
    :param extrinsics2: 4 by 4 W2C matrix
    :return:
    '''
    relative_pose = extrinsics2.dot(np.linalg.inv(extrinsics1))
    R = relative_pose[:3, :3]
    T = relative_pose[:3, 3]
    tx = skew(T)
    E = np.dot(tx, R)
    F = np.linalg.inv(intrinsics2[:3, :3]).T.dot(E).dot(np.linalg.inv(intrinsics1[:3, :3]))

    return E, F, relative_pose


def drawpointslines(img1, img2, lines1, pts2, color):
    '''
    draw corresponding epilines on img1 for the points in img2
    '''

    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt2, cl in zip(lines1, pts2, color):
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        cl = tuple(cl.tolist())
        img1 = cv2.line(img1, (x0,y0), (x1,y1), cl, 1)
        img2 = cv2.circle(img2, tuple(pt2), 5, cl, -1)
    return img1, img2


def epipolar(coord1, F, img1, img2):
    # compute epipole
    pts1 = coord1.astype(int).T
    color = np.random.randint(0, high=255, size=(len(pts1), 3))
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    if lines2 is None:
        return None
    lines2 = lines2.reshape(-1,3)

    img3, img4 = drawpointslines(img2,img1,lines2,pts1,color)
    ## print(img3.shape)
    ## print(np.concatenate((img4, img3)).shape)
    ## cv2.imwrite('vis.png', np.concatenate((img4, img3), axis=1))
    h_max = max(img3.shape[0], img4.shape[0])
    w_max = max(img3.shape[1], img4.shape[1])
    out = np.ones((h_max, w_max*2, 3))
    out[:img4.shape[0], :img4.shape[1], :] = img4
    out[:img3.shape[0], w_max:w_max+img3.shape[1], :] = img3

    # return np.concatenate((img4, img3), axis=1)
    return out


def verify_data(img1, img2, intrinsics1, extrinsics1, intrinsics2, extrinsics2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    E, F, relative_pose = two_view_geometry(intrinsics1, extrinsics1,
                                            intrinsics2, extrinsics2)

    # sift = cv2.xfeatures2d.SIFT_create(nfeatures=20)
    # kp1 = sift.detect(img1, mask=None)
    # coord1 = np.array([[kp.pt[0], kp.pt[1]] for kp in kp1]).T

    # Initiate ORB detector
    orb = cv2.ORB_create()
    # find the keypoints with ORB
    kp1 = orb.detect(img1, None)
    coord1 = np.array([[kp.pt[0], kp.pt[1]] for kp in kp1[:20]]).T
    return epipolar(coord1, F, img1, img2)


def calc_angles(c2w_1, c2w_2):
    c1 = c2w_1[:3, 3:4]
    c2 = c2w_2[:3, 3:4]

    c1 = c1 / np.linalg.norm(c1)
    c2 = c2 / np.linalg.norm(c2)
    return np.rad2deg(np.arccos(np.dot(c1.T, c2)))


if __name__ == '__main__':
    import sys
    sys.path.append('../../')
    import os
    from ibrnet.data_loaders.google_scanned_objects import GoogleScannedDataset
    from config import config_parser

    parser = config_parser()
    args = parser.parse_args()
    dataset = GoogleScannedDataset(args, mode='train')
    out_dir = 'data_verify'
    print('saving output to {}...'.format(out_dir))
    os.makedirs(out_dir, exist_ok=True)

    for k, data in enumerate(dataset):
        rgb = data['rgb'].cpu().numpy()
        camera = data['camera'].cpu().numpy()
        src_rgbs = data['src_rgbs'].cpu().numpy()
        src_cameras = data['src_cameras'].cpu().numpy()
        i = np.random.randint(low=0, high=len(src_rgbs))
        rgb_i = src_rgbs[i]
        cameras_i = src_cameras[i]
        intrinsics1 = camera[2:18].reshape(4, 4)
        intrinsics2 = cameras_i[2:18].reshape(4, 4)
        extrinsics1 = np.linalg.inv(camera[-16:].reshape(4, 4))
        extrinsics2 = np.linalg.inv(cameras_i[-16:].reshape(4, 4))

        im = verify_data(np.uint8(rgb*255.), np.uint8(rgb_i*255.),
                         intrinsics1, extrinsics1,
                         intrinsics2, extrinsics2)
        if im is not None:
            cv2.imwrite(os.path.join(out_dir, '{:03d}.png'.format(k)), im)




```

# prompts/papers/IBRNet/ibrnet/data_loaders/deepvoxels.py

``` py
# Copyright 2020 Google LLC
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

import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import glob
import sys
sys.path.append('../')
from .data_utils import deepvoxels_parse_intrinsics, get_nearest_pose_ids, rectify_inplane_rotation


class DeepVoxelsDataset(Dataset):
    def __init__(self, args, subset,
                 scenes='vase',  # string or list
                 **kwargs):

        self.folder_path = os.path.join(args.rootdir, 'data/deepvoxels/')
        self.rectify_inplane_rotation = args.rectify_inplane_rotation
        self.subset = subset  # train / test / validation
        self.num_source_views = args.num_source_views
        self.testskip = args.testskip

        if isinstance(scenes, str):
            scenes = [scenes]

        self.scenes = scenes
        self.all_rgb_files = []
        self.all_depth_files = []
        self.all_pose_files = []
        self.all_intrinsics_files = []

        for scene in scenes:
            self.scene_path = os.path.join(self.folder_path, subset, scene)

            rgb_files = [os.path.join(self.scene_path, 'rgb', f)
                         for f in sorted(os.listdir(os.path.join(self.scene_path, 'rgb')))]
            if self.subset != 'train':
                rgb_files = rgb_files[::self.testskip]
            depth_files = [f.replace('rgb', 'depth') for f in rgb_files]
            pose_files = [f.replace('rgb', 'pose').replace('png', 'txt') for f in rgb_files]
            intrinsics_file = os.path.join(self.scene_path, 'intrinsics.txt')
            self.all_rgb_files.extend(rgb_files)
            self.all_depth_files.extend(depth_files)
            self.all_pose_files.extend(pose_files)
            self.all_intrinsics_files.extend([intrinsics_file]*len(rgb_files))

    def __len__(self):
        return len(self.all_rgb_files)

    def __getitem__(self, idx):
        idx = idx % len(self.all_rgb_files)
        rgb_file = self.all_rgb_files[idx]
        pose_file = self.all_pose_files[idx]
        intrinsics_file = self.all_intrinsics_files[idx]
        intrinsics = deepvoxels_parse_intrinsics(intrinsics_file, 512)[0]

        train_rgb_files = sorted(glob.glob(os.path.join(self.scene_path.replace('/{}/'.format(self.subset),
                                                                                '/train/'), 'rgb', '*')))
        train_poses_files = [f.replace('rgb', 'pose').replace('png', 'txt') for f in train_rgb_files]
        train_poses = np.stack([np.loadtxt(file).reshape(4, 4) for file in train_poses_files], axis=0)

        if self.subset == 'train':
            id_render = train_poses_files.index(pose_file)
            subsample_factor = np.random.choice(np.arange(1, 5))
            num_source_views = np.random.randint(low=self.num_source_views-4, high=self.num_source_views+2)
        else:
            id_render = -1
            subsample_factor = 1
            num_source_views = self.num_source_views

        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.
        render_pose = np.loadtxt(pose_file).reshape(4, 4)

        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                train_poses,
                                                min(num_source_views*subsample_factor, 40),
                                                tar_id=id_render,
                                                angular_dist_method='vector')
        nearest_pose_ids = np.random.choice(nearest_pose_ids, num_source_views, replace=False)

        assert id_render not in nearest_pose_ids
        # occasionally include target image in the source views
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.subset == 'train':
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        src_rgbs = []
        src_cameras = []
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.
            train_pose = train_poses[id]
            if self.rectify_inplane_rotation:
                src_pose, src_rgb = rectify_inplane_rotation(train_pose, render_pose, src_rgb)

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), intrinsics.flatten(),
                                         train_pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        origin_depth = np.linalg.inv(render_pose.reshape(4, 4))[2, 3]

        if 'cube' in rgb_file:
            near_depth = origin_depth - 1.
            far_depth = origin_depth + 1
        else:
            near_depth = origin_depth - 0.8
            far_depth = origin_depth + 0.8

        depth_range = torch.tensor([near_depth, far_depth])

        return {'rgb': torch.from_numpy(rgb[..., :3]),
                'camera': torch.from_numpy(camera),
                'rgb_path': rgb_file,
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]),
                'src_cameras': torch.from_numpy(src_cameras),
                'depth_range': depth_range,
                'scene_path': self.scene_path
                }



```

# prompts/papers/IBRNet/ibrnet/data_loaders/google_scanned_objects.py

``` py
# Copyright 2020 Google LLC
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


import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import glob
import sys
sys.path.append('../')
from .data_utils import rectify_inplane_rotation, get_nearest_pose_ids


# only for training
class GoogleScannedDataset(Dataset):
    def __init__(self, args, mode, **kwargs):
        self.folder_path = os.path.join(args.rootdir, 'data/google_scanned_objects/')
        self.num_source_views = args.num_source_views
        self.rectify_inplane_rotation = args.rectify_inplane_rotation
        self.scene_path_list = glob.glob(os.path.join(self.folder_path, '*'))

        all_rgb_files = []
        all_pose_files = []
        all_intrinsics_files = []
        num_files = 250
        for i, scene_path in enumerate(self.scene_path_list):
            rgb_files = [os.path.join(scene_path, 'rgb', f)
                         for f in sorted(os.listdir(os.path.join(scene_path, 'rgb')))]
            pose_files = [f.replace('rgb', 'pose').replace('png', 'txt') for f in rgb_files]
            intrinsics_files = [f.replace('rgb', 'intrinsics').replace('png', 'txt') for f in rgb_files]

            if np.min([len(rgb_files), len(pose_files), len(intrinsics_files)]) \
                    < num_files:
                print(scene_path)
                continue

            all_rgb_files.append(rgb_files)
            all_pose_files.append(pose_files)
            all_intrinsics_files.append(intrinsics_files)

        index = np.arange(len(all_rgb_files))
        self.all_rgb_files = np.array(all_rgb_files)[index]
        self.all_pose_files = np.array(all_pose_files)[index]
        self.all_intrinsics_files = np.array(all_intrinsics_files)[index]

    def __len__(self):
        return len(self.all_rgb_files)

    def __getitem__(self, idx):
        rgb_files = self.all_rgb_files[idx]
        pose_files = self.all_pose_files[idx]
        intrinsics_files = self.all_intrinsics_files[idx]

        id_render = np.random.choice(np.arange(len(rgb_files)))
        train_poses = np.stack([np.loadtxt(file).reshape(4, 4) for file in pose_files], axis=0)
        render_pose = train_poses[id_render]
        subsample_factor = np.random.choice(np.arange(1, 6), p=[0.3, 0.25, 0.2, 0.2, 0.05])

        id_feat_pool = get_nearest_pose_ids(render_pose,
                                            train_poses,
                                            self.num_source_views*subsample_factor,
                                            tar_id=id_render,
                                            angular_dist_method='vector')
        id_feat = np.random.choice(id_feat_pool, self.num_source_views, replace=False)

        assert id_render not in id_feat
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]):
            id_feat[np.random.choice(len(id_feat))] = id_render

        rgb = imageio.imread(rgb_files[id_render]).astype(np.float32) / 255.

        intrinsics = np.loadtxt(intrinsics_files[id_render])
        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), intrinsics, render_pose.flatten())).astype(np.float32)

        # get depth range
        min_ratio = 0.1
        origin_depth = np.linalg.inv(render_pose)[2, 3]
        max_radius = 0.5 * np.sqrt(2) * 1.1
        near_depth = max(origin_depth - max_radius, min_ratio * origin_depth)
        far_depth = origin_depth + max_radius
        depth_range = torch.tensor([near_depth, far_depth])

        src_rgbs = []
        src_cameras = []
        for id in id_feat:
            src_rgb = imageio.imread(rgb_files[id]).astype(np.float32) / 255.
            pose = np.loadtxt(pose_files[id])
            if self.rectify_inplane_rotation:
                pose, src_rgb = rectify_inplane_rotation(pose.reshape(4, 4), render_pose, src_rgb)

            src_rgbs.append(src_rgb)
            intrinsics = np.loadtxt(intrinsics_files[id])
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), intrinsics, pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs)
        src_cameras = np.stack(src_cameras)

        return {'rgb': torch.from_numpy(rgb),
                'camera': torch.from_numpy(camera),
                'rgb_path': rgb_files[id_render],
                'src_rgbs': torch.from_numpy(src_rgbs),
                'src_cameras': torch.from_numpy(src_cameras),
                'depth_range': depth_range
                }



```

# prompts/papers/IBRNet/ibrnet/data_loaders/ibrnet_collected.py

``` py
# Copyright 2020 Google LLC
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

import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import glob
import sys
sys.path.append('../')
from .data_utils import rectify_inplane_rotation, random_crop, random_flip, get_nearest_pose_ids
from .llff_data_utils import load_llff_data, batch_parse_llff_poses


class IBRNetCollectedDataset(Dataset):
    def __init__(self, args, mode, random_crop=True, **kwargs):
        self.folder_path1 = os.path.join(args.rootdir, 'data/ibrnet_collected_1/')
        self.folder_path2 = os.path.join(args.rootdir, 'data/ibrnet_collected_2/')
        self.rectify_inplane_rotation = args.rectify_inplane_rotation
        self.mode = mode  # train / test / validation
        self.num_source_views = args.num_source_views
        self.random_crop = random_crop

        all_scenes = glob.glob(self.folder_path1 + '*') + glob.glob(self.folder_path2 + '*')

        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []

        for i, scene in enumerate(all_scenes):
            if 'ibrnet_collected_2' in scene:
                factor = 8
            else:
                factor = 2
            _, poses, bds, render_poses, i_test, rgb_files = load_llff_data(scene, load_imgs=False, factor=factor)
            near_depth = np.min(bds)
            far_depth = np.max(bds)
            intrinsics, c2w_mats = batch_parse_llff_poses(poses)
            if mode == 'train':
                i_train = np.array(np.arange(int(poses.shape[0])))
                i_render = i_train
            else:
                i_test = np.arange(poses.shape[0])[::args.llffhold]
                i_train = np.array([j for j in np.arange(int(poses.shape[0])) if
                                    (j not in i_test and j not in i_test)])
                i_render = i_test

            self.train_intrinsics.append(intrinsics[i_train])
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            num_render = len(i_render)
            self.render_rgb_files.extend(np.array(rgb_files)[i_render].tolist())
            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
            self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])
            self.render_depth_range.extend([[near_depth, far_depth]]*num_render)
            self.render_train_set_ids.extend([i]*num_render)

    def __len__(self):
        return len(self.render_rgb_files)

    def __getitem__(self, idx):
        rgb_file = self.render_rgb_files[idx]
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.
        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]
        mean_depth = np.mean(depth_range)
        world_center = (render_pose.dot(np.array([[0, 0, mean_depth, 1]]).T)).flatten()[:3]

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        if self.mode == 'train':
            id_render = train_rgb_files.index(rgb_file)
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
            num_select = self.num_source_views + np.random.randint(low=-2, high=3)
        else:
            id_render = -1
            subsample_factor = 1
            num_select = self.num_source_views


        nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                train_poses,
                                                min(self.num_source_views*subsample_factor, 22),
                                                tar_id=id_render,
                                                angular_dist_method='dist',
                                                scene_center=world_center)
        nearest_pose_ids = np.random.choice(nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False)

        assert id_render not in nearest_pose_ids
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == 'train':
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        src_rgbs = []
        src_cameras = []
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            if self.rectify_inplane_rotation:
                train_pose, src_rgb = rectify_inplane_rotation(train_pose, render_pose, src_rgb)

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(),
                                         train_pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        if self.mode == 'train' and self.random_crop:
            rgb, camera, src_rgbs, src_cameras = random_crop(rgb, camera, src_rgbs, src_cameras)

        if self.mode == 'train' and np.random.choice([0, 1], p=[0.5, 0.5]):
            rgb, camera, src_rgbs, src_cameras = random_flip(rgb, camera, src_rgbs, src_cameras)

        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.5])

        return {'rgb': torch.from_numpy(rgb[..., :3]),
                'camera': torch.from_numpy(camera),
                'rgb_path': rgb_file,
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]),
                'src_cameras': torch.from_numpy(src_cameras),
                'depth_range': depth_range,
                }


```

# prompts/papers/IBRNet/ibrnet/data_loaders/llff.py

``` py
# Copyright 2020 Google LLC
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

import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import sys
sys.path.append('../')
from .data_utils import random_crop, random_flip, get_nearest_pose_ids
from .llff_data_utils import load_llff_data, batch_parse_llff_poses


class LLFFDataset(Dataset):
    def __init__(self, args, mode, **kwargs):
        base_dir = os.path.join(args.rootdir, 'data/real_iconic_noface/')
        self.args = args
        self.mode = mode  # train / test / validation
        self.num_source_views = args.num_source_views
        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []

        scenes = os.listdir(base_dir)
        for i, scene in enumerate(scenes):
            scene_path = os.path.join(base_dir, scene)
            _, poses, bds, render_poses, i_test, rgb_files = load_llff_data(scene_path, load_imgs=False, factor=4)
            near_depth = np.min(bds)
            far_depth = np.max(bds)
            intrinsics, c2w_mats = batch_parse_llff_poses(poses)

            if mode == 'train':
                i_train = np.array(np.arange(int(poses.shape[0])))
                i_render = i_train
            else:
                i_test = np.arange(poses.shape[0])[::self.args.llffhold]
                i_train = np.array([j for j in np.arange(int(poses.shape[0])) if
                                    (j not in i_test and j not in i_test)])
                i_render = i_test

            self.train_intrinsics.append(intrinsics[i_train])
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            num_render = len(i_render)
            self.render_rgb_files.extend(np.array(rgb_files)[i_render].tolist())
            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
            self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])
            self.render_depth_range.extend([[near_depth, far_depth]]*num_render)
            self.render_train_set_ids.extend([i]*num_render)

    def __len__(self):
        return len(self.render_rgb_files)

    def __getitem__(self, idx):
        rgb_file = self.render_rgb_files[idx]
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.
        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        if self.mode == 'train':
            id_render = train_rgb_files.index(rgb_file)
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
            num_select = self.num_source_views + np.random.randint(low=-2, high=3)
        else:
            id_render = -1
            subsample_factor = 1
            num_select = self.num_source_views

        nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                train_poses,
                                                min(self.num_source_views*subsample_factor, 20),
                                                tar_id=id_render,
                                                angular_dist_method='dist')
        nearest_pose_ids = np.random.choice(nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False)

        assert id_render not in nearest_pose_ids
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == 'train':
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        src_rgbs = []
        src_cameras = []
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(),
                                              train_pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)
        if self.mode == 'train':
            crop_h = np.random.randint(low=250, high=750)
            crop_h = crop_h + 1 if crop_h % 2 == 1 else crop_h
            crop_w = int(400 * 600 / crop_h)
            crop_w = crop_w + 1 if crop_w % 2 == 1 else crop_w
            rgb, camera, src_rgbs, src_cameras = random_crop(rgb, camera, src_rgbs, src_cameras,
                                                             (crop_h, crop_w))

        if self.mode == 'train' and np.random.choice([0, 1]):
            rgb, camera, src_rgbs, src_cameras = random_flip(rgb, camera, src_rgbs, src_cameras)

        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.6])

        return {'rgb': torch.from_numpy(rgb[..., :3]),
                'camera': torch.from_numpy(camera),
                'rgb_path': rgb_file,
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]),
                'src_cameras': torch.from_numpy(src_cameras),
                'depth_range': depth_range,
                }



```

# prompts/papers/IBRNet/ibrnet/data_loaders/llff_data_utils.py

``` py
# Copyright 2020 Google LLC
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

import numpy as np
import os
import imageio
from .colmap_read_model import read_images_binary


########## Slightly modified version of LLFF data loading code
##########  see https://github.com/Fyusion/LLFF for original


def parse_llff_pose(pose):
    '''
    convert llff format pose to 4x4 matrix of intrinsics and extrinsics (opencv convention)
    Args:
        pose: matrix [3, 4]
    Returns: intrinsics [4, 4] and c2w [4, 4]
    '''
    h, w, f = pose[:3, -1]
    c2w = pose[:3, :4]
    c2w_4x4 = np.eye(4)
    c2w_4x4[:3] = c2w
    c2w_4x4[:, 1:3] *= -1
    intrinsics = np.array([[f, 0, w / 2., 0],
                           [0, f, h / 2., 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
    return intrinsics, c2w_4x4


def batch_parse_llff_poses(poses):
    all_intrinsics = []
    all_c2w_mats = []
    for pose in poses:
        intrinsics, c2w_mat = parse_llff_pose(pose)
        all_intrinsics.append(intrinsics)
        all_c2w_mats.append(c2w_mat)
    all_intrinsics = np.stack(all_intrinsics)
    all_c2w_mats = np.stack(all_c2w_mats)
    return all_intrinsics, all_c2w_mats


def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100. / r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape

    sfx = ''

    if factor is not None and factor != 1:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

    if poses.shape[-1] != len(imgfiles):
        imagesfile = os.path.join(basedir, 'sparse/0/images.bin')
        imdata = read_images_binary(imagesfile)
        imnames = [imdata[k].name[0:-4] for k in imdata]
        imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f[0:-4] in imnames]
        print('{}: Mismatch between imgs {} and poses {} !!!!'.format(basedir, len(imgfiles), poses.shape[-1]))
        return

    sh = imageio.imread(imgfiles[0]).shape
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    if not load_imgs:
        imgs = None
    else:
        imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
        imgs = np.stack(imgs, -1)
        print('Loaded image data', imgs.shape, poses[:, -1, 0])

    return poses, bds, imgs, imgfiles


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:, 4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses


def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    p34_to_44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)

    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad ** 2 - zh ** 2)
    new_poses = []

    for th in np.linspace(0., 2. * np.pi, 120):
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1)
    poses_reset = np.concatenate(
        [poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)

    return poses_reset, new_poses, bds


def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75,
                   spherify=False, path_zflat=False, load_imgs=True):
    out = _load_data(basedir, factor=factor, load_imgs=load_imgs)  # factor=8 downsamples original imgs by 8x
    if out is None:
        return
    else:
        poses, bds, imgs, imgfiles = out

    # print('Loaded', basedir, bds.min(), bds.max())

    # Correct rotation matrix ordering and move variable dim to axis 0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    if imgs is not None:
        imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
        images = imgs
        images = images.astype(np.float32)
    else:
        images = None
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)

    # Rescale if bd_factor is provided
    sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    if recenter:
        poses = recenter_poses(poses)

    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:

        c2w = poses_avg(poses)
        # print('recentered', c2w.shape)
        # print(c2w[:3, :4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
        dt = .75
        mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:, :3, 3]  # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
            #             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
            rads[2] = 0.
            N_rots = 1
            N_views /= 2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)

    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    # print('Data:')
    # print(poses.shape, images.shape, bds.shape)

    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)
    # print('HOLDOUT view is', i_test)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test, imgfiles


if __name__ == '__main__':
    scene_path = '/home/qianqianwang/datasets/nerf_llff_data/trex/'
    images, poses, bds, render_poses, i_test, img_files = load_llff_data(scene_path)
    print(bds)



```

# prompts/papers/IBRNet/ibrnet/data_loaders/llff_test.py

``` py
# Copyright 2020 Google LLC
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

import os
import numpy as np
import imageio
import torch
import sys
sys.path.append('../')
from torch.utils.data import Dataset
from .data_utils import random_crop, get_nearest_pose_ids
from .llff_data_utils import load_llff_data, batch_parse_llff_poses


class LLFFTestDataset(Dataset):
    def __init__(self, args, mode, scenes=(), random_crop=True, **kwargs):
        self.folder_path = os.path.join(args.rootdir, 'data/nerf_llff_data/')
        self.args = args
        self.mode = mode  # train / test / validation
        self.num_source_views = args.num_source_views
        self.random_crop = random_crop
        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []

        all_scenes = os.listdir(self.folder_path)
        if len(scenes) > 0:
            if isinstance(scenes, str):
                scenes = [scenes]
        else:
            scenes = all_scenes

        print("loading {} for {}".format(scenes, mode))
        for i, scene in enumerate(scenes):
            scene_path = os.path.join(self.folder_path, scene)
            _, poses, bds, render_poses, i_test, rgb_files = load_llff_data(scene_path, load_imgs=False, factor=4)
            near_depth = np.min(bds)
            far_depth = np.max(bds)
            intrinsics, c2w_mats = batch_parse_llff_poses(poses)

            i_test = np.arange(poses.shape[0])[::self.args.llffhold]
            i_train = np.array([j for j in np.arange(int(poses.shape[0])) if
                                (j not in i_test and j not in i_test)])

            if mode == 'train':
                i_render = i_train
            else:
                i_render = i_test

            self.train_intrinsics.append(intrinsics[i_train])
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            num_render = len(i_render)
            self.render_rgb_files.extend(np.array(rgb_files)[i_render].tolist())
            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
            self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])
            self.render_depth_range.extend([[near_depth, far_depth]]*num_render)
            self.render_train_set_ids.extend([i]*num_render)

    def __len__(self):
        return len(self.render_rgb_files) * 100000 if self.mode == 'train' else len(self.render_rgb_files)

    def __getitem__(self, idx):
        idx = idx % len(self.render_rgb_files)
        rgb_file = self.render_rgb_files[idx]
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.
        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        if self.mode == 'train':
            if rgb_file in train_rgb_files:
                id_render = train_rgb_files.index(rgb_file)
            else:
                id_render = -1
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
            num_select = self.num_source_views + np.random.randint(low=-2, high=2)
        else:
            id_render = -1
            subsample_factor = 1
            num_select = self.num_source_views

        nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                train_poses,
                                                min(self.num_source_views*subsample_factor, 28),
                                                tar_id=id_render,
                                                angular_dist_method='dist')
        nearest_pose_ids = np.random.choice(nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False)

        assert id_render not in nearest_pose_ids
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == 'train':
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        src_rgbs = []
        src_cameras = []
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(),
                                         train_pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)
        if self.mode == 'train' and self.random_crop:
            crop_h = np.random.randint(low=250, high=750)
            crop_h = crop_h + 1 if crop_h % 2 == 1 else crop_h
            crop_w = int(400 * 600 / crop_h)
            crop_w = crop_w + 1 if crop_w % 2 == 1 else crop_w
            rgb, camera, src_rgbs, src_cameras = random_crop(rgb, camera, src_rgbs, src_cameras,
                                                             (crop_h, crop_w))

        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.6])

        return {'rgb': torch.from_numpy(rgb[..., :3]),
                'camera': torch.from_numpy(camera),
                'rgb_path': rgb_file,
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]),
                'src_cameras': torch.from_numpy(src_cameras),
                'depth_range': depth_range,
                }



```

# prompts/papers/IBRNet/ibrnet/data_loaders/nerf_synthetic.py

``` py
# Copyright 2020 Google LLC
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

import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import sys
import json
sys.path.append('../')
from .data_utils import rectify_inplane_rotation, get_nearest_pose_ids


def read_cameras(pose_file):
    basedir = os.path.dirname(pose_file)
    with open(pose_file, 'r') as fp:
        meta = json.load(fp)

    camera_angle_x = float(meta['camera_angle_x'])
    rgb_files = []
    c2w_mats = []

    img = imageio.imread(os.path.join(basedir, meta['frames'][0]['file_path'] + '.png'))
    H, W = img.shape[:2]
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    intrinsics = get_intrinsics_from_hwf(H, W, focal)

    for i, frame in enumerate(meta['frames']):
        rgb_file = os.path.join(basedir, meta['frames'][i]['file_path'][2:] + '.png')
        rgb_files.append(rgb_file)
        c2w = np.array(frame['transform_matrix'])
        w2c_blender = np.linalg.inv(c2w)
        w2c_opencv = w2c_blender
        w2c_opencv[1:3] *= -1
        c2w_opencv = np.linalg.inv(w2c_opencv)
        c2w_mats.append(c2w_opencv)
    c2w_mats = np.array(c2w_mats)
    return rgb_files, np.array([intrinsics]*len(meta['frames'])), c2w_mats


def get_intrinsics_from_hwf(h, w, focal):
    return np.array([[focal, 0, 1.0*w/2, 0],
                     [0, focal, 1.0*h/2, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


class NerfSyntheticDataset(Dataset):
    def __init__(self, args, mode,
                 # scenes=('chair', 'drum', 'lego', 'hotdog', 'materials', 'mic', 'ship'),
                 scenes=(), **kwargs):
        self.folder_path = os.path.join(args.rootdir, 'data/nerf_synthetic/')
        self.rectify_inplane_rotation = args.rectify_inplane_rotation
        if mode == 'validation':
            mode = 'val'
        assert mode in ['train', 'val', 'test']
        self.mode = mode  # train / test / val
        self.num_source_views = args.num_source_views
        self.testskip = args.testskip

        all_scenes = ('chair', 'drums', 'lego', 'hotdog', 'materials', 'mic', 'ship')
        if len(scenes) > 0:
            if isinstance(scenes, str):
                scenes = [scenes]
        else:
            scenes = all_scenes

        print("loading {} for {}".format(scenes, mode))
        self.render_rgb_files = []
        self.render_poses = []
        self.render_intrinsics = []

        for scene in scenes:
            self.scene_path = os.path.join(self.folder_path, scene)
            pose_file = os.path.join(self.scene_path, 'transforms_{}.json'.format(mode))
            rgb_files, intrinsics, poses = read_cameras(pose_file)
            if self.mode != 'train':
                rgb_files = rgb_files[::self.testskip]
                intrinsics = intrinsics[::self.testskip]
                poses = poses[::self.testskip]
            self.render_rgb_files.extend(rgb_files)
            self.render_poses.extend(poses)
            self.render_intrinsics.extend(intrinsics)

    def __len__(self):
        return len(self.render_rgb_files)

    def __getitem__(self, idx):
        rgb_file = self.render_rgb_files[idx]
        render_pose = self.render_poses[idx]
        render_intrinsics = self.render_intrinsics[idx]

        train_pose_file = os.path.join('/'.join(rgb_file.split('/')[:-2]), 'transforms_train.json')
        train_rgb_files, train_intrinsics, train_poses = read_cameras(train_pose_file)

        if self.mode == 'train':
            id_render = int(os.path.basename(rgb_file)[:-4].split('_')[1])
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.3, 0.5, 0.2])
        else:
            id_render = -1
            subsample_factor = 1

        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.
        rgb = rgb[..., [-1]] * rgb[..., :3] + 1 - rgb[..., [-1]]
        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), render_intrinsics.flatten(),
                                 render_pose.flatten())).astype(np.float32)

        nearest_pose_ids = get_nearest_pose_ids(render_pose,
                                                train_poses,
                                                int(self.num_source_views*subsample_factor),
                                                tar_id=id_render,
                                                angular_dist_method='vector')
        nearest_pose_ids = np.random.choice(nearest_pose_ids, self.num_source_views, replace=False)

        assert id_render not in nearest_pose_ids
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == 'train':
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        src_rgbs = []
        src_cameras = []
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.
            src_rgb = src_rgb[..., [-1]] * src_rgb[..., :3] + 1 - src_rgb[..., [-1]]
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            if self.rectify_inplane_rotation:
                train_pose, src_rgb = rectify_inplane_rotation(train_pose, render_pose, src_rgb)

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(),
                                              train_pose.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        near_depth = 2.
        far_depth = 6.

        depth_range = torch.tensor([near_depth, far_depth])

        return {'rgb': torch.from_numpy(rgb[..., :3]),
                'camera': torch.from_numpy(camera),
                'rgb_path': rgb_file,
                'src_rgbs': torch.from_numpy(src_rgbs[..., :3]),
                'src_cameras': torch.from_numpy(src_cameras),
                'depth_range': depth_range,
                }



```

# prompts/papers/IBRNet/ibrnet/data_loaders/realestate.py

``` py
# Copyright 2020 Google LLC
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


import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import glob
import cv2


class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.intrinsics = np.array([[fx, 0, cx, 0],
                                    [0, fy, cy, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def unnormalize_intrinsics(intrinsics, h, w):
    intrinsics[0] *= w
    intrinsics[1] *= h
    return intrinsics


def parse_pose_file(file):
    f = open(file, 'r')
    cam_params = {}
    for i, line in enumerate(f):
        if i == 0:
            continue
        entry = [float(x) for x in line.split()]
        id = int(entry[0])
        cam_params[id] = Camera(entry)
    return cam_params


# only for training
class RealEstateDataset(Dataset):
    def __init__(self, args, mode, **kwargs):
        self.folder_path = os.path.join(args.rootdir, 'data/RealEstate10K-subset/')
        self.mode = mode  # train / test / validation
        self.num_source_views = args.num_source_views
        self.target_h, self.target_w = 450, 800
        assert mode in ['train', 'test']
        self.scene_path_list = glob.glob(os.path.join(self.folder_path, mode, 'frames', '*'))

        all_rgb_files = []
        all_timestamps = []
        for i, scene_path in enumerate(self.scene_path_list):
            rgb_files = [os.path.join(scene_path, f) for f in sorted(os.listdir(scene_path))]
            if len(rgb_files) < 10:
                print('omitting {}, too few images'.format(os.path.basename(scene_path)))
                continue
            timestamps = [int(os.path.basename(rgb_file).split('.')[0]) for rgb_file in rgb_files]
            sorted_ids = np.argsort(timestamps)
            all_rgb_files.append(np.array(rgb_files)[sorted_ids])
            all_timestamps.append(np.array(timestamps)[sorted_ids])

        index = np.arange(len(all_rgb_files))
        self.all_rgb_files = np.array(all_rgb_files)[index]
        self.all_timestamps = np.array(all_timestamps)[index]

    def __len__(self):
        return len(self.all_rgb_files)

    def __getitem__(self, idx):
        rgb_files = self.all_rgb_files[idx]
        timestamps = self.all_timestamps[idx]

        assert (timestamps == sorted(timestamps)).all()
        num_frames = len(rgb_files)
        window_size = 32
        shift = np.random.randint(low=-1, high=2)
        id_render = np.random.randint(low=4, high=num_frames-4-1)

        right_bound = min(id_render + window_size + shift, num_frames-1)
        left_bound = max(0, right_bound - 2 * window_size)
        candidate_ids = np.arange(left_bound, right_bound)
        # remove the query frame itself with high probability
        if np.random.choice([0, 1], p=[0.01, 0.99]):
            candidate_ids = candidate_ids[candidate_ids != id_render]

        id_feat = np.random.choice(candidate_ids, size=min(self.num_source_views, len(candidate_ids)),
                                   replace=False)

        rgb_file = rgb_files[id_render]
        rgb = imageio.imread(rgb_files[id_render])
        # resize the image to target size
        rgb = cv2.resize(rgb, dsize=(self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
        rgb = rgb.astype(np.float32) / 255.

        camera_file = os.path.dirname(rgb_file).replace('frames', 'cameras') + '.txt'
        cam_params = parse_pose_file(camera_file)
        cam_param = cam_params[timestamps[id_render]]

        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size),
                                 unnormalize_intrinsics(cam_param.intrinsics, self.target_h, self.target_w).flatten(),
                                 cam_param.c2w_mat.flatten())).astype(np.float32)

        # get depth range
        depth_range = torch.tensor([1., 100.])

        src_rgbs = []
        src_cameras = []
        for id in id_feat:
            src_rgb = imageio.imread(rgb_files[id])
            # resize the image to target size
            src_rgb = cv2.resize(src_rgb, dsize=(self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
            src_rgb = src_rgb.astype(np.float32) / 255.
            src_rgbs.append(src_rgb)

            img_size = src_rgb.shape[:2]
            cam_param = cam_params[timestamps[id]]
            src_camera = np.concatenate((list(img_size),
                                              unnormalize_intrinsics(cam_param.intrinsics,
                                                                     self.target_h, self.target_w).flatten(),
                                              cam_param.c2w_mat.flatten())).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs)
        src_cameras = np.stack(src_cameras)

        return {'rgb': torch.from_numpy(rgb),
                'camera': torch.from_numpy(camera),
                'rgb_path': rgb_files[id_render],
                'src_rgbs': torch.from_numpy(src_rgbs),
                'src_cameras': torch.from_numpy(src_cameras),
                'depth_range': depth_range,
                }



```

# prompts/papers/IBRNet/ibrnet/data_loaders/spaces_dataset.py

``` py
# Copyright 2020 Google LLC
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

# some code in this file is adapted from https://github.com/augmentedperception/spaces_dataset/

import sys
sys.path.append('../')
import os
import numpy as np
from PIL import Image
import imageio
import torch
from torch.utils.data import Dataset
from .data_utils import quaternion_about_axis, quaternion_matrix, random_crop, random_flip
import json


def view_obj2camera_rgb(view):
    image_path = view.image_path
    intrinsics = view.camera.intrinsics
    h_in_view, w_in_view = view.shape
    rgb = imageio.imread(image_path).astype(np.float32) / 255.
    h_img, w_img = rgb.shape[:2]
    if h_in_view != h_img or w_in_view != w_img:
        intrinsics[0] *= w_img / w_in_view
        intrinsics[1] *= h_img / h_in_view
    intrinsics_4x4 = np.eye(4)
    intrinsics_4x4[:3, :3] = intrinsics
    c2w = view.camera.w_f_c
    ref_camera = np.concatenate([list(rgb.shape[:2]), intrinsics_4x4.flatten(), c2w.flatten()])
    return ref_camera, rgb


def view_obj2camera_rgb_path(view):
    img_size = view.shape
    image_path = view.image_path
    intrinsics = view.camera.intrinsics
    intrinsics_4x4 = np.eye(4)
    intrinsics_4x4[:3, :3] = intrinsics
    c2w = view.camera.w_f_c
    return image_path, img_size, intrinsics_4x4, c2w


def sample_target_view_for_training(views, input_rig_id, input_ids):
    input_rig_views = views[input_rig_id]
    input_cam_positions = np.array([input_rig_views[i].camera.w_f_c[:3, 3] for i in input_ids])

    remaining_rig_ids = []
    remaining_cam_ids = []

    for i, rig in enumerate(views):
        for j, cam in enumerate(rig):
            if i == input_rig_id and j in input_ids:
                continue
            else:
                cam_loc = views[i][j].camera.w_f_c[:3, 3]
                # if i != input_rig_id:
                #     print(np.min(np.linalg.norm(input_cam_positions - cam_loc, axis=1)))
                if np.min(np.linalg.norm(input_cam_positions - cam_loc, axis=1)) < 0.15:
                    remaining_rig_ids.append(i)
                    remaining_cam_ids.append(j)

    selected_id = np.random.choice(len(remaining_rig_ids))
    selected_view = views[remaining_rig_ids[selected_id]][remaining_cam_ids[selected_id]]
    return selected_view


def get_all_views_in_scene(all_views):
    cameras = []
    rgbs = []
    for rig in all_views:
        for i in range(len(rig)):
            camera, rgb = view_obj2camera_rgb(rig[i])
            cameras.append(camera)
            rgbs.append(rgb)
    return cameras, rgbs


def get_all_views_in_scene_cam_path(all_views):
    c2w_mats = []
    intrinsicss = []
    rgb_paths = []
    img_sizes = []
    for rig in all_views:
        for i in range(len(rig)):
            image_path, img_size, intrinsics_4x4, c2w = view_obj2camera_rgb_path(rig[i])
            rgb_paths.append(image_path)
            intrinsicss.append(intrinsics_4x4)
            c2w_mats.append(c2w)
            img_sizes.append(img_size)
    return rgb_paths, img_sizes, intrinsicss, c2w_mats


def sort_nearby_views_by_angle(query_pose, ref_poses):
    query_direction = np.sum(query_pose[:3, 2:4], axis=-1)
    query_direction = query_direction / np.linalg.norm(query_direction)
    ref_directions = np.sum(ref_poses[:, :3, 2:4], axis=-1)
    ref_directions = ref_directions / np.linalg.norm(ref_directions, axis=-1, keepdims=True)
    inner_product = np.sum(ref_directions * query_direction[None, ...], axis=1)
    sorted_inds = np.argsort(inner_product)[::-1]
    return sorted_inds


class Camera(object):
    """Represents a Camera with intrinsics and world from/to camera transforms.
    Attributes:
      w_f_c: The world from camera 4x4 matrix.
      c_f_w: The camera from world 4x4 matrix.
      intrinsics: The camera intrinsics as a 3x3 matrix.
      inv_intrinsics: The inverse of camera intrinsics matrix.
    """

    def __init__(self, intrinsics, w_f_c):
        """Constructor.
        Args:
          intrinsics: A numpy 3x3 array representing intrinsics.
          w_f_c: A numpy 4x4 array representing wFc.
        """
        self.intrinsics = intrinsics
        self.inv_intrinsics = np.linalg.inv(intrinsics)
        self.w_f_c = w_f_c
        self.c_f_w = np.linalg.inv(w_f_c)


class View(object):
    """Represents an image and associated camera geometry.
    Attributes:
      camera: The camera for this view.
      image: The np array containing the image data.
      image_path: The file path to the image.
      shape: The 2D shape of the image.
    """

    def __init__(self, image_path, shape, camera):
        self.image_path = image_path
        self.shape = shape
        self.camera = camera
        self.image = None


def _WorldFromCameraFromViewDict(view_json):
    """Fills the world from camera transform from the view_json.
    Args:
        view_json: A dictionary of view parameters.
    Returns:
        A 4x4 transform matrix representing the world from camera transform.
    """
    transform = np.identity(4)
    position = view_json['position']
    transform[0:3, 3] = (position[0], position[1], position[2])
    orientation = view_json['orientation']
    angle_axis = np.array([orientation[0], orientation[1], orientation[2]])
    angle = np.linalg.norm(angle_axis)
    epsilon = 1e-7
    if abs(angle) < epsilon:
        # No rotation
        return transform

    axis = angle_axis / angle
    rot_mat = quaternion_matrix(quaternion_about_axis(-angle, axis))
    transform[0:3, 0:3] = rot_mat[0:3, 0:3]
    return transform


def _IntrinsicsFromViewDict(view_params):
    """Fills the intrinsics matrix from view_params.
    Args:
        view_params: Dict view parameters.
    Returns:
        A 3x3 matrix representing the camera intrinsics.
    """
    intrinsics = np.identity(3)
    intrinsics[0, 0] = view_params['focal_length']
    intrinsics[1, 1] = (view_params['focal_length'] * view_params['pixel_aspect_ratio'])
    intrinsics[0, 2] = view_params['principal_point'][0]
    intrinsics[1, 2] = view_params['principal_point'][1]
    return intrinsics


def ReadView(base_dir, view_json):
    return View(
        image_path=os.path.join(base_dir, view_json['relative_path']),
        shape=(int(view_json['height']), int(view_json['width'])),
        camera=Camera(
            _IntrinsicsFromViewDict(view_json),
            _WorldFromCameraFromViewDict(view_json)))


def ReadScene(base_dir):
    """Reads a scene from the directory base_dir."""
    with open(os.path.join(base_dir, 'models.json')) as f:
        model_json = json.load(f)

    all_views = []
    for views in model_json:
        all_views.append([ReadView(base_dir, view_json) for view_json in views])
    return all_views


def InterpolateDepths(near_depth, far_depth, num_depths):
    """Returns num_depths from (far_depth, near_depth), interpolated in inv depth.
    Args:
        near_depth: The first depth.
        far_depth: The last depth.
        num_depths: The total number of depths to create, include near_depth and
        far_depth are always included and other depths are interpolated between
        them, in inverse depth space.
    Returns:
        The depths sorted in descending order (so furthest first). This order is
        useful for back to front compositing.
  """

    inv_near_depth = 1.0 / near_depth
    inv_far_depth = 1.0 / far_depth
    depths = []
    for i in range(0, num_depths):
        fraction = float(i) / float(num_depths - 1)
        inv_depth = inv_far_depth + (inv_near_depth - inv_far_depth) * fraction
        depths.append(1.0 / inv_depth)
    return depths


def ReadViewImages(views):
    """Reads the images for the passed views."""
    for view in views:
        # Keep images unnormalized as uint8 to save RAM and transmission time to
        # and from the GPU.
        view.image = np.array(Image.open(view.image_path))


def WriteNpToImage(np_image, path):
    """Writes an image as a numpy array to the passed path.
        If the input has more than four channels only the first four will be
        written. If the input has a single channel it will be duplicated and
        written as a three channel image.
    Args:
        np_image: A numpy array.
        path: The path to write to.
    Raises:
        IOError: if the image format isn't recognized.
    """

    min_value = np.amin(np_image)
    max_value = np.amax(np_image)
    if min_value < 0.0 or max_value > 255.1:
        print('Warning: Outside image bounds, min: %f, max:%f, clipping.', min_value, max_value)
        np.clip(np_image, 0.0, 255.0)
    if np_image.shape[2] == 1:
        np_image = np.concatenate((np_image, np_image, np_image), axis=2)

    if np_image.shape[2] == 3:
        image = Image.fromarray(np_image.astype(np.uint8))
    elif np_image.shape[2] == 4:
        image = Image.fromarray(np_image.astype(np.uint8), 'RGBA')

    _, ext = os.path.splitext(path)
    ext = ext[1:]
    if ext.lower() == 'png':
        image.save(path, format='PNG')
    elif ext.lower() in ('jpg', 'jpeg'):
        image.save(path, format='JPEG')
    else:
        raise IOError('Unrecognized format for %s' % path)


# only for training
class SpacesDataset(Dataset):
    def __init__(self, args, mode, **kwargs):
        self.folder_path = os.path.join(args.rootdir, 'data/spaces_dataset/data/800/')
        self.num_source_views = args.num_source_views
        self.mode = mode
        assert mode in ['train', 'test', 'validation']
        eval_scene_ids = [0, 9, 10, 23, 24, 52, 56, 62, 63, 73]
        train_scene_ids = [i for i in np.arange(0, 100) if i not in eval_scene_ids]
        if mode == 'train':
            self.scene_dirs = [os.path.join(self.folder_path, 'scene_{:03d}'.format(i)) for i in train_scene_ids]
        else:
            self.scene_dirs = [os.path.join(self.folder_path, 'scene_{:03d}'.format(i)) for i in eval_scene_ids]

        self.all_views_scenes = []
        for scene_dir in self.scene_dirs:
            views = ReadScene(scene_dir)
            self.all_views_scenes.append(views)

        self.input_view_types = ["small_quad", "medium_quad", "large_quad", "dense"]
        self.eval_view_indices_dict = {
            "small_quad": [5, 6, 7],
            "medium_quad": [2, 4, 5, 6, 7, 11],
            "large_quad": [1, 2, 4, 5, 6, 7, 8, 10, 11],
            "dense": [5, 7, 10, 11]
        }
        self.input_indices_dict = {
            "small_quad": [1, 2, 10, 11],
            "medium_quad": [1, 3, 10, 12],
            "large_quad": [0, 3, 9, 12],
            "dense": [0, 1, 2, 3, 4, 6, 8, 9, 12, 13, 14, 15]
        }

    def __len__(self):
        return len(self.all_views_scenes)

    def __getitem__(self, idx):
        all_views = self.all_views_scenes[idx]
        num_rigs = len(all_views)
        selected_rig_id = np.random.randint(low=0, high=num_rigs)  # select a rig position
        rig_selected = all_views[selected_rig_id]
        type = np.random.choice(self.input_view_types)  # select an input type
        input_ids = self.input_indices_dict[type]
        if len(input_ids) > self.num_source_views:
            input_ids = np.random.choice(input_ids, self.num_source_views, replace=False)

        ref_cameras = []
        ref_rgbs = []
        w_max, h_max = 0, 0
        for id in input_ids:
            ref_camera, ref_rgb = view_obj2camera_rgb(rig_selected[id])
            ref_rgbs.append(ref_rgb)
            ref_cameras.append(ref_camera)
            h, w = ref_rgb.shape[:2]
            w_max = max(w, w_max)
            h_max = max(h, h_max)

        ref_rgbs_np = np.zeros((len(ref_rgbs), h_max, w_max, 3), dtype=np.float32)
        for i, ref_rgb in enumerate(ref_rgbs):
            orig_h, orig_w = ref_rgb.shape[:2]
            h_start = int((h_max - orig_h) / 2.)
            w_start = int((w_max - orig_w) / 2.)
            ref_rgbs_np[i, h_start:h_start+orig_h, w_start:w_start+orig_w] = ref_rgb
            ref_cameras[i][4] += (w_max - orig_w) / 2.
            ref_cameras[i][8] += (h_max - orig_h) / 2.
            ref_cameras[i][0] = h_max
            ref_cameras[i][1] = w_max

        # select target view
        if self.mode != 'train':
            target_id = np.random.choice(self.eval_view_indices_dict[type])
            target_view = rig_selected[target_id]
            target_camera, target_rgb = view_obj2camera_rgb(target_view)
        else:
            target_view = sample_target_view_for_training(all_views, selected_rig_id, input_ids)
            target_camera, target_rgb = view_obj2camera_rgb(target_view)

        ref_cameras = np.array(ref_cameras)
        if np.random.choice([0, 1], p=[0.5, 0.5]) and self.mode == 'train':
            target_rgb, target_camera, ref_rgbs_np, ref_cameras = random_flip(target_rgb, target_camera,
                                                                              ref_rgbs_np, ref_cameras)

        near_depth = 1.
        far_depth = 100.
        depth_range = torch.tensor([near_depth, far_depth])

        return {'rgb': torch.from_numpy(target_rgb).float(),
                'camera': torch.from_numpy(target_camera).float(),
                'rgb_path': target_view.image_path,
                'src_rgbs': torch.from_numpy(ref_rgbs_np).float(),
                'src_cameras': torch.from_numpy(np.stack(ref_cameras, axis=0)).float(),
                'depth_range': depth_range
                }


class SpacesFreeDataset(Dataset):
    def __init__(self, args, mode, **kwargs):
        self.folder_path = os.path.join(args.rootdir, 'data/spaces_dataset/data/800/')
        self.mode = mode
        self.num_source_views = args.num_source_views
        self.random_crop = True
        assert mode in ['train', 'test', 'validation']
        # eval_scene_ids = [0, 9, 10, 23, 24, 52, 56, 62, 63, 73]
        eval_scene_ids = []
        # use all 100 scenes in spaces dataset for training
        train_scene_ids = [i for i in np.arange(0, 100) if i not in eval_scene_ids]
        if mode == 'train':
            self.scene_dirs = [os.path.join(self.folder_path, 'scene_{:03d}'.format(i)) for i in train_scene_ids]
        else:
            self.scene_dirs = [os.path.join(self.folder_path, 'scene_{:03d}'.format(i)) for i in eval_scene_ids]

        self.all_views_scenes = []
        self.all_rgb_paths_scenes = []
        self.all_intrinsics_scenes = []
        self.all_img_sizes_scenes = []
        self.all_c2w_scenes = []
        for scene_dir in self.scene_dirs:
            views = ReadScene(scene_dir)
            self.all_views_scenes.append(views)
            rgb_paths, img_sizes, intrinsicss, c2w_mats = get_all_views_in_scene_cam_path(views)
            self.all_rgb_paths_scenes.append(rgb_paths)
            self.all_img_sizes_scenes.append(img_sizes)
            self.all_intrinsics_scenes.append(intrinsicss)
            self.all_c2w_scenes.append(c2w_mats)

    def __len__(self):
        return len(self.all_views_scenes)

    def __getitem__(self, idx):
        all_views = self.all_views_scenes[idx]
        num_rigs = len(all_views)
        selected_rig_id = np.random.randint(low=0, high=num_rigs)  # select a rig position
        rig_selected = all_views[selected_rig_id]
        cam_id_selected = np.random.choice(16)
        cam_selected = rig_selected[cam_id_selected]
        render_camera, render_rgb = view_obj2camera_rgb(cam_selected)
        all_c2w_mats = self.all_c2w_scenes[idx]
        all_rgb_paths = self.all_rgb_paths_scenes[idx]
        all_intrinsics = self.all_intrinsics_scenes[idx]
        all_img_sizes = self.all_img_sizes_scenes[idx]
        sorted_ids = sort_nearby_views_by_angle(render_camera[-16:].reshape(4, 4), np.array(all_c2w_mats))
        nearby_view_ids_selected = np.random.choice(sorted_ids[1:],
                                                    self.num_source_views, replace=False)

        ref_cameras = []
        ref_rgbs = []
        w_max, h_max = 0, 0
        for id in nearby_view_ids_selected:
            rgb_path = all_rgb_paths[id]
            ref_rgb = imageio.imread(rgb_path).astype(np.float32) / 255.
            h_in_view, w_in_view = all_img_sizes[id]
            h_img, w_img = ref_rgb.shape[:2]
            ref_rgbs.append(ref_rgb)
            ref_intrinsics = all_intrinsics[id]
            if h_in_view != h_img or w_in_view != w_img:
                ref_intrinsics[0] *= w_img / w_in_view
                ref_intrinsics[1] *= h_img / h_in_view
            ref_c2w = all_c2w_mats[id]
            ref_camera = np.concatenate([list(ref_rgb.shape[:2]), ref_intrinsics.flatten(), ref_c2w.flatten()])
            ref_cameras.append(ref_camera)
            h, w = ref_rgb.shape[:2]
            w_max = max(w, w_max)
            h_max = max(h, h_max)

        ref_rgbs_np = np.ones((len(ref_rgbs), h_max, w_max, 3), dtype=np.float32)
        for i, ref_rgb in enumerate(ref_rgbs):
            orig_h, orig_w = ref_rgb.shape[:2]
            h_start = int((h_max - orig_h) / 2.)
            w_start = int((w_max - orig_w) / 2.)
            ref_rgbs_np[i, h_start:h_start+orig_h, w_start:w_start+orig_w] = ref_rgb
            ref_cameras[i][4] += (w_max - orig_w) / 2.
            ref_cameras[i][8] += (h_max - orig_h) / 2.
            ref_cameras[i][0] = h_max
            ref_cameras[i][1] = w_max

        ref_cameras = np.array(ref_cameras)
        if self.mode == 'train' and self.random_crop:
            render_rgb, render_camera, ref_rgbs_np, ref_cameras = random_crop(render_rgb, render_camera,
                                                                              ref_rgbs_np, ref_cameras)

        if self.mode == 'train' and np.random.choice([0, 1]):
            render_rgb, render_camera, ref_rgbs_np, ref_cameras = random_flip(render_rgb, render_camera,
                                                                              ref_rgbs_np, ref_cameras)

        near_depth = 0.7
        far_depth = 100
        depth_range = torch.tensor([near_depth, far_depth])

        return {'rgb': torch.from_numpy(render_rgb).float(),
                'camera': torch.from_numpy(render_camera).float(),
                'rgb_path': cam_selected.image_path,
                'src_rgbs': torch.from_numpy(ref_rgbs_np).float(),
                'src_cameras': torch.from_numpy(np.stack(ref_cameras, axis=0)).float(),
                'depth_range': depth_range
                }



```

# prompts/papers/IBRNet/ibrnet/feature_network.py

``` py
# Copyright 2020 Google LLC
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib


def class_for_name(module_name, class_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    return getattr(m, class_name)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation, padding_mode='reflect')


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, padding_mode='reflect')


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, track_running_stats=False, affine=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width, track_running_stats=False, affine=True)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width, track_running_stats=False, affine=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(num_in_layers,
                              num_out_layers,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(self.kernel_size - 1) // 2,
                              padding_mode='reflect')
        self.bn = nn.InstanceNorm2d(num_out_layers, track_running_stats=False, affine=True)

    def forward(self, x):
        return F.elu(self.bn(self.conv(x)), inplace=True)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, align_corners=True, mode='bilinear')
        return self.conv(x)


class ResUNet(nn.Module):
    def __init__(self,
                 encoder='resnet34',
                 coarse_out_ch=32,
                 fine_out_ch=32,
                 norm_layer=None,
                 coarse_only=False
                 ):

        super(ResUNet, self).__init__()
        assert encoder in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], "Incorrect encoder type"
        if encoder in ['resnet18', 'resnet34']:
            filters = [64, 128, 256, 512]
        else:
            filters = [256, 512, 1024, 2048]
        self.coarse_only = coarse_only
        if self.coarse_only:
            fine_out_ch = 0
        self.coarse_out_ch = coarse_out_ch
        self.fine_out_ch = fine_out_ch
        out_ch = coarse_out_ch + fine_out_ch

        # original
        layers = [3, 4, 6, 3]
        if norm_layer is None:
            # norm_layer = nn.BatchNorm2d
            norm_layer = nn.InstanceNorm2d
        self._norm_layer = norm_layer
        self.dilation = 1
        block = BasicBlock
        replace_stride_with_dilation = [False, False, False]
        self.inplanes = 64
        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False, padding_mode='reflect')
        self.bn1 = norm_layer(self.inplanes, track_running_stats=False, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])

        # decoder
        self.upconv3 = upconv(filters[2], 128, 3, 2)
        self.iconv3 = conv(filters[1] + 128, 128, 3, 1)
        self.upconv2 = upconv(128, 64, 3, 2)
        self.iconv2 = conv(filters[0] + 64, out_ch, 3, 1)

        # fine-level conv
        self.out_conv = nn.Conv2d(out_ch, out_ch, 1, 1)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion, track_running_stats=False, affine=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def skipconnect(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        x = self.upconv3(x3)
        x = self.skipconnect(x2, x)
        x = self.iconv3(x)

        x = self.upconv2(x)
        x = self.skipconnect(x1, x)
        x = self.iconv2(x)

        x_out = self.out_conv(x)

        if self.coarse_only:
            x_coarse = x_out
            x_fine = None
        else:
            x_coarse = x_out[:, :self.coarse_out_ch, :]
            x_fine = x_out[:, -self.fine_out_ch:, :]
        return x_coarse, x_fine


```

# prompts/papers/IBRNet/ibrnet/mlp_network.py

``` py
# Copyright 2020 Google LLC
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            # attn = attn * mask

        attn = F.softmax(attn, dim=-1)
        # attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        # x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # q = self.dropout(self.fc(q))
        q = self.fc(q)
        q += residual

        q = self.layer_norm(q)

        return q, attn


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


# default tensorflow initialization of linear layers
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


@torch.jit.script
def fused_mean_variance(x, weight):
    mean = torch.sum(x*weight, dim=2, keepdim=True)
    var = torch.sum(weight * (x - mean)**2, dim=2, keepdim=True)
    return mean, var


class IBRNet(nn.Module):
    def __init__(self, args, in_feat_ch=32, n_samples=64, **kwargs):
        super(IBRNet, self).__init__()
        self.args = args
        self.anti_alias_pooling = args.anti_alias_pooling
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        activation_func = nn.ELU(inplace=True)
        self.n_samples = n_samples
        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16, in_feat_ch + 3),
                                        activation_func)

        self.base_fc = nn.Sequential(nn.Linear((in_feat_ch+3)*3, 64),
                                     activation_func,
                                     nn.Linear(64, 32),
                                     activation_func)

        self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                    activation_func,
                                    nn.Linear(32, 33),
                                    activation_func,
                                    )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                     activation_func,
                                     nn.Linear(32, 1),
                                     nn.Sigmoid()
                                     )

        self.geometry_fc = nn.Sequential(nn.Linear(32*2+1, 64),
                                         activation_func,
                                         nn.Linear(64, 16),
                                         activation_func)

        self.ray_attention = MultiHeadAttention(4, 16, 4, 4)
        self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16),
                                             activation_func,
                                             nn.Linear(16, 1),
                                             nn.ReLU())

        self.rgb_fc = nn.Sequential(nn.Linear(32+1+4, 16),
                                    activation_func,
                                    nn.Linear(16, 8),
                                    activation_func,
                                    nn.Linear(8, 1))

        self.pos_encoding = self.posenc(d_hid=16, n_samples=self.n_samples)

        self.base_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        self.vis_fc.apply(weights_init)
        self.geometry_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)

    def posenc(self, d_hid, n_samples):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table).to("cuda:{}".format(self.args.local_rank)).float().unsqueeze(0)
        return sinusoid_table

    def forward(self, rgb_feat, ray_diff, mask):
        '''
        :param rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        :param ray_diff: ray direction difference [n_rays, n_samples, n_views, 4], first 3 channels are directions,
        last channel is inner product
        :param mask: mask for whether each projection is valid or not. [n_rays, n_samples, n_views, 1]
        :return: rgb and density output, [n_rays, n_samples, 4]
        '''

        num_views = rgb_feat.shape[2]
        direction_feat = self.ray_dir_fc(ray_diff)
        rgb_in = rgb_feat[..., :3]
        rgb_feat = rgb_feat + direction_feat
        if self.anti_alias_pooling:
            _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=2, keepdim=True)[0]) * mask
            weight = weight / (torch.sum(weight, dim=2, keepdim=True) + 1e-8)
        else:
            weight = mask / (torch.sum(mask, dim=2, keepdim=True) + 1e-8)

        # compute mean and variance across different views for each point
        mean, var = fused_mean_variance(rgb_feat, weight)  # [n_rays, n_samples, 1, n_feat]
        globalfeat = torch.cat([mean, var], dim=-1)  # [n_rays, n_samples, 1, 2*n_feat]

        x = torch.cat([globalfeat.expand(-1, -1, num_views, -1), rgb_feat], dim=-1)  # [n_rays, n_samples, n_views, 3*n_feat]
        x = self.base_fc(x)

        x_vis = self.vis_fc(x * weight)
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1]-1, 1], dim=-1)
        vis = F.sigmoid(vis) * mask
        x = x + x_res
        vis = self.vis_fc2(x * vis) * mask
        weight = vis / (torch.sum(vis, dim=2, keepdim=True) + 1e-8)

        mean, var = fused_mean_variance(x, weight)
        globalfeat = torch.cat([mean.squeeze(2), var.squeeze(2), weight.mean(dim=2)], dim=-1)  # [n_rays, n_samples, 32*2+1]
        globalfeat = self.geometry_fc(globalfeat)  # [n_rays, n_samples, 16]
        num_valid_obs = torch.sum(mask, dim=2)
        globalfeat = globalfeat + self.pos_encoding
        globalfeat, _ = self.ray_attention(globalfeat, globalfeat, globalfeat,
                                           mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
        sigma = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
        sigma_out = sigma.masked_fill(num_valid_obs < 1, 0.)  # set the sigma of invalid point to zero

        # rgb computation
        x = torch.cat([x, vis, ray_diff], dim=-1)
        x = self.rgb_fc(x)
        x = x.masked_fill(mask == 0, -1e9)
        blending_weights_valid = F.softmax(x, dim=2)  # color blending
        rgb_out = torch.sum(rgb_in*blending_weights_valid, dim=2)
        out = torch.cat([rgb_out, sigma_out], dim=-1)
        return out



```

# prompts/papers/IBRNet/ibrnet/model.py

``` py
# Copyright 2020 Google LLC
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


import torch
import os
from ibrnet.mlp_network import IBRNet
from ibrnet.feature_network import ResUNet


def de_parallel(model):
    return model.module if hasattr(model, 'module') else model

########################################################################################################################
# creation/saving/loading of nerf
########################################################################################################################


class IBRNetModel(object):
    def __init__(self, args, load_opt=True, load_scheduler=True):
        self.args = args
        device = torch.device('cuda:{}'.format(args.local_rank))
        # create coarse IBRNet
        self.net_coarse = IBRNet(args,
                                 in_feat_ch=self.args.coarse_feat_dim,
                                 n_samples=self.args.N_samples).to(device)
        if args.coarse_only:
            self.net_fine = None
        else:
            # create coarse IBRNet
            self.net_fine = IBRNet(args,
                                   in_feat_ch=self.args.fine_feat_dim,
                                   n_samples=self.args.N_samples+self.args.N_importance).to(device)

        # create feature extraction network
        self.feature_net = ResUNet(coarse_out_ch=self.args.coarse_feat_dim,
                                   fine_out_ch=self.args.fine_feat_dim,
                                   coarse_only=self.args.coarse_only).cuda()

        # optimizer and learning rate scheduler
        learnable_params = list(self.net_coarse.parameters())
        learnable_params += list(self.feature_net.parameters())
        if self.net_fine is not None:
            learnable_params += list(self.net_fine.parameters())

        if self.net_fine is not None:
            self.optimizer = torch.optim.Adam([
                {'params': self.net_coarse.parameters()},
                {'params': self.net_fine.parameters()},
                {'params': self.feature_net.parameters(), 'lr': args.lrate_feature}],
                lr=args.lrate_mlp)
        else:
            self.optimizer = torch.optim.Adam([
                {'params': self.net_coarse.parameters()},
                {'params': self.feature_net.parameters(), 'lr': args.lrate_feature}],
                lr=args.lrate_mlp)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=args.lrate_decay_steps,
                                                         gamma=args.lrate_decay_factor)

        out_folder = os.path.join(args.rootdir, 'out', args.expname)
        self.start_step = self.load_from_ckpt(out_folder,
                                              load_opt=load_opt,
                                              load_scheduler=load_scheduler)

        if args.distributed:
            self.net_coarse = torch.nn.parallel.DistributedDataParallel(
                self.net_coarse,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )

            self.feature_net = torch.nn.parallel.DistributedDataParallel(
                self.feature_net,
                device_ids=[args.local_rank],
                output_device=args.local_rank
            )

            if self.net_fine is not None:
                self.net_fine = torch.nn.parallel.DistributedDataParallel(
                    self.net_fine,
                    device_ids=[args.local_rank],
                    output_device=args.local_rank
                )

    def switch_to_eval(self):
        self.net_coarse.eval()
        self.feature_net.eval()
        if self.net_fine is not None:
            self.net_fine.eval()

    def switch_to_train(self):
        self.net_coarse.train()
        self.feature_net.train()
        if self.net_fine is not None:
            self.net_fine.train()

    def save_model(self, filename):
        to_save = {'optimizer': self.optimizer.state_dict(),
                   'scheduler': self.scheduler.state_dict(),
                   'net_coarse': de_parallel(self.net_coarse).state_dict(),
                   'feature_net': de_parallel(self.feature_net).state_dict()
                   }

        if self.net_fine is not None:
            to_save['net_fine'] = de_parallel(self.net_fine).state_dict()

        torch.save(to_save, filename)

    def load_model(self, filename, load_opt=True, load_scheduler=True):
        if self.args.distributed:
            to_load = torch.load(filename, map_location='cuda:{}'.format(self.args.local_rank))
        else:
            to_load = torch.load(filename)

        if load_opt:
            self.optimizer.load_state_dict(to_load['optimizer'])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load['scheduler'])

        self.net_coarse.load_state_dict(to_load['net_coarse'])
        self.feature_net.load_state_dict(to_load['feature_net'])

        if self.net_fine is not None and 'net_fine' in to_load.keys():
            self.net_fine.load_state_dict(to_load['net_fine'])

    def load_from_ckpt(self, out_folder,
                       load_opt=True,
                       load_scheduler=True,
                       force_latest_ckpt=False):
        '''
        load model from existing checkpoints and return the current step
        :param out_folder: the directory that stores ckpts
        :return: the current starting step
        '''

        # all existing ckpts
        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [os.path.join(out_folder, f)
                     for f in sorted(os.listdir(out_folder)) if f.endswith('.pth')]

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):  # load the specified ckpt
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath, load_opt, load_scheduler)
            step = int(fpath[-10:-4])
            print('Reloading from {}, starting at step={}'.format(fpath, step))
        else:
            print('No ckpts found, training from scratch...')
            step = 0

        return step




```

# prompts/papers/IBRNet/ibrnet/projection.py

``` py
# Copyright 2020 Google LLC
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


import torch
import torch.nn.functional as F


class Projector():
    def __init__(self, device):
        self.device = device

    def inbound(self, pixel_locations, h, w):
        '''
        check if the pixel locations are in valid range
        :param pixel_locations: [..., 2]
        :param h: height
        :param w: weight
        :return: mask, bool, [...]
        '''
        return (pixel_locations[..., 0] <= w - 1.) & \
               (pixel_locations[..., 0] >= 0) & \
               (pixel_locations[..., 1] <= h - 1.) &\
               (pixel_locations[..., 1] >= 0)

    def normalize(self, pixel_locations, h, w):
        resize_factor = torch.tensor([w-1., h-1.]).to(pixel_locations.device)[None, None, :]
        normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.  # [n_views, n_points, 2]
        return normalized_pixel_locations

    def compute_projections(self, xyz, train_cameras):
        '''
        project 3D points into cameras
        :param xyz: [..., 3]
        :param train_cameras: [n_views, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        :return: pixel locations [..., 2], mask [...]
        '''
        original_shape = xyz.shape[:2]
        xyz = xyz.reshape(-1, 3)
        num_views = len(train_cameras)
        train_intrinsics = train_cameras[:, 2:18].reshape(-1, 4, 4)  # [n_views, 4, 4]
        train_poses = train_cameras[:, -16:].reshape(-1, 4, 4)  # [n_views, 4, 4]
        xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)  # [n_points, 4]
        projections = train_intrinsics.bmm(torch.inverse(train_poses)) \
            .bmm(xyz_h.t()[None, ...].repeat(num_views, 1, 1))  # [n_views, 4, n_points]
        projections = projections.permute(0, 2, 1)  # [n_views, n_points, 4]
        pixel_locations = projections[..., :2] / torch.clamp(projections[..., 2:3], min=1e-8)  # [n_views, n_points, 2]
        pixel_locations = torch.clamp(pixel_locations, min=-1e6, max=1e6)
        mask = projections[..., 2] > 0   # a point is invalid if behind the camera
        return pixel_locations.reshape((num_views, ) + original_shape + (2, )), \
               mask.reshape((num_views, ) + original_shape)

    def compute_angle(self, xyz, query_camera, train_cameras):
        '''
        :param xyz: [..., 3]
        :param query_camera: [34, ]
        :param train_cameras: [n_views, 34]
        :return: [n_views, ..., 4]; The first 3 channels are unit-length vector of the difference between
        query and target ray directions, the last channel is the inner product of the two directions.
        '''
        original_shape = xyz.shape[:2]
        xyz = xyz.reshape(-1, 3)
        train_poses = train_cameras[:, -16:].reshape(-1, 4, 4)  # [n_views, 4, 4]
        num_views = len(train_poses)
        query_pose = query_camera[-16:].reshape(-1, 4, 4).repeat(num_views, 1, 1)  # [n_views, 4, 4]
        ray2tar_pose = (query_pose[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0))
        ray2tar_pose /= (torch.norm(ray2tar_pose, dim=-1, keepdim=True) + 1e-6)
        ray2train_pose = (train_poses[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0))
        ray2train_pose /= (torch.norm(ray2train_pose, dim=-1, keepdim=True) + 1e-6)
        ray_diff = ray2tar_pose - ray2train_pose
        ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)
        ray_diff_dot = torch.sum(ray2tar_pose * ray2train_pose, dim=-1, keepdim=True)
        ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)
        ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1)
        ray_diff = ray_diff.reshape((num_views, ) + original_shape + (4, ))
        return ray_diff

    def compute(self,  xyz, query_camera, train_imgs, train_cameras, featmaps):
        '''
        :param xyz: [n_rays, n_samples, 3]
        :param query_camera: [1, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        :param train_imgs: [1, n_views, h, w, 3]
        :param train_cameras: [1, n_views, 34]
        :param featmaps: [n_views, d, h, w]
        :return: rgb_feat_sampled: [n_rays, n_samples, 3+n_feat],
                 ray_diff: [n_rays, n_samples, 4],
                 mask: [n_rays, n_samples, 1]
        '''
        assert (train_imgs.shape[0] == 1) \
               and (train_cameras.shape[0] == 1) \
               and (query_camera.shape[0] == 1), 'only support batch_size=1 for now'

        train_imgs = train_imgs.squeeze(0)  # [n_views, h, w, 3]
        train_cameras = train_cameras.squeeze(0)  # [n_views, 34]
        query_camera = query_camera.squeeze(0)  # [34, ]

        train_imgs = train_imgs.permute(0, 3, 1, 2)  # [n_views, 3, h, w]

        h, w = train_cameras[0][:2]

        # compute the projection of the query points to each reference image
        pixel_locations, mask_in_front = self.compute_projections(xyz, train_cameras)
        normalized_pixel_locations = self.normalize(pixel_locations, h, w)   # [n_views, n_rays, n_samples, 2]

        # rgb sampling
        rgbs_sampled = F.grid_sample(train_imgs, normalized_pixel_locations, align_corners=True)
        rgb_sampled = rgbs_sampled.permute(2, 3, 0, 1)  # [n_rays, n_samples, n_views, 3]

        # deep feature sampling
        feat_sampled = F.grid_sample(featmaps, normalized_pixel_locations, align_corners=True)
        feat_sampled = feat_sampled.permute(2, 3, 0, 1)  # [n_rays, n_samples, n_views, d]
        rgb_feat_sampled = torch.cat([rgb_sampled, feat_sampled], dim=-1)   # [n_rays, n_samples, n_views, d+3]

        # mask
        inbound = self.inbound(pixel_locations, h, w)
        ray_diff = self.compute_angle(xyz, query_camera, train_cameras)
        ray_diff = ray_diff.permute(1, 2, 0, 3)
        mask = (inbound * mask_in_front).float().permute(1, 2, 0)[..., None]   # [n_rays, n_samples, n_views, 1]
        return rgb_feat_sampled, ray_diff, mask








```

# prompts/papers/IBRNet/ibrnet/render_image.py

``` py
# Copyright 2020 Google LLC
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


import torch
from collections import OrderedDict
from ibrnet.render_ray import render_rays


def render_single_image(ray_sampler,
                        ray_batch,
                        model,
                        projector,
                        chunk_size,
                        N_samples,
                        inv_uniform=False,
                        N_importance=0,
                        det=False,
                        white_bkgd=False,
                        render_stride=1,
                        featmaps=None):
    '''
    :param ray_sampler: RaySamplingSingleImage for this view
    :param model:  {'net_coarse': , 'net_fine': , ...}
    :param chunk_size: number of rays in a chunk
    :param N_samples: samples along each ray (for both coarse and fine model)
    :param inv_uniform: if True, uniformly sample inverse depth for coarse model
    :param N_importance: additional samples along each ray produced by importance sampling (for fine model)
    :return: {'outputs_coarse': {'rgb': numpy, 'depth': numpy, ...}, 'outputs_fine': {}}
    '''

    all_ret = OrderedDict([('outputs_coarse', OrderedDict()),
                           ('outputs_fine', OrderedDict())])

    N_rays = ray_batch['ray_o'].shape[0]

    for i in range(0, N_rays, chunk_size):
        chunk = OrderedDict()
        for k in ray_batch:
            if k in ['camera', 'depth_range', 'src_rgbs', 'src_cameras']:
                chunk[k] = ray_batch[k]
            elif ray_batch[k] is not None:
                chunk[k] = ray_batch[k][i:i+chunk_size]
            else:
                chunk[k] = None

        ret = render_rays(chunk, model, featmaps,
                          projector=projector,
                          N_samples=N_samples,
                          inv_uniform=inv_uniform,
                          N_importance=N_importance,
                          det=det,
                          white_bkgd=white_bkgd)

        # handle both coarse and fine outputs
        # cache chunk results on cpu
        if i == 0:
            for k in ret['outputs_coarse']:
                all_ret['outputs_coarse'][k] = []

            if ret['outputs_fine'] is None:
                all_ret['outputs_fine'] = None
            else:
                for k in ret['outputs_fine']:
                    all_ret['outputs_fine'][k] = []

        for k in ret['outputs_coarse']:
            all_ret['outputs_coarse'][k].append(ret['outputs_coarse'][k].cpu())

        if ret['outputs_fine'] is not None:
            for k in ret['outputs_fine']:
                all_ret['outputs_fine'][k].append(ret['outputs_fine'][k].cpu())

    rgb_strided = torch.ones(ray_sampler.H, ray_sampler.W, 3)[::render_stride, ::render_stride, :]
    # merge chunk results and reshape
    for k in all_ret['outputs_coarse']:
        if k == 'random_sigma':
            continue
        tmp = torch.cat(all_ret['outputs_coarse'][k], dim=0).reshape((rgb_strided.shape[0],
                                                                      rgb_strided.shape[1], -1))
        all_ret['outputs_coarse'][k] = tmp.squeeze()

    all_ret['outputs_coarse']['rgb'][all_ret['outputs_coarse']['mask'] == 0] = 1.
    if all_ret['outputs_fine'] is not None:
        for k in all_ret['outputs_fine']:
            if k == 'random_sigma':
                continue
            tmp = torch.cat(all_ret['outputs_fine'][k], dim=0).reshape((rgb_strided.shape[0],
                                                                        rgb_strided.shape[1], -1))

            all_ret['outputs_fine'][k] = tmp.squeeze()

    return all_ret





```

# prompts/papers/IBRNet/ibrnet/render_ray.py

``` py
# Copyright 2020 Google LLC
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


import torch
from collections import OrderedDict

########################################################################################################################
# helper functions for nerf ray rendering
########################################################################################################################


def sample_pdf(bins, weights, N_samples, det=False):
    '''
    :param bins: tensor of shape [N_rays, M+1], M is the number of bins
    :param weights: tensor of shape [N_rays, M]
    :param N_samples: number of samples along each ray
    :param det: if True, will perform deterministic sampling
    :return: [N_rays, N_samples]
    '''

    M = weights.shape[1]
    weights += 1e-5
    # Get pdf
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)    # [N_rays, M]
    cdf = torch.cumsum(pdf, dim=-1)  # [N_rays, M]
    cdf = torch.cat([torch.zeros_like(cdf[:, 0:1]), cdf], dim=-1) # [N_rays, M+1]

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., N_samples, device=bins.device)
        u = u.unsqueeze(0).repeat(bins.shape[0], 1)       # [N_rays, N_samples]
    else:
        u = torch.rand(bins.shape[0], N_samples, device=bins.device)

    # Invert CDF
    above_inds = torch.zeros_like(u, dtype=torch.long)       # [N_rays, N_samples]
    for i in range(M):
        above_inds += (u >= cdf[:, i:i+1]).long()

    # random sample inside each bin
    below_inds = torch.clamp(above_inds-1, min=0)
    inds_g = torch.stack((below_inds, above_inds), dim=2)     # [N_rays, N_samples, 2]

    cdf = cdf.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
    cdf_g = torch.gather(input=cdf, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    bins = bins.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, M+1]
    bins_g = torch.gather(input=bins, dim=-1, index=inds_g)  # [N_rays, N_samples, 2]

    # t = (u-cdf_g[:, :, 0]) / (cdf_g[:, :, 1] - cdf_g[:, :, 0] + TINY_NUMBER)  # [N_rays, N_samples]
    # fix numeric issue
    denom = cdf_g[:, :, 1] - cdf_g[:, :, 0]      # [N_rays, N_samples]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[:, :, 0]) / denom

    samples = bins_g[:, :, 0] + t * (bins_g[:, :, 1]-bins_g[:, :, 0])

    return samples


def sample_along_camera_ray(ray_o, ray_d, depth_range,
                            N_samples,
                            inv_uniform=False,
                            det=False):
    '''
    :param ray_o: origin of the ray in scene coordinate system; tensor of shape [N_rays, 3]
    :param ray_d: homogeneous ray direction vectors in scene coordinate system; tensor of shape [N_rays, 3]
    :param depth_range: [near_depth, far_depth]
    :param inv_uniform: if True, uniformly sampling inverse depth
    :param det: if True, will perform deterministic sampling
    :return: tensor of shape [N_rays, N_samples, 3]
    '''
    # will sample inside [near_depth, far_depth]
    # assume the nearest possible depth is at least (min_ratio * depth)
    near_depth_value = depth_range[0, 0]
    far_depth_value = depth_range[0, 1]
    assert near_depth_value > 0 and far_depth_value > 0 and far_depth_value > near_depth_value

    near_depth = near_depth_value * torch.ones_like(ray_d[..., 0])

    far_depth = far_depth_value * torch.ones_like(ray_d[..., 0])
    if inv_uniform:
        start = 1. / near_depth     # [N_rays,]
        step = (1. / far_depth - start) / (N_samples-1)
        inv_z_vals = torch.stack([start+i*step for i in range(N_samples)], dim=1)  # [N_rays, N_samples]
        z_vals = 1. / inv_z_vals
    else:
        start = near_depth
        step = (far_depth - near_depth) / (N_samples-1)
        z_vals = torch.stack([start+i*step for i in range(N_samples)], dim=1)  # [N_rays, N_samples]

    if not det:
        # get intervals between samples
        mids = .5 * (z_vals[:, 1:] + z_vals[:, :-1])
        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
        lower = torch.cat([z_vals[:, 0:1], mids], dim=-1)
        # uniform samples in those intervals
        t_rand = torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * t_rand   # [N_rays, N_samples]

    ray_d = ray_d.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, 3]
    ray_o = ray_o.unsqueeze(1).repeat(1, N_samples, 1)
    pts = z_vals.unsqueeze(2) * ray_d + ray_o       # [N_rays, N_samples, 3]
    return pts, z_vals


########################################################################################################################
# ray rendering of nerf
########################################################################################################################

def raw2outputs(raw, z_vals, mask, white_bkgd=False):
    '''
    :param raw: raw network output; tensor of shape [N_rays, N_samples, 4]
    :param z_vals: depth of point samples along rays; tensor of shape [N_rays, N_samples]
    :param ray_d: [N_rays, 3]
    :return: {'rgb': [N_rays, 3], 'depth': [N_rays,], 'weights': [N_rays,], 'depth_std': [N_rays,]}
    '''
    rgb = raw[:, :, :3]     # [N_rays, N_samples, 3]
    sigma = raw[:, :, 3]    # [N_rays, N_samples]

    # note: we did not use the intervals here, because in practice different scenes from COLMAP can have
    # very different scales, and using interval can affect the model's generalization ability.
    # Therefore we don't use the intervals for both training and evaluation.
    sigma2alpha = lambda sigma, dists: 1. - torch.exp(-sigma)

    # point samples are ordered with increasing depth
    # interval between samples
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat((dists, dists[:, -1:]), dim=-1)  # [N_rays, N_samples]

    alpha = sigma2alpha(sigma, dists)  # [N_rays, N_samples]

    # Eq. (3): T
    T = torch.cumprod(1. - alpha + 1e-10, dim=-1)[:, :-1]   # [N_rays, N_samples-1]
    T = torch.cat((torch.ones_like(T[:, 0:1]), T), dim=-1)  # [N_rays, N_samples]

    # maths show weights, and summation of weights along a ray, are always inside [0, 1]
    weights = alpha * T     # [N_rays, N_samples]
    rgb_map = torch.sum(weights.unsqueeze(2) * rgb, dim=1)  # [N_rays, 3]

    if white_bkgd:
        rgb_map = rgb_map + (1. - torch.sum(weights, dim=-1, keepdim=True))

    mask = mask.float().sum(dim=1) > 8  # should at least have 8 valid observation on the ray, otherwise don't consider its loss
    depth_map = torch.sum(weights * z_vals, dim=-1)     # [N_rays,]

    ret = OrderedDict([('rgb', rgb_map),
                       ('depth', depth_map),
                       ('weights', weights),                # used for importance sampling of fine samples
                       ('mask', mask),
                       ('alpha', alpha),
                       ('z_vals', z_vals)
                       ])

    return ret


def render_rays(ray_batch,
                model,
                featmaps,
                projector,
                N_samples,
                inv_uniform=False,
                N_importance=0,
                det=False,
                white_bkgd=False):
    '''
    :param ray_batch: {'ray_o': [N_rays, 3] , 'ray_d': [N_rays, 3], 'view_dir': [N_rays, 2]}
    :param model:  {'net_coarse':  , 'net_fine': }
    :param N_samples: samples along each ray (for both coarse and fine model)
    :param inv_uniform: if True, uniformly sample inverse depth for coarse model
    :param N_importance: additional samples along each ray produced by importance sampling (for fine model)
    :param det: if True, will deterministicly sample depths
    :return: {'outputs_coarse': {}, 'outputs_fine': {}}
    '''

    ret = {'outputs_coarse': None,
           'outputs_fine': None}

    # pts: [N_rays, N_samples, 3]
    # z_vals: [N_rays, N_samples]
    pts, z_vals = sample_along_camera_ray(ray_o=ray_batch['ray_o'],
                                          ray_d=ray_batch['ray_d'],
                                          depth_range=ray_batch['depth_range'],
                                          N_samples=N_samples, inv_uniform=inv_uniform, det=det)
    N_rays, N_samples = pts.shape[:2]

    rgb_feat, ray_diff, mask = projector.compute(pts, ray_batch['camera'],
                                                 ray_batch['src_rgbs'],
                                                 ray_batch['src_cameras'],
                                                 featmaps=featmaps[0])  # [N_rays, N_samples, N_views, x]
    pixel_mask = mask[..., 0].sum(dim=2) > 1   # [N_rays, N_samples], should at least have 2 observations
    raw_coarse = model.net_coarse(rgb_feat, ray_diff, mask)   # [N_rays, N_samples, 4]
    outputs_coarse = raw2outputs(raw_coarse, z_vals, pixel_mask,
                                 white_bkgd=white_bkgd)
    ret['outputs_coarse'] = outputs_coarse

    if N_importance > 0:
        assert model.net_fine is not None
        # detach since we would like to decouple the coarse and fine networks
        weights = outputs_coarse['weights'].clone().detach()            # [N_rays, N_samples]
        if inv_uniform:
            inv_z_vals = 1. / z_vals
            inv_z_vals_mid = .5 * (inv_z_vals[:, 1:] + inv_z_vals[:, :-1])   # [N_rays, N_samples-1]
            weights = weights[:, 1:-1]      # [N_rays, N_samples-2]
            inv_z_vals = sample_pdf(bins=torch.flip(inv_z_vals_mid, dims=[1]),
                                    weights=torch.flip(weights, dims=[1]),
                                    N_samples=N_importance, det=det)  # [N_rays, N_importance]
            z_samples = 1. / inv_z_vals
        else:
            # take mid-points of depth samples
            z_vals_mid = .5 * (z_vals[:, 1:] + z_vals[:, :-1])   # [N_rays, N_samples-1]
            weights = weights[:, 1:-1]      # [N_rays, N_samples-2]
            z_samples = sample_pdf(bins=z_vals_mid, weights=weights,
                                   N_samples=N_importance, det=det)  # [N_rays, N_importance]

        z_vals = torch.cat((z_vals, z_samples), dim=-1)  # [N_rays, N_samples + N_importance]

        # samples are sorted with increasing depth
        z_vals, _ = torch.sort(z_vals, dim=-1)
        N_total_samples = N_samples + N_importance

        viewdirs = ray_batch['ray_d'].unsqueeze(1).repeat(1, N_total_samples, 1)
        ray_o = ray_batch['ray_o'].unsqueeze(1).repeat(1, N_total_samples, 1)
        pts = z_vals.unsqueeze(2) * viewdirs + ray_o  # [N_rays, N_samples + N_importance, 3]

        rgb_feat_sampled, ray_diff, mask = projector.compute(pts, ray_batch['camera'],
                                                             ray_batch['src_rgbs'],
                                                             ray_batch['src_cameras'],
                                                             featmaps=featmaps[1])

        pixel_mask = mask[..., 0].sum(dim=2) > 1  # [N_rays, N_samples]. should at least have 2 observations
        raw_fine = model.net_fine(rgb_feat_sampled, ray_diff, mask)
        outputs_fine = raw2outputs(raw_fine, z_vals, pixel_mask,
                                   white_bkgd=white_bkgd)
        ret['outputs_fine'] = outputs_fine

    return ret


```

# prompts/papers/IBRNet/ibrnet/sample_ray.py

``` py
# Copyright 2020 Google LLC
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

import numpy as np
import torch
import torch.nn.functional as F


rng = np.random.RandomState(234)

########################################################################################################################
# ray batch sampling
########################################################################################################################


def parse_camera(params):
    H = params[:, 0]
    W = params[:, 1]
    intrinsics = params[:, 2:18].reshape((-1, 4, 4))
    c2w = params[:, 18:34].reshape((-1, 4, 4))
    return W, H, intrinsics, c2w


def dilate_img(img, kernel_size=20):
    import cv2
    assert img.dtype == np.uint8
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilation = cv2.dilate(img / 255, kernel, iterations=1) * 255
    return dilation


class RaySamplerSingleImage(object):
    def __init__(self, data, device, resize_factor=1, render_stride=1):
        super().__init__()
        self.render_stride = render_stride
        self.rgb = data['rgb'] if 'rgb' in data.keys() else None
        self.camera = data['camera']
        self.rgb_path = data['rgb_path']
        self.depth_range = data['depth_range']
        self.device = device
        W, H, self.intrinsics, self.c2w_mat = parse_camera(self.camera)
        self.batch_size = len(self.camera)

        self.H = int(H[0])
        self.W = int(W[0])

        # half-resolution output
        if resize_factor != 1:
            self.W = int(self.W * resize_factor)
            self.H = int(self.H * resize_factor)
            self.intrinsics[:, :2, :3] *= resize_factor
            if self.rgb is not None:
                self.rgb = F.interpolate(self.rgb.permute(0, 3, 1, 2), scale_factor=resize_factor).permute(0, 2, 3, 1)

        self.rays_o, self.rays_d = self.get_rays_single_image(self.H, self.W, self.intrinsics, self.c2w_mat)
        if self.rgb is not None:
            self.rgb = self.rgb.reshape(-1, 3)

        if 'src_rgbs' in data.keys():
            self.src_rgbs = data['src_rgbs']
        else:
            self.src_rgbs = None
        if 'src_cameras' in data.keys():
            self.src_cameras = data['src_cameras']
        else:
            self.src_cameras = None

    def get_rays_single_image(self, H, W, intrinsics, c2w):
        '''
        :param H: image height
        :param W: image width
        :param intrinsics: 4 by 4 intrinsic matrix
        :param c2w: 4 by 4 camera to world extrinsic matrix
        :return:
        '''
        u, v = np.meshgrid(np.arange(W)[::self.render_stride], np.arange(H)[::self.render_stride])
        u = u.reshape(-1).astype(dtype=np.float32)  # + 0.5    # add half pixel
        v = v.reshape(-1).astype(dtype=np.float32)  # + 0.5
        pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)
        pixels = torch.from_numpy(pixels)
        batched_pixels = pixels.unsqueeze(0).repeat(self.batch_size, 1, 1)

        rays_d = (c2w[:, :3, :3].bmm(torch.inverse(intrinsics[:, :3, :3])).bmm(batched_pixels)).transpose(1, 2)
        rays_d = rays_d.reshape(-1, 3)
        rays_o = c2w[:, :3, 3].unsqueeze(1).repeat(1, rays_d.shape[0], 1).reshape(-1, 3)  # B x HW x 3
        return rays_o, rays_d

    def get_all(self):
        ret = {'ray_o': self.rays_o.cuda(),
               'ray_d': self.rays_d.cuda(),
               'depth_range': self.depth_range.cuda(),
               'camera': self.camera.cuda(),
               'rgb': self.rgb.cuda() if self.rgb is not None else None,
               'src_rgbs': self.src_rgbs.cuda() if self.src_rgbs is not None else None,
               'src_cameras': self.src_cameras.cuda() if self.src_cameras is not None else None,
        }
        return ret

    def sample_random_pixel(self, N_rand, sample_mode, center_ratio=0.8):
        if sample_mode == 'center':
            border_H = int(self.H * (1 - center_ratio) / 2.)
            border_W = int(self.W * (1 - center_ratio) / 2.)

            # pixel coordinates
            u, v = np.meshgrid(np.arange(border_H, self.H - border_H),
                               np.arange(border_W, self.W - border_W))
            u = u.reshape(-1)
            v = v.reshape(-1)

            select_inds = rng.choice(u.shape[0], size=(N_rand,), replace=False)
            select_inds = v[select_inds] + self.W * u[select_inds]

        elif sample_mode == 'uniform':
            # Random from one image
            select_inds = rng.choice(self.H*self.W, size=(N_rand,), replace=False)
        else:
            raise Exception("unknown sample mode!")

        return select_inds

    def random_sample(self, N_rand, sample_mode, center_ratio=0.8):
        '''
        :param N_rand: number of rays to be casted
        :return:
        '''

        select_inds = self.sample_random_pixel(N_rand, sample_mode, center_ratio)

        rays_o = self.rays_o[select_inds]
        rays_d = self.rays_d[select_inds]

        if self.rgb is not None:
            rgb = self.rgb[select_inds]
        else:
            rgb = None

        ret = {'ray_o': rays_o.cuda(),
               'ray_d': rays_d.cuda(),
               'camera': self.camera.cuda(),
               'depth_range': self.depth_range.cuda(),
               'rgb': rgb.cuda() if rgb is not None else None,
               'src_rgbs': self.src_rgbs.cuda() if self.src_rgbs is not None else None,
               'src_cameras': self.src_cameras.cuda() if self.src_cameras is not None else None,
               'selected_inds': select_inds
        }
        return ret


```

# prompts/papers/IBRNet/train.py

``` py
# Copyright 2020 Google LLC
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


import os
import time
import numpy as np
import shutil
import torch
import torch.utils.data.distributed

from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from ibrnet.data_loaders import dataset_dict
from ibrnet.render_ray import render_rays
from ibrnet.render_image import render_single_image
from ibrnet.model import IBRNetModel
from ibrnet.sample_ray import RaySamplerSingleImage
from ibrnet.criterion import Criterion
from utils import img2mse, mse2psnr, img_HWC2CHW, colorize, cycle, img2psnr, save_current_code
import config
import torch.distributed as dist
from ibrnet.projection import Projector
from ibrnet.data_loaders.create_training_dataset import create_training_dataset


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def train(args):

    device = "cuda:{}".format(args.local_rank)
    out_folder = os.path.join(args.rootdir, 'out', args.expname)
    print('outputs will be saved to {}'.format(out_folder))
    os.makedirs(out_folder, exist_ok=True)

    # save the args and config files
    f = os.path.join(out_folder, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    if args.config is not None:
        f = os.path.join(out_folder, 'config.txt')
        if not os.path.isfile(f):
            shutil.copy(args.config, f)

    # create training dataset
    train_dataset, train_sampler = create_training_dataset(args)
    # currently only support batch_size=1 (i.e., one set of target and source views) for each GPU node
    # please use distributed parallel on multiple GPUs to train multiple target views per batch
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                               worker_init_fn=lambda _: np.random.seed(),
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               shuffle=True if train_sampler is None else False)

    # create validation dataset
    val_dataset = dataset_dict[args.eval_dataset](args, 'validation',
                                                  scenes=args.eval_scenes)

    val_loader = DataLoader(val_dataset, batch_size=1)
    val_loader_iterator = iter(cycle(val_loader))

    # Create IBRNet model
    model = IBRNetModel(args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler)
    # create projector
    projector = Projector(device=device)

    # Create criterion
    criterion = Criterion()
    tb_dir = os.path.join(args.rootdir, 'logs/', args.expname)
    if args.local_rank == 0:
        writer = SummaryWriter(tb_dir)
        print('saving tensorboard files to {}'.format(tb_dir))
    scalars_to_log = {}

    global_step = model.start_step + 1
    epoch = 0
    while global_step < model.start_step + args.n_iters + 1:
        np.random.seed()
        for train_data in train_loader:
            time0 = time.time()

            if args.distributed:
                train_sampler.set_epoch(epoch)

            # Start of core optimization loop

            # load training rays
            ray_sampler = RaySamplerSingleImage(train_data, device)
            N_rand = int(1.0 * args.N_rand * args.num_source_views / train_data['src_rgbs'][0].shape[0])
            ray_batch = ray_sampler.random_sample(N_rand,
                                                  sample_mode=args.sample_mode,
                                                  center_ratio=args.center_ratio,
                                                  )

            featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))

            ret = render_rays(ray_batch=ray_batch,
                              model=model,
                              projector=projector,
                              featmaps=featmaps,
                              N_samples=args.N_samples,
                              inv_uniform=args.inv_uniform,
                              N_importance=args.N_importance,
                              det=args.det,
                              white_bkgd=args.white_bkgd)

            # compute loss
            model.optimizer.zero_grad()
            loss, scalars_to_log = criterion(ret['outputs_coarse'], ray_batch, scalars_to_log)

            if ret['outputs_fine'] is not None:
                fine_loss, scalars_to_log = criterion(ret['outputs_fine'], ray_batch, scalars_to_log)
                loss += fine_loss

            loss.backward()
            scalars_to_log['loss'] = loss.item()
            model.optimizer.step()
            model.scheduler.step()

            scalars_to_log['lr'] = model.scheduler.get_last_lr()[0]
            # end of core optimization loop
            dt = time.time() - time0

            # Rest is logging
            if args.local_rank == 0:
                if global_step % args.i_print == 0 or global_step < 10:
                    # write mse and psnr stats
                    mse_error = img2mse(ret['outputs_coarse']['rgb'], ray_batch['rgb']).item()
                    scalars_to_log['train/coarse-loss'] = mse_error
                    scalars_to_log['train/coarse-psnr-training-batch'] = mse2psnr(mse_error)
                    if ret['outputs_fine'] is not None:
                        mse_error = img2mse(ret['outputs_fine']['rgb'], ray_batch['rgb']).item()
                        scalars_to_log['train/fine-loss'] = mse_error
                        scalars_to_log['train/fine-psnr-training-batch'] = mse2psnr(mse_error)

                    logstr = '{} Epoch: {}  step: {} '.format(args.expname, epoch, global_step)
                    for k in scalars_to_log.keys():
                        logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                        writer.add_scalar(k, scalars_to_log[k], global_step)
                    print(logstr)
                    print('each iter time {:.05f} seconds'.format(dt))

                if global_step % args.i_weights == 0:
                    print('Saving checkpoints at {} to {}...'.format(global_step, out_folder))
                    fpath = os.path.join(out_folder, 'model_{:06d}.pth'.format(global_step))
                    model.save_model(fpath)

                if global_step % args.i_img == 0:
                    print('Logging a random validation view...')
                    val_data = next(val_loader_iterator)
                    tmp_ray_sampler = RaySamplerSingleImage(val_data, device, render_stride=args.render_stride)
                    H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
                    gt_img = tmp_ray_sampler.rgb.reshape(H, W, 3)
                    log_view_to_tb(writer, global_step, args, model, tmp_ray_sampler, projector,
                                   gt_img, render_stride=args.render_stride, prefix='val/')
                    torch.cuda.empty_cache()

                    print('Logging current training view...')
                    tmp_ray_train_sampler = RaySamplerSingleImage(train_data, device,
                                                                  render_stride=1)
                    H, W = tmp_ray_train_sampler.H, tmp_ray_train_sampler.W
                    gt_img = tmp_ray_train_sampler.rgb.reshape(H, W, 3)
                    log_view_to_tb(writer, global_step, args, model, tmp_ray_train_sampler, projector,
                                   gt_img, render_stride=1, prefix='train/')
            global_step += 1
            if global_step > model.start_step + args.n_iters + 1:
                break
        epoch += 1


def log_view_to_tb(writer, global_step, args, model, ray_sampler, projector, gt_img,
                   render_stride=1, prefix=''):
    model.switch_to_eval()
    with torch.no_grad():
        ray_batch = ray_sampler.get_all()
        if model.feature_net is not None:
            featmaps = model.feature_net(ray_batch['src_rgbs'].squeeze(0).permute(0, 3, 1, 2))
        else:
            featmaps = [None, None]
        ret = render_single_image(ray_sampler=ray_sampler,
                                  ray_batch=ray_batch,
                                  model=model,
                                  projector=projector,
                                  chunk_size=args.chunk_size,
                                  N_samples=args.N_samples,
                                  inv_uniform=args.inv_uniform,
                                  det=True,
                                  N_importance=args.N_importance,
                                  white_bkgd=args.white_bkgd,
                                  render_stride=render_stride,
                                  featmaps=featmaps)

    average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))

    if args.render_stride != 1:
        gt_img = gt_img[::render_stride, ::render_stride]
        average_im = average_im[::render_stride, ::render_stride]

    rgb_gt = img_HWC2CHW(gt_img)
    average_im = img_HWC2CHW(average_im)

    rgb_pred = img_HWC2CHW(ret['outputs_coarse']['rgb'].detach().cpu())

    h_max = max(rgb_gt.shape[-2], rgb_pred.shape[-2], average_im.shape[-2])
    w_max = max(rgb_gt.shape[-1], rgb_pred.shape[-1], average_im.shape[-1])
    rgb_im = torch.zeros(3, h_max, 3*w_max)
    rgb_im[:, :average_im.shape[-2], :average_im.shape[-1]] = average_im
    rgb_im[:, :rgb_gt.shape[-2], w_max:w_max+rgb_gt.shape[-1]] = rgb_gt
    rgb_im[:, :rgb_pred.shape[-2], 2*w_max:2*w_max+rgb_pred.shape[-1]] = rgb_pred

    depth_im = ret['outputs_coarse']['depth'].detach().cpu()
    acc_map = torch.sum(ret['outputs_coarse']['weights'], dim=-1).detach().cpu()

    if ret['outputs_fine'] is None:
        depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))
        acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))
    else:
        rgb_fine = img_HWC2CHW(ret['outputs_fine']['rgb'].detach().cpu())
        rgb_fine_ = torch.zeros(3, h_max, w_max)
        rgb_fine_[:, :rgb_fine.shape[-2], :rgb_fine.shape[-1]] = rgb_fine
        rgb_im = torch.cat((rgb_im, rgb_fine_), dim=-1)
        depth_im = torch.cat((depth_im, ret['outputs_fine']['depth'].detach().cpu()), dim=-1)
        depth_im = img_HWC2CHW(colorize(depth_im, cmap_name='jet', append_cbar=True))
        acc_map = torch.cat((acc_map, torch.sum(ret['outputs_fine']['weights'], dim=-1).detach().cpu()), dim=-1)
        acc_map = img_HWC2CHW(colorize(acc_map, range=(0., 1.), cmap_name='jet', append_cbar=False))

    # write the pred/gt rgb images and depths
    writer.add_image(prefix + 'rgb_gt-coarse-fine', rgb_im, global_step)
    writer.add_image(prefix + 'depth_gt-coarse-fine', depth_im, global_step)
    writer.add_image(prefix + 'acc-coarse-fine', acc_map, global_step)

    # write scalar
    pred_rgb = ret['outputs_fine']['rgb'] if ret['outputs_fine'] is not None else ret['outputs_coarse']['rgb']
    psnr_curr_img = img2psnr(pred_rgb.detach().cpu(), gt_img)
    writer.add_scalar(prefix + 'psnr_image', psnr_curr_img, global_step)

    model.switch_to_train()


if __name__ == '__main__':
    parser = config.config_parser()
    args = parser.parse_args()

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    train(args)


```

# prompts/papers/IBRNet/utils.py

``` py
# Copyright 2020 Google LLC
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


import torch
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib as mpl
from matplotlib import cm
import cv2
import os
from datetime import datetime
import shutil

HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision

img_HWC2CHW = lambda x: x.permute(2, 0, 1)
gray2rgb = lambda x: x.unsqueeze(2).repeat(1, 1, 3)


to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
mse2psnr = lambda x: -10. * np.log(x+TINY_NUMBER) / np.log(10.)


def save_current_code(outdir):
    now = datetime.now()  # current date and time
    date_time = now.strftime("%m_%d-%H:%M:%S")
    src_dir = '.'
    dst_dir = os.path.join(outdir, 'code_{}'.format(date_time))
    shutil.copytree(src_dir, dst_dir,
                    ignore=shutil.ignore_patterns('data*', 'pretrained*', 'logs*', 'out*', '*.png', '*.mp4',
                                                  '*__pycache__*', '*.git*', '*.idea*', '*.zip', '*.jpg'))


def img2mse(x, y, mask=None):
    '''
    :param x: img 1, [(...), 3]
    :param y: img 2, [(...), 3]
    :param mask: optional, [(...)]
    :return: mse score
    '''
    if mask is None:
        return torch.mean((x - y) * (x - y))
    else:
        return torch.sum((x - y) * (x - y) * mask.unsqueeze(-1)) / (torch.sum(mask) * x.shape[-1] + TINY_NUMBER)


def img2psnr(x, y, mask=None):
    return mse2psnr(img2mse(x, y, mask).item())


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


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

