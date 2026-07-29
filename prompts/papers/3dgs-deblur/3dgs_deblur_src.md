# prompts/papers/3dgs-deblur/combine.py

``` py
"""Combine COLMAP poses with sai-cli velocities"""
import os
import json
import shutil

def process(input_folder, args):
    if args.override_calibration is None:
        override_calibration = None
    else:
        with open(args.override_calibration, 'rt') as f:
            calib_json = json.load(f)
        calib_json_cam0, = calib_json['cameras']
        override_calibration = calib_json_cam0
    
    name = os.path.basename(os.path.normpath(input_folder))
    print('name', name)
    SAI_INPUT_ROOT = 'data/inputs-processed/' + args.dataset

    def read_json(path):
        with open(path) as f:
            return json.load(f)

    if args.sai_input_folder is None:
        sai_folder = os.path.join(SAI_INPUT_ROOT, name)
    else:
        sai_folder = args.sai_input_folder

    if args.pose_opt_pass_dir is None:
        src_poses = read_json(os.path.join(input_folder, 'transforms.json'))
        image_folder = os.path.join(input_folder, 'images')
        ply_pc = os.path.join(input_folder, 'sparse_pc.ply')
    else:
        model_f = os.path.join(input_folder, args.model_name)
        input_json_path =  os.path.join(model_f, os.listdir(model_f)[0], 'transforms_train.json')
        src_poses = { 'frames': read_json(input_json_path) }
        image_folder = os.path.join(sai_folder, 'images')
        ply_pc = os.path.join(sai_folder, 'sparse_pc.ply')

    sai_poses = read_json(os.path.join(sai_folder, 'transforms.json'))

    src_poses_by_filename = { './images/' + os.path.basename(f['file_path']): f for f in src_poses['frames'] }
    if len(src_poses_by_filename) == 0:
        print('skipping: no source poses found')
        return

    # print([(k, src_poses_by_filename[k]['file_path']) for k in sorted(src_poses_by_filename.keys())])

    combined_frames = []

    import numpy as np
    frame_centers_sai = []
    frame_centers_src = []

    for sai_frame in sai_poses['frames']:
        id = sai_frame['file_path']
        if id.startswith('images'): id = './' + id
        frame = src_poses_by_filename.get(id, None)

        if frame is None:
            print('warning: could not find source pose for %s, skipping' % id)
            if not args.tolerate_missing: return
            continue
        # print('found frame', id)
        
        if 'transform' in frame:
            frame['transform_matrix'] = frame['transform']
            frame['transform_matrix'].append([0, 0, 0, 1])
            del frame['transform']

        frame['file_path'] = id

        frame_centers_sai.append(np.array(sai_frame['transform_matrix'])[:3, 3].tolist())
        frame_centers_src.append(np.array(frame['transform_matrix'])[:3, 3].tolist())

        for prop in ['camera_angular_velocity', 'camera_linear_velocity']:
            if prop in sai_frame:
                frame[prop] = sai_frame[prop]

        for prop in ['motion_blur_score']:
            if prop in sai_frame:
                frame[prop] = sai_frame[prop]

        for prop in ['colmap_im_id']:
            if prop in frame:
                del frame[prop]

        combined_frames.append(frame)

    # scale velocities to match COLMAP
    frame_centers_sai = np.array(frame_centers_sai)
    frame_centers_src = np.array(frame_centers_src)
    frame_centers_sai -= np.mean(frame_centers_sai, axis=0)
    frame_centers_src -= np.mean(frame_centers_src, axis=0)
    scale_factor = np.sqrt(np.sum(frame_centers_src**2)) / np.sqrt(np.sum(frame_centers_sai**2))
    print('scene scale factor %.12f' % scale_factor)
    if args.pose_opt_pass_dir is None: 
        print('scaling linear velocities')

        for frame in combined_frames:
            # only linear velocity should be scaled
            frame['camera_linear_velocity'] = [v * scale_factor for v in frame['camera_linear_velocity']]
    
    processed_prefix = 'data/inputs-processed'
    
    if args.pose_opt_pass_dir is not None:
        output_prefix = os.path.join(processed_prefix, args.dataset + '-2nd-pass')
        combined_poses = sai_poses

    elif args.keep_intrinsics or override_calibration is not None:
        combined_poses = sai_poses

        if override_calibration is not None:
            assert(override_calibration['model'] == 'brown-conrady')
            def write_to_calib(names, values):
                for i, n in enumerate(names):
                    combined_poses[n] = values[i]

            write_to_calib('k1 k2 p1 p2 k3'.split(), override_calibration['distortionCoefficients'][:5])
            write_to_calib('fl_x fl_y cx cy'.split(),  [override_calibration[c] for c in 'focalLengthX focalLengthY principalPointX principalPointY'.split()])

        if override_calibration is None and args.set_rolling_shutter_to is None:
            intrinsics_postfix = 'orig'
        else:
            intrinsics_postfix = 'calib'

        output_prefix = os.path.join(processed_prefix, 'colmap-' + args.dataset + '-' + intrinsics_postfix + '-intrinsics')
        combined_poses['applied_transform'] = src_poses['applied_transform']

        for prop in ['orientation_override', 'auto_scale_poses_override', 'fx', 'fy']:
            if prop in combined_poses:
                del combined_poses[prop]
    else:
        output_prefix = os.path.join(processed_prefix, 'colmap-' + args.dataset + '-vels')
        combined_poses = src_poses
        for prop in ['exposure_time', 'rolling_shutter_time']:
            if prop in sai_poses:
                combined_poses[prop] = sai_poses[prop]

    combined_poses['frames'] = combined_frames
    if args.set_rolling_shutter_to is not None:
        combined_poses['rolling_shutter_time'] = args.set_rolling_shutter_to

    if args.output_folder is None:
        output_folder = os.path.join(output_prefix, name)
    else:
        output_folder = args.output_folder

    print('Output folder: ' + output_folder)
    if not args.dry_run:
        if os.path.exists(output_folder): shutil.rmtree(output_folder)
        shutil.copytree(image_folder, os.path.join(output_folder, 'images'))
        # shutil.copytree(colmap_folder, os.path.join(output_folder, 'colmap'))
        shutil.copyfile(ply_pc, os.path.join(output_folder, 'sparse_pc.ply'))
        with open (os.path.join(output_folder, 'transforms.json'), 'w') as f:
            json.dump(combined_poses, f, indent=4)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("input_folder", type=str, default=None, nargs='?')
    parser.add_argument('sai_input_folder', default=None, nargs='?')
    parser.add_argument('output_folder', default=None, nargs='?')
    parser.add_argument('--dataset', default='sai-cli')
    parser.add_argument('--set_rolling_shutter_to', default=None, type=float)
    parser.add_argument('--keep_intrinsics', action='store_true')
    parser.add_argument('--tolerate_missing', action='store_true')
    parser.add_argument('--override_calibration', type=str, default=None)
    parser.add_argument('--pose_opt_pass_dir', type=str, default=None)
    parser.add_argument('--model_name', default='splatfacto')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--case_number', type=int, default=-1)
    args = parser.parse_args()

    if args.input_folder in ['all']:
        args.case_number = 0
        args.input_folder = None

    selected_cases = []

    if args.input_folder is None:
        if args.pose_opt_pass_dir is None:
            src_folder = 'data/inputs-processed/colmap-' + args.dataset + '-imgs'
        else:
            src_folder = args.pose_opt_pass_dir

        cases = [os.path.join(src_folder, f) for f in sorted(os.listdir(src_folder))]

        if args.case_number == -1:
            print('valid cases')
            for i, c in enumerate(cases): print(str(i+1) + ':\t' + c)
        elif args.case_number == 0:
            selected_cases = cases
        else:
            selected_cases = [cases[args.case_number - 1]]
    else:
        selected_cases = [args.input_folder]

    for case in selected_cases:
        print('Processing ' + case)
        process(case, args)


```

# prompts/papers/3dgs-deblur/download_data.py

``` py
"""Script to download processed datasets."""
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro


@dataclass
class DownloadProcessedData:
    save_dir: Path = Path(os.getcwd() + "/data")
    """Save directory. Default /data."""
    dataset: Literal["synthetic", "sai", "synthetic-raw", "sai-raw", "all"] = "synthetic"
    """Dataset download name. Set to 'synthetic' to download all synthetic data. Set to 'spectacular' for real world smartphone captures."""

    def main(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)

        urls = {
            "inputs-processed": {
                "synthetic-all": "https://zenodo.org/records/10847884/files/processed-nerfstudio.zip",
                "colmap-sai-cli-orig-intrinsics-blur-scored": "https://zenodo.org/records/10848124/files/colmap-sai-cli-orig-intrinsics-blur-scored.tar.xz",
                "colmap-sai-cli-calib-intrinsics-blur-scored": "https://zenodo.org/records/10848124/files/colmap-sai-cli-calib-intrinsics-blur-scored.tar.xz",
                "colmap-sai-cli-vels-blur-scored": "https://zenodo.org/records/10848124/files/colmap-sai-cli-vels-blur-scored.zip",
            },
            "inputs-raw": {
                "spectacular-rec": "https://zenodo.org/records/10848124/files/spectacular-rec.zip",
                "spectacular-rec-extras": "https://zenodo.org/records/10848124/files/spectacular-rec-extras.zip",
                "synthetic-raw": "https://zenodo.org/records/10847884/files/renders.zip"
            }
        }

        def download_dataset(dataset):
            for subfolder, sub_urls in urls.items():
                if dataset not in sub_urls: continue
                
                save_dir = self.save_dir / subfolder
                save_dir.mkdir(parents=True, exist_ok=True)
                download_command = ["wget", "-P", str(self.save_dir), sub_urls[dataset]]

                # download
                try:
                    subprocess.run(download_command, check=True)
                    print("File file downloaded succesfully.")
                except subprocess.CalledProcessError as e:
                    print(f"Error downloading file: {e}")

                file_name = Path(sub_urls[dataset]).name

                # subsubfolder for sai data
                subsubfolder = dataset if "sai" in file_name or subfolder == "inputs-raw" else ""
                if subsubfolder:
                    Path(self.save_dir / subfolder / subsubfolder).mkdir(
                        parents=True, exist_ok=True
                    )

                # deal with zip or tar formats
                if Path(sub_urls[dataset]).suffix == ".zip":
                    extract_command = [
                        "unzip",
                        self.save_dir / file_name,
                        "-d",
                        self.save_dir / Path(subfolder) / subsubfolder,
                    ]
                else:
                    extract_command = [
                        "tar",
                        "-xvJf",
                        self.save_dir / file_name,
                        "-C",
                        self.save_dir / Path(subfolder) / subsubfolder,
                    ]

                # extract
                try:
                    subprocess.run(extract_command, check=True)
                    os.remove(self.save_dir / file_name)
                    print("Extraction complete.")
                except subprocess.CalledProcessError as e:
                    print(f"Extraction failed: {e}")

        def download_dataset_by_short_name(dataset):
            if dataset == "synthetic":
                for dataset in urls["inputs-processed"].keys():
                    if "synthetic" in dataset:
                        download_dataset(dataset)
            elif dataset == "sai":
                for dataset in urls["inputs-processed"].keys():
                    if "sai" in dataset:
                        download_dataset(dataset)
            elif dataset == "synthetic-raw":
                download_dataset("synthetic-raw")

            elif dataset == "sai-raw":
                download_dataset("spectacular-rec")
                download_dataset("spectacular-rec-extras")

            else:
                raise NotImplementedError
            
        if self.dataset == "all":
            for ds in ["synthetic", "sai", "synthetic-raw", "sai-raw"]:
                download_dataset_by_short_name(ds)
        else:
            download_dataset_by_short_name(self.dataset)

if __name__ == "__main__":
    tyro.cli(DownloadProcessedData).main()


```

# prompts/papers/3dgs-deblur/parse_outputs.py

``` py
"""Parse output metrics from JSON files"""
import os
import json

def parse_metrics(metrics_path):
    with open(metrics_path) as f:
        return json.load(f)

def find_and_parse_directories_containing_splatting_metrics(root_dir):
    matching_dirs = []

    def parse_dir(dirpath, filename):
        run_name = dirpath[len(root_dir)+1:]
        dataset, _, rest = run_name.partition('/')

        rest_split = rest.split('/')
        if len(rest_split) != 4: return None
        variant, session, method, ts = rest_split
        if method != 'splatfacto': return None

        m = parse_metrics(os.path.join(dirpath, filename))

        d = {
            #'dataset': dataset[:1],
            'dataset': dataset,
            'variant': variant,
            'session': session,
            'path': dirpath,
            'time': m.get('wall_clock_time_seconds', -1)
        }

        
        for k, v in m['results'].items(): d[k] = v
        # print(d)
        return d

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # print(dirpath, filename)
            if filename == 'metrics.json':
                parsed = parse_dir(dirpath, filename)
                if parsed is not None:
                    matching_dirs.append(parsed)
                break

    return sorted(matching_dirs, key=lambda x: x['path'])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=str, nargs='?', default=None)
    parser.add_argument('-f', '--output_format', choices=['csv', 'txt'], default='txt')
    args = parser.parse_args()

    import pandas as pd
    pd.set_option("display.max_rows", None)
    df = pd.DataFrame(find_and_parse_directories_containing_splatting_metrics('data/outputs'))
    cols = 'dataset variant session psnr ssim lpips time'.split()
    df = df[cols]
    if args.dataset is not None:
        df = df[df['dataset'] == args.dataset].drop('dataset', axis=1)
        
    if args.output_format == 'csv':
        print(df.to_csv(index=False))
    elif args.output_format == 'txt':
        print(df)
    else:
        raise ValueError(f'Unknown format: {args.output_format}')

```

# prompts/papers/3dgs-deblur/process_deblur_nerf_inputs.py

``` py
"""Run COLMAP on a single sequence through Nerfstudio scripts"""
import os
import subprocess
import shutil
import tempfile
import json

from process_synthetic_inputs import generate_seed_points_match_and_triangulate

def process(input_folder, args, pass_no=1):

    name = os.path.basename(os.path.normpath(input_folder))

    # 'Wine' is 'Trolley' (see https://github.com/limacv/Deblur-NeRF/issues/39)
    out_name = name.replace('blur', '').replace('2', '').replace('wine', 'trolley')

    test_image_folder = None
    first_pass_folder = None
    input_image_folder = os.path.join(input_folder, 'images_1')

    if args.hloc:
        method = 'hloc'
    else:
        method = 'colmap'

    if args.dataset == 'synthetic_camera_motion_blur':
        paper = 'deblurnerf'
    if args.dataset == 'synthetic_release':
        paper = 'exblurf'
    elif args.dataset == 'nerf_llff_data':
        paper = 'bad-nerf'
    elif args.dataset == 'synthetic-mb':
        input_image_folder = os.path.join(input_folder, 'images')
        paper = 'sai-mb'
    elif args.dataset == 'synthetic-rs':
        input_image_folder = os.path.join(input_folder, 'images')
        paper = 'sai-rs'
    elif args.dataset == 'bad-nerf-gtK-colmap-nvs':
        # this data contains a fixed version of the Tanabata scene
        # where the wine trolley is in the same place in sharp and blurry images
        paper = 'bad-gaussians'
        input_image_folder = os.path.join(input_folder, 'images')
    elif args.dataset == 'colmap-bad-gaussians-synthetic-novel-view-deblurred-training':
        input_image_folder = os.path.join(input_folder, 'images')
        paper = 'mpr-deblurred'

    basename = method + '-' + paper + '-synthetic'

    if pass_no == 1:
        if args.use_all_images:
            dataset_name = basename + '-all'
        else:
            dataset_name = basename + '-novel-view-temp'
    elif pass_no == 2:
        first_pass_folder = os.path.join('data/inputs-processed/' + basename + '-novel-view-temp', out_name)
        dataset_name = basename + '-novel-view'
    elif pass_no == 3:
        dataset_name = basename + '-deblurring'
        input_image_folder = os.path.join(input_folder, 'images')
        test_image_folder = os.path.join(input_folder, 'images_test')
    else:
        assert False

    if pass_no != 1 or args.use_all_images:
        if args.exact_intrinsics:
            dataset_name += '-exact-intrinsics'
        if args.manual_point_cloud:
            dataset_name += '-manual-pc'

    output_folder = os.path.join('data/inputs-processed/' + dataset_name, out_name)

    temp_dir = tempfile.TemporaryDirectory()
    n = 0
    for index, f in enumerate(sorted(os.listdir(input_image_folder))):
        if 'depth' in f: continue
        if not args.dry_run:
            new_name = f
            if test_image_folder is not None:
                new_name = 'train_' + f
            if pass_no == 1 and index % 8 == 0 and not args.use_all_images:
                continue
            shutil.copyfile(os.path.join(input_image_folder, f), os.path.join(temp_dir.name, new_name))
        n += 1
    print('%d images (would be) copied in a temporary directory' % n)

    # Print the path to the temporary directory
    cmd = [
        'ns-process-data',
        'images',
        '--data', temp_dir.name,
        '--output-dir', output_folder,
        '--num-downscales', '1',
        '--matching-method', 'exhaustive',
        '--camera-type', 'simple_pinhole',
    ]

    if args.hloc:
        cmd.extend([
            '--feature-type', 'superpoint',
            '--matcher-type', 'superpoint+lightglue',
        ])

    if not args.post_process_only:
        print(cmd)
        if not args.dry_run:
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            subprocess.check_call(cmd)

    json_fn = os.path.join(output_folder, 'transforms.json')
    if os.path.exists(json_fn):
        with open(json_fn, 'r') as f:
            transforms = json.load(f)
    else:
        transforms = { 'frames': [] }
        assert args.dry_run

    if test_image_folder is not None:
        assert first_pass_folder is None

        test_images = sorted(os.listdir(test_image_folder))
        test_frames = []

        if not any('train_' in f['file_path'] for f in transforms['frames']):
            for index, frame in enumerate(sorted(transforms['frames'], key=lambda x: x['file_path'])):
                orig_fn = test_images[index]
                test_image_fn = 'eval_' + orig_fn
                test_image_path = 'images/' + test_image_fn

                if not args.dry_run:
                    shutil.copyfile(os.path.join(test_image_folder, orig_fn), os.path.join(output_folder, test_image_path))

                if 'train_' not in frame['file_path']:
                    train_path = 'images/train_' + orig_fn
                    if not args.dry_run:
                        shutil.move(os.path.join(output_folder, frame['file_path']), os.path.join(output_folder, train_path))
                    frame['file_path'] = train_path

                test_frame = { k: v for k, v in frame.items() }
                test_frame['file_path'] = test_image_path
                test_frames.append(test_frame)

            transforms['frames'].extend(test_frames)

    elif first_pass_folder is not None:
        with open(os.path.join(first_pass_folder, 'transforms.json'), 'r') as f:
            first_pass_transforms = json.load(f)

        import numpy as np
        to_pose_mat = lambda f : np.array(f['transform_matrix'])
        get_frame_idx = lambda f: int(f['file_path'].split('_')[-1].split('.')[0], base=10) - 1

        train_frame_c2ws = { get_frame_idx(f): to_pose_mat(f) for f in first_pass_transforms['frames'] }
        all_frames_c2ws = { get_frame_idx(f): to_pose_mat(f) for f in transforms['frames'] }

        combined_transforms = { k: v for k, v in first_pass_transforms.items() }
        combined_transforms['frames'] = []

        orig_index = 0
        for index, frame in enumerate(sorted(transforms['frames'], key=lambda x: x['file_path'])):
            #print(frame['file_path'])
            if index % 8 == 0:
                ref_frame = index - 1
                ref_frame_orig_index = orig_index - 1
                if ref_frame < 0:
                    ref_frame = index + 1
                    ref_frame_orig_index = orig_index # the next frame

                # print(index, orig_index, ref_frame, ref_frame_orig_index)

                pose_cur_pred_c2w = train_frame_c2ws[ref_frame_orig_index] @ np.linalg.inv(all_frames_c2ws[ref_frame]) @ all_frames_c2ws[index]
                frame['transform_matrix'] = pose_cur_pred_c2w.tolist()
            else:
                frame['transform_matrix'] = train_frame_c2ws[orig_index].tolist()
                orig_index += 1

            combined_transforms['frames'].append(frame)

        transforms = combined_transforms
        if not args.dry_run:
            shutil.copyfile(os.path.join(first_pass_folder, 'sparse_pc.ply'), os.path.join(output_folder, 'sparse_pc.ply'))

    if args.exact_intrinsics:
        KNOWN_INTRINSICS = {
            "w": 600,
            "h": 400,
            "cx": 300.0,
            "cy": 200.0,
            "fl_x": 541.8502321581475,
            "fl_y": 541.8502321581475,
            "k1": 0,
            "k2": 0,
            "p1": 0,
            "p2": 0,
        }
        for k, v in KNOWN_INTRINSICS.items():
            transforms[k] = v

    print('writing %s' % json_fn)
    if not args.dry_run:
        with open(json_fn, 'wt') as f:
            json.dump(transforms, f, indent=4)

    if pass_no == 1 and args.manual_point_cloud:
        if os.path.exists(output_folder):
            if not args.dry_run:
                backup_ply = os.path.join(output_folder, 'sparse_pc_colmap.ply')
                backup_json = os.path.join(output_folder, 'transforms_colmap.json')
                if not os.path.exists(backup_ply):
                    ply_fn = os.path.join(output_folder, 'sparse_pc.ply')
                    assert os.path.exists(ply_fn) and os.path.exists(json_fn)
                    shutil.copyfile(ply_fn, backup_ply)
                if not os.path.exists(backup_json):
                    shutil.copyfile(json_fn, backup_json)
            generate_seed_points_match_and_triangulate(output_folder, dry_run=args.dry_run, visualize=args.dry_run)
        else:
            assert args.dry_run
    
    temp_dir.cleanup()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("input_folder", type=str, default=None, nargs='?')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--dataset', default='synthetic_camera_motion_blur')
    parser.add_argument('--post_process_only', action='store_true')
    parser.add_argument('--manual_point_cloud', action='store_true')
    parser.add_argument('--deblurring_version', action='store_true')
    parser.add_argument('--exact_intrinsics', action='store_true')
    parser.add_argument('--hloc', action='store_true')
    parser.add_argument('--use_all_images', action='store_true',
                        help='Use both blurry training and sharp test images for training pose registration')
    parser.add_argument('--case_number', type=int, default=-1)
    
    args = parser.parse_args()

    if args.input_folder in ['all']:
        args.case_number = 0
        args.input_folder = None
        
    selected_cases = []
    misc = False

    if args.dataset.endswith('/'): args.dataset = args.dataset[:-1]

    if args.input_folder is None:
        sai_dataset = args.dataset.startswith('synthetic-')

        if sai_dataset:
            input_root = os.path.join('data/inputs-processed/', args.dataset)
        else:
            input_root = os.path.join('data/inputs-raw/', args.dataset)
        cases = [os.path.join(input_root, f)
            for f in sorted(os.listdir(input_root))
            if f.startswith('blur') or sai_dataset or args.dataset == 'colmap-bad-gaussians-synthetic-novel-view-deblurred-training'
        ]

        if args.case_number == -1:
            print('valid cases')
            for i, c in enumerate(cases): print(str(i+1) + ':\t' + c)
        elif args.case_number == 0:
            selected_cases = cases
        else:
            selected_cases = [cases[args.case_number - 1]]
    else:
        selected_cases = [args.input_folder]

    for case in selected_cases:
        print('Processing ' + case)
        process(case, args)
        if not args.use_all_images:
            if args.deblurring_version:
                process(case, args, pass_no=3)
            else:
                process(case, args, pass_no=2)


```

# prompts/papers/3dgs-deblur/process_sai_custom.py

``` py
"""Process a single custom SAI input"""
import os
import subprocess
import shutil
import json
import tempfile

from process_sai_inputs import SAI_CLI_PROCESS_PARAMS

DEFAULT_OUT_FOLDER = 'data/inputs-processed/custom'

def ensure_exposure_time(target, input_folder):
    trans_fn = os.path.join(target, 'transforms.json')
    with open(trans_fn) as f:
        transforms = json.load(f)
    
    if 'exposure_time' in transforms: return

    with open(os.path.join(input_folder, 'data.jsonl')) as f:
        for line in f:
            d = json.loads(line)
            if 'frames' in d:
                e = d['frames'][0].get('exposureTimeSeconds', None)
                if e is not None:
                    print('got exposure time %g from data.jsonl' % e)
                    transforms['exposure_time'] = e
                    with open(trans_fn, 'wt') as f:
                        json.dump(transforms, f, indent=4)
                    return
    
    raise RuntimeError("no exposure time available")

def process(args):
    def maybe_run_cmd(cmd):
        print('COMMAND:', cmd)
        if not args.dry_run: subprocess.check_call(cmd)

    def maybe_unzip(fn):
        name = os.path.basename(fn)
        if name.endswith('.zip'):
            name = name[:-4]
            tempdir = tempfile.mkdtemp()
            input_folder = os.path.join(tempdir, 'recording')
            extract_command = [
                "unzip",
                fn,
                "-d",
                input_folder,
            ]
            maybe_run_cmd(extract_command)
            if not args.dry_run:
                # handle folder inside zip
                for f in os.listdir(input_folder):
                    if f == name:
                        input_folder = os.path.join(input_folder, f)
                        break
        else:
            input_folder = fn
        
        return name, input_folder

    sai_params = json.loads(json.dumps(SAI_CLI_PROCESS_PARAMS))
    sai_params['key_frame_distance'] = args.key_frame_distance

    tempdir = None
    name, input_folder = maybe_unzip(args.spectacular_rec_input_folder_or_zip)

    sai_params_list = []
    for k, v in sai_params.items():
        if k == 'internal':
            for k2, v2 in v.items():
                sai_params_list.append(f'--{k}={k2}:{v2}')
        else:
            if v is None:
                sai_params_list.append(f'--{k}')
            else:
                sai_params_list.append(f'--{k}={v}')
        
    result_name = name

    if args.output_folder is None:
        final_target = os.path.join(DEFAULT_OUT_FOLDER, result_name)
    else:
        final_target = args.output_folder

    if not args.skip_colmap:
        if tempdir is None: tempdir = tempfile.mkdtemp()
        target = os.path.join(tempdir, 'sai-cli', result_name)
    else:
        target = final_target

    cmd = [
        'sai-cli', 'process',
        input_folder,
        target
    ] + sai_params_list

    if args.preview:
        cmd.extend(['--preview', '--preview3d'])

    if os.path.exists(target): shutil.rmtree(target)
    maybe_run_cmd(cmd)
    if not args.dry_run: ensure_exposure_time(target, input_folder)

    if not args.skip_colmap:
        colmap_target = os.path.join(tempdir, 'colmap-sai-cli-imgs', result_name)
        colmap_cmd = [
            'python', 'run_colmap.py',
            target,
            colmap_target
        ]
        maybe_run_cmd(colmap_cmd)
        
        combine_cmd = [
            'python', 'combine.py',
            colmap_target,
            target,
            final_target,
            '--tolerate_missing'
        ]
        if args.keep_intrinsics:
            combine_cmd.append('--keep_intrinsics')
        
        if os.path.exists(final_target): shutil.rmtree(final_target)
        maybe_run_cmd(combine_cmd)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spectacular_rec_input_folder_or_zip", type=str)
    parser.add_argument("output_folder", type=str, default=None, nargs='?')
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--skip_colmap', action='store_true')
    parser.add_argument('--keep_intrinsics', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--key_frame_distance', type=float, default=0.1,
        help="Minimum key frame distance in meters, default (0.1), increase for larger scenes")
    args = parser.parse_args()

    process(args)


```

# prompts/papers/3dgs-deblur/process_sai_inputs.py

``` py
"""Process raw input data to the main benchmark format"""
import os
import subprocess
import shutil
import json

SAI_CLI_PROCESS_PARAMS = {
    'image_format': 'png',
    'no_undistort': None,
    'key_frame_distance': 0.1,
    'internal': {
        'maxKeypoints': 2000,
        'optimizerMaxIterations': 50,
    }
}

DATASET_SPECIFIC_PARAMETERS = {}

def process_subfolders(spec, output_folder, method='sai', only_this_case_number=None, dry_run=False, preview=False):
    def process(folder, counter, prefix, named):
        if named:
            name = os.path.basename(folder)
        else:
            name = "%02d" % counter
        
        if prefix is not None:
            name = prefix + '-' + name

        sai_params = json.loads(json.dumps(SAI_CLI_PROCESS_PARAMS)) # deep copy
        out_dataset_folder = output_folder
        if args.no_blur_score_filter:
            out_dataset_folder += '-no-blur-select'
            sai_params['blur_filter_range'] = 0
            sai_params['internal']['keyFrameCandidateSelectionBufferSize'] = 1

        for k, v in DATASET_SPECIFIC_PARAMETERS.get(prefix, {}).items():
            if k == 'internal':
                for k2, v2 in v.items():
                    sai_params['internal'][k2] = v2
            else:
                sai_params[k] = v

        sai_params_list = []
        for k, v in sai_params.items():
            if k == 'internal':
                for k2, v2 in v.items():
                    sai_params_list.append(f'--{k}={k2}:{v2}')
            else:
                if v is None:
                    sai_params_list.append(f'--{k}')
                else:
                    sai_params_list.append(f'--{k}={v}')
            
        target = os.path.join(out_dataset_folder, name.replace('_', '-').replace('-capture', ''))

        if method == 'sai':
            cmd = [
                'sai-cli', 'process',
                folder,
                target
            ] + sai_params_list

            if preview:
                cmd.extend(['--preview', '--preview3d'])

        elif method == 'colmap-video':
            [
                'ns-process-data',
                'video',
                '--data', os.path.join(folder, 'data.mp4'),
                '--output-dir', target
            ]
        else:
            assert(False)

        if dry_run:
            print(cmd)
            return
        print(f"Processing: {folder} -> {target}")

        if os.path.exists(target): shutil.rmtree(target)
        subprocess.check_call(cmd)

    counter = 1
    for (base_folder, prefix, named) in spec:
        items = os.listdir(base_folder)
        directories = sorted([item for item in items if os.path.isdir(os.path.join(base_folder, item))])

        dir_counter = 1
        # Loop through each directory and run a command
        for directory in directories:
            full_path = os.path.join(base_folder, directory)
            if only_this_case_number is None or only_this_case_number == counter:
                print('case %d: %s' % (counter, full_path))
                process(full_path, dir_counter, prefix, named)
            counter += 1
            dir_counter += 1

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case_number", type=int, default=None)
    parser.add_argument('--method', choices={'sai', 'colmap-video'}, default='sai')
    parser.add_argument('--no_blur_score_filter', action='store_true')
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    if args.method == 'sai':
        out_folder ='data/inputs-processed/sai-cli'
    elif args.method == 'colmap-video':
        out_folder ='data/inputs-processed/colmap-video'
    else:
        assert(False)

    process_subfolders([
            ('data/inputs-raw/spectacular-rec', None, True),
        ],
        out_folder,
        method=args.method,
        only_this_case_number=args.case_number,
        dry_run=args.dry_run,
        preview=args.preview)


```

# prompts/papers/3dgs-deblur/process_synthetic_inputs.py

``` py
"""Process raw synthetic input data to the main benchmark format"""
import os
import json
import shutil
import cv2
import numpy as np

POSE_POSITION_NOISE_REL = 0.05
POSE_ORIENTATION_NOISE_DEG = 1

INTRINSIC_NOISE_REL = 0.01

def rotation_matrix_to_rotvec(R):
    # Using a proven/stable algorithm. Other options are sketchy for small rotation
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()

def quaternion_to_rotation_matrix(q_wxyz):
    q = q_wxyz
    return np.array([
        [q[0]*q[0]+q[1]*q[1]-q[2]*q[2]-q[3]*q[3], 2*q[1]*q[2] - 2*q[0]*q[3], 2*q[1]*q[3] + 2*q[0]*q[2]],
        [2*q[1]*q[2] + 2*q[0]*q[3], q[0]*q[0] - q[1]*q[1] + q[2]*q[2] - q[3]*q[3], 2*q[2]*q[3] - 2*q[0]*q[1]],
        [2*q[1]*q[3] - 2*q[0]*q[2], 2*q[2]*q[3] + 2*q[0]*q[1], q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3]]
    ])

def deterministic_uniform_rand_generator(seed=1000):
    """
    A simple pseudorandom number generator that returns the
    same random sequence on all machines. The quality of these
    random numbers is low but this is fine for this particular
    application.
    """

    # see https://en.cppreference.com/w/cpp/numeric/random/linear_congruential_engine

    a, c, m = 48271, 0, 2147483647
    x = seed + 1
    uniform_steps = 999

    while True:
        x = (a * x + c) % m
        yield float(x % uniform_steps) / uniform_steps

def process(data_path, target, noisy_poses=False, noisy_intrinsics=False):
    """
    # --- Based on
    # https://github.com/limacv/Deblur-NeRF/blob/766ca3cfafa026ea45f75ee1d3186ec3d9e13d99/scripts/synthe2poses.py
    # and used under the following license

    MIT License

    Copyright (c) 2020 bmild

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    print(f"Processing: {data_path} -> {target}")
    if os.path.exists(target): shutil.rmtree(target)

    input_path = data_path
    json_path = os.path.join(input_path, "transforms.json")
    out_path = os.path.join(target, "images")
    converted_json_path = os.path.join(target, "transforms.json")
    os.makedirs(out_path, exist_ok=True)

    rand = deterministic_uniform_rand_generator()
    def rand3():
        nonlocal rand
        return np.array([next(rand) for _ in range(3)]) * 2 - 1

    def convert_pose_c2w(pose, scaling):
        pose = np.array(pose)
        pose[:3, :] *= scaling
        return pose

    def get_scaling(m):
        return 1.0 / np.sqrt((m[:3,:3].transpose() @ m[:3,:3])[0,0])

    with open(json_path, 'r') as metaf:
        meta = json.load(metaf)
        frames_data = meta["frames"]
        fov = meta["fov"]
        h, w = meta['h'], meta['w']
        exposure_time = meta["exposure_time"]
        rolling_shutter_time = meta["rolling_shutter_time"]

    focal_length = w / 2 / np.tan(fov / 2)

    if noisy_intrinsics:
        # slight (fixed) error in intrinsics
        intrinsic_noisy_scaling_x = 1 + INTRINSIC_NOISE_REL
        intrinsic_noisy_scaling_y = 1 - INTRINSIC_NOISE_REL
    else:
        intrinsic_noisy_scaling_x = 1
        intrinsic_noisy_scaling_y = 1

    converted_meta = {
        "aabb_scale": 16,
        "w": w,
        "h": h,
        "cx": w/2,
        "cy": h/2,
        "orientation_override": "none",
        "exposure_time": exposure_time,
        "rolling_shutter_time": rolling_shutter_time,
        "fl_x": focal_length * intrinsic_noisy_scaling_x,
        "fl_y": focal_length * intrinsic_noisy_scaling_y,
        "k1": 0,
        "k2": 0,
        "p1": 0,
        "p2": 0,
        "frames": []
    }

    scaling = None

    cam_positions = []

    for frame_data in frames_data:
        pose = np.array(frame_data["transform_matrix"])
        if scaling is None:
            scaling = get_scaling(pose)
        pose = convert_pose_c2w(pose, scaling)
        cam_positions.append(pose[:3, 3])
        img_path = os.path.join(data_path, frame_data["filename"])
        img_name = os.path.basename(img_path)
        img_out = os.path.join(out_path, img_name)

        if frame_data["blurcount"] == 0:
            img = cv2.imread(img_path)
            cv2.imwrite(img_out, img)

            velocity_cam = np.array([0, 0, 0])
            ang_vel_cam = np.array([0, 0, 0])
        else:
            img = cv2.imread(img_path)
            blur_poses = []
            for bluri in range(frame_data["blurcount"]):
                blur_poses.append(convert_pose_c2w(frame_data['blur_matrices'][bluri], scaling))

            velocity_w = (blur_poses[-1][:3, 3] - blur_poses[0][:3, 3]) / (exposure_time + rolling_shutter_time)
            rot = blur_poses[-1][:3, :3] @ blur_poses[0][:3, :3].transpose()
            rot_vec = rotation_matrix_to_rotvec(rot)
            # print(rot, rot_vec, np.linalg.norm(rot_vec))
            ang_vel_w = rot_vec / (exposure_time + rolling_shutter_time)

            R_w2c = pose[:3, :3].transpose()
            velocity_cam = R_w2c @ velocity_w
            ang_vel_cam = R_w2c @ ang_vel_w
            # print(velocity_cam, ang_vel_cam)
            cv2.imwrite(img_out, img)

        print(f"frame {img_name} saved!")

        converted_meta["frames"].append({
            "camera_linear_velocity": velocity_cam.tolist(),
            "camera_angular_velocity": ang_vel_cam.tolist(),
            "file_path": f"./images/{img_name}",
            "transform_matrix": pose.tolist()
        })
    
    if noisy_poses:
        center = np.mean(cam_positions, axis=0)
        scene_motion_scale = np.max(np.linalg.norm(cam_positions - center, axis=1))
        pos_noise_scale = POSE_POSITION_NOISE_REL * scene_motion_scale
        print('center point of scene cameras %s scale %g, pose noise scale +-%g' % (
            str(center.tolist()),
            scene_motion_scale,
            pos_noise_scale))
        for f in converted_meta['frames']:
            pose = np.array(f['transform_matrix'])
            pose[:3, 3] + rand3() * pos_noise_scale
            noise_ang = 0
            while noise_ang < 1e-6:
                noise_rot_vec = rand3() * POSE_ORIENTATION_NOISE_DEG / 180.0 * np.pi
                noise_ang = np.linalg.norm(noise_rot_vec)

            noise_rot_dir = noise_rot_vec / noise_ang
            noise_quat = [np.cos(noise_ang*0.5)] + (np.sin(noise_ang*0.5) * noise_rot_dir).tolist()
            noise_R = quaternion_to_rotation_matrix(noise_quat)
            pose[:3, :3] = pose[:3, :3] @ noise_R
            f['transform_matrix'] = pose.tolist()

    with open(converted_json_path, 'wt') as f:
        json.dump(converted_meta, f, indent=4)

def point_cloud_to_ply(xyzrgbs, out_fn):
    with open(out_fn, 'wt') as f:
        f.write('\n'.join([
            'ply',
            'format ascii 1.0',
            'element vertex %d' % len(xyzrgbs),
            'property float x',
            'property float y',
            'property float z',
            'property uint8 red',
            'property uint8 green',
            'property uint8 blue',
            'end_header'
        ]) + '\n')
        for r in xyzrgbs:
            for i in range(3): r[i+3] = int(r[i+3])
            f.write(' '.join([str(v) for v in r]) + '\n')

def triangulate_point(o1, d1, o2, d2):
    A = np.stack([d1, -d2]).T
    b = o2 - o1
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    P1 = o1 + x[0] * d1
    P2 = o2 + x[1] * d2
    P = (P1 + P2) / 2
    return P

def reproject_point(p, c2w, intrinsics):
    p_cam = c2w[:3, :3].transpose() @ (p - c2w[:3, 3])
    MIN_D = 1e-6

    if -p_cam[2] <= MIN_D: return None

    p_img = p_cam[:2] / -p_cam[2]
    p_px = [p_img[0] * intrinsics['fl_x'] + intrinsics['cx'], -p_img[1] * intrinsics['fl_y'] + intrinsics['cy']]
    return p_px

def reprojection_error(p_reproj, p_orig):
    if p_reproj is None: return 1e6
    return np.linalg.norm(p_reproj - np.array(p_orig))

def triangulate(points1, points2, c2w_i, c2w_j, matches, intrinsics, reprojection_error_pixels):
    filtered_matches = []
    points3d = []
    rejected_matches = []

    for match in matches:
        i, j = match.queryIdx, match.trainIdx

        def to_dir(p):
            px = (p[0] - intrinsics['cx']) / intrinsics['fl_x']
            py = -(p[1] - intrinsics['cy']) / intrinsics['fl_y']
            h = [px, py, -1]
            return np.array(h) / np.linalg.norm(h)

        p1 = points1[i].pt
        p2 = points2[j].pt

        dir_i_cam = to_dir(p1)
        dir_j_cam = to_dir(p2)

        dir_i = c2w_i[:3, :3] @ dir_i_cam
        dir_j = c2w_j[:3, :3] @ dir_j_cam

        P = triangulate_point(c2w_i[:3, 3], dir_i, c2w_j[:3, 3], dir_j)

        rp1 = reproject_point(P, c2w_i, intrinsics)
        rp2 = reproject_point(P, c2w_j, intrinsics)

        err = max(
            reprojection_error(rp1, p1),
            reprojection_error(rp2, p2))

        if err > reprojection_error_pixels:
            rejected_matches.append((match, rp1, rp2))
            continue

        filtered_matches.append(match)
        points3d.append(P)

    return filtered_matches, points3d, rejected_matches

def generate_seed_points_match_and_triangulate(target, visualize=False, dry_run=False, reprojection_error_pixels=10):
    json_path = os.path.join(target, "transforms.json")
    def is_eval_frame(i, frame):
        if i % 8 == 0:
            if 'camera_linear_velocity' in frame:
                vel = np.linalg.norm(frame['camera_linear_velocity']) + np.linalg.norm(frame['camera_angular_velocity'])
                assert(vel == 0)
            return True
        return False

    with open(json_path, 'rt') as f: transforms = json.load(f)
    training_frames = [f for i, f in enumerate(sorted(transforms['frames'], key=lambda fr: fr['file_path'])) if not is_eval_frame(i, f)]

    transforms['ply_file_path'] = './sparse_pc.ply'
    converted_json = transforms

    images = [cv2.imread(os.path.join(target, frame['file_path'])) for frame in training_frames]

    # --- By ChatGPT
    def find_keypoints_and_descriptors(images, detector):
        """Find keypoints and descriptors for each image using the given detector."""
        keypoints_and_descriptors = []
        for image in images:
            keypoints, descriptors = detector.detectAndCompute(image, None)
            keypoints_and_descriptors.append((keypoints, descriptors))
        return keypoints_and_descriptors

    def match_descriptors_and_triangulate(descriptor_pairs, matcher, frames, intrinsics):
        """Match descriptors between all pairs of images."""
        matches = {}
        n = len(descriptor_pairs)
        for i in range(n):
            for j in range(i+1, n):
                matches_ij = matcher.match(descriptor_pairs[i][1], descriptor_pairs[j][1])
                matches_ij = sorted(matches_ij, key=lambda x: x.distance)

                c2w_i = np.array(frames[i]['transform_matrix'])
                c2w_j = np.array(frames[j]['transform_matrix'])

                matches_ij, points3d, rejected_matches = triangulate(
                    descriptor_pairs[i][0],
                    descriptor_pairs[j][0],
                    c2w_i, c2w_j,
                    matches_ij, intrinsics, reprojection_error_pixels)
                matches[(i, j)] = (matches_ij, points3d, rejected_matches)

        return matches

    def visualize_matches(images, keypoints_and_descriptors, matches, pair):
        """Visualize the matches for a specific pair of images."""
        img1, img2 = images[pair[0]], images[pair[1]]
        kp1, kp2 = keypoints_and_descriptors[pair[0]][0], keypoints_and_descriptors[pair[1]][0]
        matches_ij, points, rejected_matches = matches[pair]
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches_ij, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        for rm in rejected_matches:
            match, rp1, rp2 = rm
            p1_orig = tuple(map(int, kp1[match.queryIdx].pt))
            p2_orig_x, p2_orig_y = tuple(map(int, kp2[match.trainIdx].pt))
            p2_orig_x += img1.shape[1]
            p2_orig = (p2_orig_x, p2_orig_y)
            cv2.circle(img_matches, p1_orig, 3, (0, 0, 255), 1)
            cv2.circle(img_matches, p2_orig, 3, (0, 0, 255), 1)
            if rp1 is not None:
                cv2.line(img_matches, p1_orig, tuple(map(int, rp1)), (0, 0, 255), 1)
            if rp2 is not None:
                rp2_x, rp2_y = tuple(map(int, rp2))
                cv2.line(img_matches, p2_orig, (rp2_x + img1.shape[1], rp2_y), (0, 0, 255), 1)
        cv2.imshow(f"Matches between image {pair[0]} and {pair[1]}", img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    detector = cv2.SIFT_create()
    print('finding keypoints and descriptors...')
    keypoints_and_descriptors = find_keypoints_and_descriptors(images, detector)
    print('matching descriptors...')
    #bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = match_descriptors_and_triangulate(keypoints_and_descriptors, bf_matcher, training_frames, transforms)
    if visualize:
        visualize_matches(images, keypoints_and_descriptors, matches, (0, 1))

    xyzrgbs = []
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            matches_ij, points, rejected_matches = matches[(i, j)]
            for (k, match) in enumerate(matches_ij):
                p = points[k]
                kp1 = keypoints_and_descriptors[i][0][match.queryIdx].pt
                color = images[i][int(kp1[1]), int(kp1[0]), [2, 1, 0]]
                xyzrgbs.append(p.tolist() + color.tolist())
    print('Triangulated %d points' % len(xyzrgbs))

    if not dry_run:
        with open(json_path, 'wt') as f:
            json.dump(converted_json, f, indent=4)

        seed_ply_path = os.path.join(target, "sparse_pc.ply")
        point_cloud_to_ply(xyzrgbs, seed_ply_path)

def process_dataset_folder(
        base_folder, 
        output_folder,
        subfolder,
        points_only=False,
        noisy_poses=False,
        noisy_intrinsics=False,
        dry_run=False,
        visualize=False):
    items = os.listdir(base_folder)
    directories = sorted([item for item in items if os.path.isdir(os.path.join(base_folder, item))])

    for directory in directories:
        print(directory)
        full_path = os.path.join(base_folder, directory, subfolder)
        if not os.path.exists(full_path): continue
        out_path = os.path.join(output_folder, directory)
        if not points_only and not dry_run:
            process(full_path, out_path, noisy_poses=noisy_poses, noisy_intrinsics=noisy_intrinsics)
        if os.path.exists(out_path):
            generate_seed_points_match_and_triangulate(out_path, visualize=visualize)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--points_only', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    process_dataset_folder(
        'data/inputs-raw/synthetic-raw',
        'data/inputs-processed/synthetic-posenoise',
        subfolder='raw_clear',
        noisy_poses=True,
        **vars(args))

    process_dataset_folder(
        'data/inputs-raw/synthetic-raw',
        'data/inputs-processed/synthetic-rs',
        subfolder='raw_rs',
        **vars(args))

    process_dataset_folder(
        'data/inputs-raw/synthetic-raw',
        'data/inputs-processed/synthetic-mb',
        subfolder='raw_mb',
        **vars(args))

    process_dataset_folder(
        'data/inputs-raw/synthetic-raw',
        'data/inputs-processed/synthetic-mb-posenoise',
        subfolder='raw_mb',
        noisy_poses=True,
        **vars(args))

    process_dataset_folder(
        'data/inputs-raw/synthetic-raw',
        'data/inputs-processed/synthetic-clear',
        subfolder='raw_clear',
        **vars(args))

    process_dataset_folder(
        'data/inputs-raw/synthetic-raw',
        'data/inputs-processed/synthetic-mbrs',
        subfolder='raw_mbrs',
        **vars(args))

    process_dataset_folder(
        'data/inputs-raw/synthetic-raw',
        'data/inputs-processed/synthetic-mbrs-posenoise',
        subfolder='raw_mbrs',
        noisy_poses=True,
        **vars(args))

    process_dataset_folder(
        'data/inputs-raw/synthetic-raw',
        'data/inputs-processed/synthetic-mbrs-pose-calib-noise',
        subfolder='raw_mbrs',
        noisy_poses=True,
        noisy_intrinsics=True,
        **vars(args))

```

# prompts/papers/3dgs-deblur/render_model.py

``` py
"""Load g model and render all outputs to disc"""

from dataclasses import dataclass
from pathlib import Path
import torch
import tyro
import os
import numpy as np
import shutil

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils import colormaps
from nerfstudio.data.datasets.base_dataset import InputDataset
from PIL import Image
from torch import Tensor

from typing import List, Literal, Optional, Union

def save_img(image, image_path, verbose=True) -> None:
    """helper to save images

    Args:
        image: image to save (numpy, Tensor)
        image_path: path to save
        verbose: whether to print save path

    Returns:
        None
    """
    if image.shape[-1] == 1 and torch.is_tensor(image):
        image = image.repeat(1, 1, 3)
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy() * 255
        image = image.astype(np.uint8)
    if not Path(os.path.dirname(image_path)).exists():
        Path(os.path.dirname(image_path)).mkdir(parents=True)
    im = Image.fromarray(image)
    if verbose:
        print("saving to: ", image_path)
    im.save(image_path)

# Depth Scale Factor m to mm
SCALE_FACTOR = 0.001
SAVE_RAW_DEPTH = False

def save_depth(depth, depth_path, verbose=True, scale_factor=SCALE_FACTOR) -> None:
    """helper to save metric depths

    Args:
        depth: image to save (numpy, Tensor)
        depth_path: path to save
        verbose: whether to print save path
        scale_factor: depth metric scaling factor

    Returns:
        None
    """
    if torch.is_tensor(depth):
        depth = depth.float() / scale_factor
        depth = depth.detach().cpu().numpy()
    else:
        depth = depth / scale_factor
    if not Path(os.path.dirname(depth_path)).exists():
        Path(os.path.dirname(depth_path)).mkdir(parents=True)
    if verbose:
        print("saving to: ", depth_path)
    np.save(depth_path, depth)

def save_outputs_helper(
    rgb_out: Optional[Tensor],
    gt_img: Optional[Tensor],
    depth_color: Optional[Tensor],
    depth_gt_color: Optional[Tensor],
    depth_gt: Optional[Tensor],
    depth: Optional[Tensor],
    normal_gt: Optional[Tensor],
    normal: Optional[Tensor],
    render_output_path: Path,
    image_name: Optional[str],
) -> None:
    """Helper to save model rgb/depth/gt outputs to disk

    Args:
        rgb_out: rgb image
        gt_img: gt rgb image
        depth_color: colored depth image
        depth_gt_color: gt colored depth image
        depth_gt: gt depth map
        depth: depth map
        render_output_path: save directory path
        image_name: stem of save name

    Returns:
        None
    """
    if image_name is None:
        image_name = ""

    if rgb_out is not None and gt_img is not None:
        # easier consecutive compare
        save_img(rgb_out, os.getcwd() + f"/{render_output_path}/{image_name}_pred.png", False)
        save_img(gt_img, os.getcwd() + f"/{render_output_path}/{image_name}_gt.png", False)

    if depth_color is not None:
        save_img(
            depth_color,
            os.getcwd()
            + f"/{render_output_path}/pred/depth/colorised/{image_name}.png",
            False,
        )
    if depth_gt_color is not None:
        save_img(
            depth_gt_color,
            os.getcwd() + f"/{render_output_path}/gt/depth/colorised/{image_name}.png",
            False,
        )
    if depth_gt is not None:
        # save metric depths
        save_depth(
            depth_gt,
            os.getcwd() + f"/{render_output_path}/gt/depth/raw/{image_name}.npy",
            False,
        )

    if SAVE_RAW_DEPTH:
        if depth is not None:
            save_depth(
                depth,
                os.getcwd() + f"/{render_output_path}/pred/depth/raw/{image_name}.npy",
                False,
            )

    if normal is not None:
        save_normal(
            normal,
            os.getcwd() + f"/{render_output_path}/pred/normal/{image_name}.png",
            verbose=False,
        )

    if normal_gt is not None:
        save_normal(
            normal_gt,
            os.getcwd() + f"/{render_output_path}/gt/normal/{image_name}.png",
            verbose=False,
        )

@dataclass
class RenderModel:
    """Render outputs of a GS model."""

    load_config: Path = Path("outputs/")
    """Path to the config YAML file."""
    output_dir: Path = Path("./data/renders/")
    """Path to the output directory."""
    set: Literal["train", "eval"] = "eval"
    """Dataset to test with (train or eval)"""
    output_same_dir: bool = True
    """Output to the subdirectory of the load_config path"""

    def main(self):
        if self.output_same_dir:
            self.output_dir = os.path.join(os.path.dirname(self.load_config), 'renders')

        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        print('writing %s' % str(self.output_dir))

        _, pipeline, _, _ = eval_setup(self.load_config)

        assert isinstance(pipeline.model, SplatfactoModel)

        model: SplatfactoModel = pipeline.model
        dataset: InputDataset

        with torch.no_grad():
            if self.set == "train":
                dataset = pipeline.datamanager.train_dataset
                images = pipeline.datamanager.cached_train
            elif self.set == "eval":
                dataset = pipeline.datamanager.eval_dataset
                images = pipeline.datamanager.cached_eval
            else:
                raise RuntimeError("Invalid set")
        
            cameras: Cameras = dataset.cameras  # type: ignore
            for image_idx in range(len(dataset)):  # type: ignore
                data = images[image_idx]

                # process batch gt data
                mask = None
                if "mask" in data:
                    mask = data["mask"]

                gt_img = 256 - data["image"] # not sure why negative
                if "sensor_depth" in data:
                    depth_gt = data["sensor_depth"]
                    depth_gt_color = colormaps.apply_depth_colormap(
                        data["sensor_depth"]
                    )
                else:
                    depth_gt = None
                    depth_gt_color = None
                if "normal" in data:
                    normal_gt = data["normal"]
                else:
                    normal_gt = None

                # process pred outputs
                camera = cameras[image_idx : image_idx + 1].to("cpu")
                #if self.set == "train":
                # camera idx is used to fetch camera optimizer adjustments
                # and should not be used for 'eval' data
                camera.metadata['cam_idx'] = image_idx
                outputs = model.get_outputs_for_camera(camera=camera)

                rgb_out, depth_out = outputs["rgb"], outputs["depth"]

                normal = None
                if "normal" in outputs:
                    normal = outputs["normal"]

                seq_name = Path(dataset.image_filenames[image_idx])
                image_name = f"{seq_name.stem}"

                depth_color = colormaps.apply_depth_colormap(depth_out)
                depth = depth_out.detach().cpu().numpy()

                if mask is not None:
                    rgb_out = rgb_out * mask
                    gt_img = gt_img * mask
                    if depth_color is not None:
                        depth_color = depth_color * mask
                    if depth_gt_color is not None:
                        depth_gt_color = depth_gt_color * mask
                    if depth_gt is not None:
                        depth_gt = depth_gt * mask
                    if depth is not None:
                        depth = depth * mask
                    if normal_gt is not None:
                        normal_gt = normal_gt * mask
                    if normal is not None:
                        normal = normal * mask

                # save all outputs
                save_outputs_helper(
                    rgb_out,
                    gt_img,
                    depth_color,
                    depth_gt_color,
                    depth_gt,
                    depth,
                    normal_gt,
                    normal,
                    self.output_dir,
                    image_name,
                )


if __name__ == "__main__":
    tyro.cli(RenderModel).main()


```

# prompts/papers/3dgs-deblur/render_video.py

``` py
"""Generate demo video camera trajectory"""
import os
import json
import subprocess
import numpy as np

class SplineInterpolator:        
    def __init__(self, target, frames_per_transition):
        self.target = target
        self.positions = []
        self.orientations = []
        self.loop = False
        self.tension = 0.0
        self.model_frame = None
        self.frames_per_transition = frames_per_transition

    def push(self, frame):
        from scipy.spatial.transform import Rotation
        m = np.array(frame['camera_to_world'])
        self.positions.append(m[:3, 3].tolist())
        q_xyzw = Rotation.from_matrix(m[:3, :3]).as_quat().tolist()
        self.orientations.append(q_xyzw)
        if self.model_frame is None:
            self.model_frame = frame

    def finish(self):
        import splines
        import splines.quaternion
        from scipy.spatial.transform import Rotation

        # as in Nerfstudio
        end_cond = "closed" if self.loop else "natural"

        orientation_spline = splines.quaternion.KochanekBartels(
            [
                splines.quaternion.UnitQuaternion.from_unit_xyzw(q)
                for q in self.orientations
            ],
            tcb=(self.tension, 0.0, 0.0),
            endconditions=end_cond,
        )

        position_spline = splines.KochanekBartels(
            self.positions,
            tcb=(self.tension, 0.0, 0.0),
            endconditions=end_cond,
        )

        n = len(self.positions)
        for t in np.linspace(0, n-1, num=(n-1)*self.frames_per_transition, endpoint=True):
            f = { k: v for k, v in self.model_frame.items() }

            q = orientation_spline.evaluate(t)
            p = position_spline.evaluate(t)
            m = np.eye(4)
            m[:3, 3] = p
            m[:3, :3] = Rotation.from_quat([*q.vector, q.scalar]).as_matrix()

            f['camera_to_world'] = m.tolist()
            self.target.append(f)

def look_at(cam_pos, cam_target, up_dir=np.array([0, 0, 1])):
    z = cam_target - cam_pos
    z = z / np.linalg.norm(z)
    x = np.cross(z, up_dir)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    y = y / np.linalg.norm(y)
    m = np.eye(4)
    m[:3, 3] = cam_pos
    m[:3, :3] = np.column_stack((x, -y, -z))
    return m

def get_original_length_seconds(raw_input_data_jsonl):
    with open(raw_input_data_jsonl, 'rt') as f:
        first_ts = None
        for line in f:
            d = json.loads(line)
            if 'time' in d:
                last_ts = d['time']
                if first_ts is None:
                    first_ts = last_ts
    return last_ts - first_ts

def add_velocities(camera_path, loop=False):
    from scipy.spatial.transform import Rotation

    path = camera_path['camera_path']
    for i in range(len(path)):
        if loop:
            i_prev = (i - 1) % len(path)
            i_next = (i + 1) % len(path)
        else:
            i_prev = max(0, i - 1)
            i_next = min(len(path) - 1, i + 1)
        
        delta_t = i_next - i_prev

        prev_pose = np.array(path[i_prev]['camera_to_world'])
        next_pose = np.array(path[i_next]['camera_to_world'])

        velocity_w = (next_pose[:3, 3] - prev_pose[:3, 3]) / delta_t

        cur_pose = np.array(path[i]['camera_to_world'])

        rot = next_pose[:3, :3] @ prev_pose[:3, :3].transpose()
        rot_vec = Rotation.from_matrix(rot).as_rotvec()
        ang_vel_w = rot_vec / delta_t

        R_w2c = cur_pose[:3, :3].transpose()
        velocity_cam = R_w2c @ velocity_w
        ang_vel_cam = R_w2c @ ang_vel_w

        path[i]['camera_linear_velocity'] = velocity_cam.tolist()
        path[i]['camera_angular_velocity'] = ang_vel_cam.tolist()

def process(out_folder, args):
    import numpy as np

    path = os.path.normpath(out_folder)
    name = os.path.basename(path)
    variant_folder = os.path.split(path)[0]
    # variant = os.path.basename(variant_folder)
    dataset_folder = os.path.split(variant_folder)[0]
    dataset = os.path.basename(dataset_folder)
    result_folder = os.path.join(out_folder, 'splatfacto', os.listdir(os.path.join(out_folder, 'splatfacto'))[0])
    config_file = os.path.join(result_folder, 'config.yml')

    input_folder = os.path.join('data/inputs-processed', dataset, name)

    with open(os.path.join(input_folder, 'transforms.json'), 'rt') as f:
        transforms = json.load(f)

    with open(os.path.join(result_folder, 'dataparser_transforms.json'), 'rt') as f:
        parser_transforms = json.load(f)

    def transform_func(m):
        if 'applied_transform' in transforms:
            M1 = np.array(transforms['applied_transform'] + [[0,0,0,1]])
        else:
            M1 = np.eye(4)
        M = np.array(parser_transforms['transform'] + [[0,0,0,1]])

        m = np.array(m)
        M = M @ np.linalg.inv(M1)
        m = M @ m
        m[:3, 3] *= parser_transforms['scale']
        return m

    if args.original_trajectory:        
        raw_input_data_jsonl = os.path.join('data', 'inputs-raw', 'spectacular-rec', name, 'data.jsonl')
        
        if os.path.exists(raw_input_data_jsonl):
            length_seconds = get_original_length_seconds(raw_input_data_jsonl)
            print('original length %g' % length_seconds)
        else:
            length_seconds = len(transforms['frames']) * 0.3
            print('approx. length %g' % length_seconds)

        length_seconds /= args.playback_speed
        
        def get_frame_number(frame):
            return int(frame['file_path'].rpartition('_')[-1].split('.')[0])
        
        frames = sorted(transforms['frames'], key=get_frame_number)
        frames = frames[::args.key_frame_stride]

        if args.max_duration is not None:
            max_frames = round(args.max_duration / length_seconds * len(frames))
            if max_frames < len(frames):
                length_seconds = length_seconds * max_frames / len(frames)
                print('keeping %d/%d key frames to cut duration to %g' % (max_frames, len(frames), length_seconds))
                frames = frames[:max_frames]

        frame_poses = [transform_func(frame['transform_matrix']) for frame in frames]
        loop = False
    else:
        length_seconds = args.artificial_length_seconds
        loop = True

        rough_up_dir = np.array([0, 0, 1])

        frame_poses_np = [transform_func(frame['transform_matrix']) for frame in transforms['frames']]
        scene_cam_center = np.mean([m[:3, 3] for m in frame_poses_np], axis=0)
        scene_cam_mean_dir = np.mean([-m[:3, 2] for m in frame_poses_np], axis=0)
        scene_cam_mean_dir = scene_cam_mean_dir / np.linalg.norm(scene_cam_mean_dir)

        scene_scale = np.max([np.linalg.norm(m[:3, 3] - scene_cam_center) for m in frame_poses_np])
        cam_target = scene_cam_center + scene_cam_mean_dir * scene_scale * args.artificial_relative_look_at_distance
        left = np.cross(rough_up_dir, scene_cam_mean_dir)
        left = left / np.linalg.norm(left)
        up = np.cross(scene_cam_mean_dir, left)

        up_dim = np.max(np.abs(np.dot([m[:3, 3] - scene_cam_center for m in frame_poses_np], up)))
        left_dim = np.max(np.abs(np.dot([m[:3, 3] - scene_cam_center for m in frame_poses_np], left)))
        
        frame_poses = []
        for t in np.linspace(0, 2*np.pi, endpoint=False, num=100):
            frame_poses.append(look_at(
                scene_cam_center + args.artificial_relative_motion_scale * (
                    up_dim * up * np.sin(t * args.artificial_y_rounds) +
                    left_dim * left * np.cos(t)
                ),
                cam_target,
                rough_up_dir
            ))

        center_cam_to_world = look_at(scene_cam_center, cam_target, rough_up_dir)

    fov = 2.0 * np.arctan(0.5 * transforms['h'] / transforms['fl_y']) / np.pi * 180.0 / args.zoom
    frames_per_transition = round((length_seconds *  args.fps) / (len(frame_poses) - 1))

    width = transforms['w']
    height = transforms['h']
    if args.resolution is not None:
        width, height = [int(x) for x in args.resolution.split('x')]

    aspect = width / float(height)

    cam_path = {
        'render_width': width,
        'render_height': height,
        'fps': args.fps,
        'seconds': length_seconds,
        'camera_path': []
    }
                
    interpolator = SplineInterpolator(cam_path['camera_path'], frames_per_transition=frames_per_transition)
    interpolator.loop = loop

    for pose in frame_poses:
        # print(frame['file_path'])
        interpolator.push({
            'aspect': aspect,
            'fov': fov,
            'camera_to_world': pose
        })

    interpolator.finish()

    add_velocities(cam_path)
    cam_path['rolling_shutter_time'] = args.rolling_shutter_time
    cam_path['exposure_time'] = args.exposure_time

    if args.artificial_keep_center_pose:
        for c in cam_path['camera_path']: c['camera_to_world'] = center_cam_to_world.tolist()

    trajectory_file = os.path.join(result_folder, 'demo_video_camera_path.json')

    if args.output_video_file is None:
        video_fn = ['demo_video']
        if args.rolling_shutter_time > 0:
            video_fn.append('rs')
        if args.exposure_time > 0:
            video_fn.append('mb')
        
        video_file = os.path.join(result_folder, '-'.join(video_fn) + '.mp4')
    else:
        video_file = args.output_video_file

    render_cmd = [
        'ns-render',
        'camera-path',
        '--load-config', config_file,
        '--camera-path-filename', trajectory_file,
        '--output-path', video_file
    ]

    if args.video_crf is not None:
        render_cmd.extend(['--crf', str(args.video_crf)])

    if not args.dry_run:
        with open(trajectory_file, 'wt') as f:
            json.dump(cam_path, f, indent=4)

        subprocess.check_call(render_cmd)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("input_folder", type=str, default=None, nargs='?')
    parser.add_argument('--output_variant_folder', default='data/outputs/colmap-sai-cli-imgs/baseline', type=str)
    parser.add_argument('-o', '--output_video_file', default=None, type=str)
    parser.add_argument('--key_frame_stride', default=3, type=int)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--original_trajectory', action='store_true')
    parser.add_argument('--fps', default=30, type=int)
    parser.add_argument('--playback_speed', default=0.5, type=float)
    parser.add_argument('--artificial_relative_motion_scale', default=0.6, type=float)
    parser.add_argument('--artificial_relative_look_at_distance', default=3, type=float)
    parser.add_argument('--artificial_y_rounds', default=1, type=int)
    parser.add_argument('--artificial_length_seconds', default=8, type=float)
    parser.add_argument('--artificial_keep_center_pose', action='store_true')
    parser.add_argument('--rolling_shutter_time', default=0.0, type=float)
    parser.add_argument('--max_duration', default=None, type=float)
    parser.add_argument('--resolution', type=str, default=None)
    parser.add_argument('--exposure_time', default=0.0, type=float)
    parser.add_argument('--zoom', default=1.0, type=float)
    parser.add_argument('--video_crf', default=None, type=int)
    parser.add_argument('--case_number', type=int, default=-1)
    args = parser.parse_args()

    if args.input_folder in ['all']:
        args.case_number = 0
        args.input_folder = None

    selected_cases = []

    if args.input_folder is None:
        src_folder = args.output_variant_folder
        cases = [os.path.join(src_folder, f) for f in sorted(os.listdir(src_folder))]

        if args.case_number == -1:
            print('valid cases')
            for i, c in enumerate(cases): print(str(i+1) + ':\t' + c)
        elif args.case_number == 0:
            selected_cases = cases
        else:
            selected_cases = [cases[args.case_number - 1]]
    else:
        selected_cases = [args.input_folder]

    for case in selected_cases:
        print('Processing ' + case)
        process(case, args)


```

# prompts/papers/3dgs-deblur/run_colmap.py

``` py
"""Run COLMAP on a single sequence through Nerfstudio scripts"""
import os
import subprocess
import shutil
import sys
import tempfile

def process(input_folder, args):
    name = os.path.basename(os.path.normpath(input_folder))
    postf = 'colmap-' + args.dataset + '-imgs'
    if args.output_folder is None:
        output_folder = os.path.join('data/inputs-processed/' + postf, name)
    else:
        output_folder = args.output_folder

    input_image_folder = os.path.join(input_folder, 'images')

    temp_dir = tempfile.TemporaryDirectory()
    n = 0
    for f in os.listdir(input_image_folder):
        if 'depth' in f: continue
        if not args.dry_run:
            shutil.copyfile(os.path.join(input_image_folder, f), os.path.join(temp_dir.name, f))
        n += 1
    print('%d images (would be) copied in a temporary directory' % n)

    # Print the path to the temporary directory
    cmd = [
        'ns-process-data',
        'images',
        '--data', temp_dir.name,
        '--output-dir', output_folder
    ]

    print(cmd)
    success = False
    if not args.dry_run:
        for itr in range(args.max_retries):
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)

            ret = subprocess.run(cmd, check=True, capture_output=True)
            success = any([b'CONGRATS' in s for s in [ret.stdout, ret.stderr]]) # hacky
            if success:
                break
            else:
                print('COLMAP failed')
                print('--- stdout ---')
                sys.stdout.buffer.write(ret.stdout)
                print('--- stderr ---')
                sys.stderr.buffer.write(ret.stderr)
                if itr != args.max_retries - 1:
                    print('Retrying...')

        if not success:
            raise RuntimeError('Could not get COLMAP to succeed')
    
    temp_dir.cleanup()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("input_folder", type=str, default=None, nargs='?')
    parser.add_argument("output_folder", type=str, default=None, nargs='?')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--dataset', default='sai-cli')
    parser.add_argument('--case_number', type=int, default=-1)
    parser.add_argument('--max_retries', type=int, default=1)
    
    args = parser.parse_args()

    if args.input_folder in ['all']:
        args.case_number = 0
        args.input_folder = None
        
    selected_cases = []
    misc = False

    PROCESSED_PREFIX = 'data/inputs-processed/'
    if args.dataset.startswith(PROCESSED_PREFIX):
        args.dataset = args.dataset[len(PROCESSED_PREFIX):]
    if args.dataset.endswith('/'): args.dataset = args.dataset[:-1]

    if args.input_folder is None:
        input_root = os.path.join(PROCESSED_PREFIX, args.dataset)
        cases = [os.path.join(input_root, f) for f in sorted(os.listdir(input_root))]

        if args.case_number == -1:
            print('valid cases')
            for i, c in enumerate(cases): print(str(i+1) + ':\t' + c)
        elif args.case_number == 0:
            selected_cases = cases
        else:
            selected_cases = [cases[args.case_number - 1]]
    else:
        selected_cases = [args.input_folder]

    for case in selected_cases:
        print('Processing ' + case)
        process(case, args)


```

# prompts/papers/3dgs-deblur/train.py

``` py
"""Train a single instance"""
import os
import subprocess
import shutil
import sys
import time
import datetime
import json
import re

DATASET_SPECIFIC_PARAMETERS = {
    r".*synthetic.*": [
        # '--max-num-iterations', '20000', # this would be enough, usually
        '--pipeline.model.num-downscales', '0', # low resolution -> no downscaling
        # These help reconstructing large areas with very smooth color,
        # i.e., the synthetic sky. With defaults, large holes can easily appear
        '--pipeline.model.background-color', 'auto',
        '--pipeline.model.cull-scale-thresh', '2.0',
        # Evaluation data is known to be static. Don't try to optimize camera velocities
        '--pipeline.model.optimize-eval-velocities=False',
        # Hight motion blur, needs more samples
        '--pipeline.model.blur-samples=10',
    ]
}

def print_cmd(cmd):
    print('RUNNING COMMAND: ' + ' '.join(cmd))

def flags_to_variant_name_and_cmd(args):    
    cmd = []
    variant = []

    use_gamma_correction = False
    optimize_eval_cameras = False

    if not args.get('no_pose_opt', False):
        optimize_eval_cameras = True
        variant.append('pose_opt')
        cmd.extend([
            '--pipeline.model.camera-optimizer.mode=SO3xR3',
            ## '--pipeline.model.sh-degree=0'
        ])

    if not args.get('no_motion_blur', False):
        variant.append('motion_blur')
        # default blur samples: 5
        use_gamma_correction = not args.get('no_gamma', False)
        if not use_gamma_correction:
            variant.append('no_gamma')
    else:
        cmd.append('--pipeline.model.blur-samples=0')

    if not args.get('no_rolling_shutter', False):
        variant.append('rolling_shutter')
    else:
        cmd.append('--pipeline.model.rolling-shutter-compensation=False')

    if use_gamma_correction:
        # min RGB level only seems necessary with gamma correction
        cmd.append('--pipeline.model.min-rgb-level=10')
    else:
        cmd.append('--pipeline.model.gamma=1')

    if not args.get('no_velocity_opt', False):
        optimize_eval_cameras = True
        cmd.append('--pipeline.model.camera-velocity-optimizer.enabled=True')
        variant.append('velocity_opt')

    if args.get('velocity_opt_zero_init', False):
        cmd.append('--pipeline.model.camera-velocity-optimizer.zero-initial-velocities=True')
        variant.append('zero_init')

    if len(variant) == 0:
        variant.append('baseline')

    return '-'.join(variant), cmd, optimize_eval_cameras

def evaluate(output_folder, elapsed_time, dry_run=False, render_images=True):
    result_paths = find_config_path(output_folder)
    if result_paths is None:
        if dry_run: return
        assert(False)

    out_path, config_path = result_paths
    metrics_path = os.path.join(out_path, 'metrics.json')
    elapsed_time
    eval_cmd = [
        'ns-eval',
         '--load-config', config_path,
         '--output-path', metrics_path
    ]

    print_cmd(eval_cmd)
    if not dry_run:
        subprocess.check_call(eval_cmd)
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics['wall_clock_time_seconds'] = elapsed_time
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
    
    if render_images:
        render_cmd = [
            'python', 'render_model.py',
            '--load-config', config_path
        ]
        print_cmd(render_cmd)
        if not dry_run:
            subprocess.check_call(render_cmd)

def process(input_folder, args):
    name = os.path.split(input_folder)[-1]

    cmd = [
        'ns-train',
        'splatfacto',
        '--data',  input_folder,
        '--viewer.quit-on-train-completion', 'True',
        '--pipeline.model.rasterize-mode', 'antialiased',
        '--pipeline.model.use-scale-regularization', 'True',
        # '--logging.local-writer.max-log-size=0'
    ]

    for pattern, values in DATASET_SPECIFIC_PARAMETERS.items():
        if re.match(pattern, args.dataset):
            cmd.extend(values)

    if '--max-num-iterations' not in cmd:
        if args.draft:
            cmd.extend(['--max-num-iterations', '3000'])
        else:
            cmd.extend(['--max-num-iterations', '20000'])

    if args.preview:
        cmd.extend([
            '--vis=viewer+tensorboard',
            '--viewer.websocket-host=127.0.0.1'
        ])
    else:
        cmd.append('--vis=tensorboard')

    variant, variant_cmd, optimize_eval_cameras = flags_to_variant_name_and_cmd(vars(args))
    cmd.extend(variant_cmd)

    if args.case_number is None:
        dataset_folder = 'custom'
    else:
        dataset_folder = args.dataset
        
    variant_folder = os.path.join(dataset_folder, variant)

    output_prefix = 'data/outputs'

    # note: 'name' is automatically added by Nerfstudio
    output_root = os.path.join(output_prefix, variant_folder)

    cmd.extend(['--output-dir', output_root])

    cmd.extend([
        'nerfstudio-data',
        '--orientation-method', 'none',
    ])

    if args.train_all:
        cmd.extend([
            '--eval-mode', 'all'
        ])
        optimize_eval_cameras = False
    elif '-scored' in args.input_folder or args.dataset == 'colmap-bad-nerf-synthetic-deblurring':
        cmd.extend([
            '--eval-mode', 'filename'
        ])
    else:
        cmd.extend([
            '--eval-mode', 'interval',
            '--eval-interval', '8'
        ])
        #cmd.extend(['--eval-mode', 'all'])

    if optimize_eval_cameras:
        cmd.extend([
            '--optimize-eval-cameras', 'True',
        ])

    print_cmd(cmd)
    output_folder = os.path.join(output_root, name)
    elapsed_time = 0
    if not args.dry_run and not args.eval_only:
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        start_time = time.time()
        subprocess.check_call(cmd)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Training time: %s' % str(datetime.timedelta(seconds=elapsed_time)))
    
    if not args.no_eval:
        evaluate(output_folder, elapsed_time,
            dry_run=args.dry_run,
            render_images=args.render_images)

def find_config_path(output_folder):
    model_folder = os.path.join(output_folder, 'splatfacto')
    paths = []
    if os.path.exists(model_folder):
        for subdir in os.listdir(model_folder):
            out_path = os.path.join(model_folder, subdir)
            config_path = os.path.join(out_path, 'config.yml')
            if os.path.exists(config_path):
                paths.append((out_path, config_path))
    if len(paths) == 0: return None
    assert(len(paths) == 1)
    return paths[0]

def add_velocity_opt_variants(variants, dataset):
    has_velocity_info = ('sai-' in dataset
        or 'spectacular-rec' in dataset
        or ('synthetic-' in dataset and 'colmap' not in dataset and 'hloc' not in dataset)
    )

    new_variants = []
    for v in variants:
        v1 = v.copy()
        no_velocity_to_optimize = 'no_rolling_shutter' in v and 'no_motion_blur' in v
        if has_velocity_info or no_velocity_to_optimize:
            v1.add('no_velocity_opt')
            new_variants.append(v1)

        if no_velocity_to_optimize: continue

        if has_velocity_info:
            new_variants.append(v)

        v2 = v.copy()
        v2.add('velocity_opt_zero_init')
        new_variants.append(v2)

    return new_variants

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    # note: velocity optimization arguments are auto-added to all of these
    baseline = {
        'no_pose_opt',
        'no_motion_blur',
        'no_rolling_shutter'
    }

    no_rolling_shutter_variants = [
        baseline,
        { 'no_rolling_shutter', 'no_pose_opt' },
        { 'no_rolling_shutter', 'no_motion_blur' },
        { 'no_rolling_shutter' }
    ]
    
    full_variants = no_rolling_shutter_variants + [
        { 'no_pose_opt', 'no_motion_blur' },
        { 'no_pose_opt' },
        { 'no_motion_blur' },
        set([])
    ]

    default_variants = full_variants
    bad_nerf_variants = [
        baseline,
        { 'no_rolling_shutter', 'no_pose_opt' },
        { 'no_rolling_shutter' }
    ]

    add_popt = lambda a: a + [o - {'no_pose_opt'} for o in a if 'no_pose_opt' in o]

    variants_by_dataset = {
        'synthetic-clear': [
            baseline
        ],
        'synthetic-mb': add_popt([
            baseline,
            { 'no_pose_opt', 'no_rolling_shutter' }
        ]),
        'synthetic-rs': add_popt([
            baseline,
            { 'no_pose_opt', 'no_motion_blur' }
        ]),
        'synthetic-posenoise': add_popt([
            baseline,
            { 'no_rolling_shutter', 'no_motion_blur' }
        ]),
        'synthetic-mbrs': add_popt([
            baseline,
            { 'no_pose_opt' },
            { 'no_pose_opt', 'no_motion_blur' },
            { 'no_pose_opt', 'no_rolling_shutter' }
        ]),
        'synthetic-posenoise-2nd-pass': [
            baseline
        ],
        'colmap-bad-nerf-synthetic-deblurring': bad_nerf_variants,
        'colmap-bad-nerf-synthetic-novel-view': bad_nerf_variants,
        'colmap-bad-nerf-synthetic-novel-view-manual-pc': add_popt(bad_nerf_variants),
        'colmap-exblurf-synthetic-novel-view-manual-pc': bad_nerf_variants,
        'hloc-exblurf-synthetic-novel-view-manual-pc': bad_nerf_variants,
        'hloc-bad-nerf-synthetic-novel-view-manual-pc': bad_nerf_variants,
        'hloc-bad-nerf-synthetic-novel-view-exact-intrinsics-manual-pc': bad_nerf_variants,
        'hloc-bad-gaussians-synthetic-novel-view-manual-pc': bad_nerf_variants,
        'colmap-bad-gaussians-synthetic-novel-view-manual-pc': bad_nerf_variants,
        'colmap-mpr-deblurred-synthetic-all-manual-pc': bad_nerf_variants,
        'colmap-mpr-deblurred-synthetic-novel-view-manual-pc': bad_nerf_variants + [{ 'no_rolling_shutter', 'no_motion_blur' }],
    }

    parser.add_argument("input_folder", type=str, default=None, nargs='?')
    parser.add_argument("--preview", action='store_true', help='show Viser preview')
    parser.add_argument("--no_pose_opt", action='store_true')
    parser.add_argument("--no_motion_blur", action='store_true')
    parser.add_argument('--no_rolling_shutter', action='store_true')
    parser.add_argument('--no_velocity_opt', action='store_true')
    parser.add_argument('--velocity_opt_zero_init', action='store_true')
    parser.add_argument('--dataset', type=str, default='colmap-sai-cli-vels-blur-scored')
    parser.add_argument('--draft', action='store_true')
    parser.add_argument('--no_gamma', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--render_images', action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--no_eval', action='store_true')
    parser.add_argument('--train_all', action='store_true')

    parser.add_argument('--case_number', type=int, default=None)
    args = parser.parse_args()

    if args.input_folder is None and args.case_number is None:
        args.case_number = -1

    if args.case_number is not None:
        INPUT_ROOT = 'data/inputs-processed/' + args.dataset
        sessions = [os.path.join(INPUT_ROOT, f) for f in sorted(os.listdir(INPUT_ROOT))]
        variants = add_velocity_opt_variants(variants_by_dataset.get(args.dataset, default_variants), args.dataset)
        cases = [(s, v) for v in variants for s in sessions]

        if args.case_number <= 0:
            print('valid cases')
            for i, (c, v) in enumerate(cases):
                variant = flags_to_variant_name_and_cmd({k: True for k in v})[0]
                print(str(i+1) + ':\t' + variant + '\t' + c)
            sys.exit(0)
        else:
            args.input_folder, variant = cases[args.case_number - 1]
            for p in variant: setattr(args, p, True)
            print('Running %s %s' % (args.input_folder, str(variant)))

    process(args.input_folder, args)


```

# prompts/papers/3dgs-deblur/train_eval_split_by_blur_score.py

``` py
"""Combine COLMAP poses with sai-cli velocities"""
import os
import json
import shutil

def process(input_folder, output_prefix, args):
    name = os.path.basename(os.path.normpath(input_folder))
    print('name', name)

    def read_json(folder):
        with open(os.path.join(folder, 'transforms.json')) as f:
            return json.load(f)

    print(input_folder)
    output_folder = os.path.join(output_prefix, name)

    input_image_folder = os.path.join(input_folder, 'images')
    output_image_folder = os.path.join(output_folder, 'images')
    
    poses = read_json(input_folder)
    poses['frames'].sort(key=lambda x: x['file_path'])

    if not args.dry_run:
        if os.path.exists(output_folder): shutil.rmtree(output_folder)
        os.makedirs(output_image_folder)
        
    ival_start = 0
    while ival_start < len(poses['frames']):
        ival_end = ival_start + args.interval
        least_blur = sorted(poses['frames'][ival_start:ival_end], key=lambda x: x['motion_blur_score'])[0]['file_path']

        for frame in poses['frames'][ival_start:ival_end]:
            id = frame['file_path']
            if id == least_blur:
                new_name = f'eval_' + os.path.basename(id)
            else:
                new_name = f'train_' + os.path.basename(id)

            old_file_name = os.path.join(input_image_folder, os.path.basename(id))
            new_file_name = os.path.join(output_image_folder, new_name)

            frame['file_path'] = os.path.join('images', new_name)
            print("%s -> %s (%g)" % (old_file_name, new_file_name, frame['motion_blur_score']))
            if not args.dry_run:
                shutil.copyfile(old_file_name, new_file_name)

        ival_start = ival_end

    # colmap_folder = os.path.join(args.input_folder, 'colmap')
    ply_pc = os.path.join(input_folder, 'sparse_pc.ply')

    print('Output folder: ' + output_folder)
    if not args.dry_run:
        # shutil.copytree(colmap_folder, os.path.join(output_folder, 'colmap'))
        shutil.copyfile(ply_pc, os.path.join(output_folder, 'sparse_pc.ply'))
        with open (os.path.join(output_folder, 'transforms.json'), 'w') as f:
            json.dump(poses, f, indent=4)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('dataset')
    parser.add_argument("input_folder", type=str, default=None, nargs='?')
    parser.add_argument('--interval', type=int, default=8)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--case_number', type=int, default=-1)
    args = parser.parse_args()

    if args.input_folder in ['all']:
        args.case_number = 0
        args.input_folder = None

    selected_cases = []

    PROCESSED_PREFIX = 'data/inputs-processed/'
    if args.dataset.startswith(PROCESSED_PREFIX):
        args.dataset = args.dataset[len(PROCESSED_PREFIX):]

    out_folder = os.path.join(PROCESSED_PREFIX, args.dataset + '-blur-scored')

    if args.input_folder is None:
        processed_prefix = os.path.join(PROCESSED_PREFIX, args.dataset)
        cases = [os.path.join(processed_prefix, f) for f in sorted(os.listdir(processed_prefix))]

        if args.case_number == -1:
            print('valid cases')
            for i, c in enumerate(cases): print(str(i+1) + ':\t' + c)
        elif args.case_number == 0:
            selected_cases = cases
        else:
            selected_cases = [cases[args.case_number - 1]]
    else:
        selected_cases = [args.input_folder]

    for case in selected_cases:
        print('Processing ' + case)
        process(case, out_folder, args)


```

