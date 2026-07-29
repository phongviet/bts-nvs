"""Analysis 04 / Experiment X3: depth-guided image-based rendering (DIBR) pilot.

The geometry analysis (01) showed every scene is a dense-interpolation regime:
the median test camera sits 0.2-0.5 inter-frame spacings from a train camera,
with train frames on both sides. So instead of asking a lossy 3DGS model to
re-synthesize pixels, warp the REAL pixels of the 1-3 nearest train photos into
the test pose using the 3DGS model only for geometry (depth) + occlusion fill.
This attacks LPIPS (0.4 of Score) directly: warped real texture keeps the
high-frequency detail that splat renders blur away.

Pipeline per test view:
  1. render rgb_T + depth_T at the test pose from the trained splatfacto ckpt;
  2. unproject every test pixel to a 3D point (pinhole CSV intrinsics);
  3. project into the K nearest train cameras (SIMPLE_RADIAL: distorted coords
     to sample the RAW train JPG; undistorted pinhole coords to sample that
     neighbor's rendered depth for the occlusion z-test);
  4. blend valid samples (weights 1/distance, feathered); occluded/out-of-view
     pixels fall back to the 3DGS render.

Validation mode (--mode traincheck) warps neighbors into a held-out TRAIN view
and scores the central 60% crop (corner distortion < 1 px there) against the
real photo — proving depth/convention/warp correctness on GT we possess.

Run:
  conda run -n airace python Analysis/04_x3_dibr_pilot.py --scene hcm0034 --mode traincheck
  conda run -n airace python Analysis/04_x3_dibr_pilot.py --scene hcm0034 --mode test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "phase1" / "public_set"
PRIVATE = REPO / "data" / "raw" / "phase1" / "private_set1"
ROUND2 = REPO / "data" / "raw" / "VAI_NVS_DATA_ROUND2"


def scene_raw(scene: str) -> Path:
    """Phase-1 public scenes live under public_set, private under private_set1;
    round-2 scenes (the only ones still graded) under VAI_NVS_DATA_ROUND2."""
    for root in (RAW, PRIVATE, ROUND2):
        if (root / scene).exists():
            return root
    raise SystemExit(f"scene {scene!r} not found under {RAW}, {PRIVATE} or {ROUND2}")


def is_round2(scene: str) -> bool:
    return (ROUND2 / scene).exists()
OUT = Path(__file__).resolve().parent / "X3_dibr"
sys.path.insert(0, str(REPO))

from src.render import colmap_pose_to_c2w, apply_dataparser_transform, load_test_poses  # noqa: E402


# --- vendored COLMAP binary readers -------------------------------------------
# Standard COLMAP cameras.bin/images.bin parsers (identical output shape to
# nerfstudio.data.utils.colmap_parsing_utils' read_*_binary, minus the nerfstudio
# dependency). This lets the COLMAP-native refiner path (SSS render + RaDe-GS
# depth, no splatfacto) run in a plain torch env without nerfstudio installed.
# The splatfacto path still imports nerfstudio lazily via eval_setup.
import struct as _struct  # noqa: E402
from collections import namedtuple as _nt  # noqa: E402

_CamRec = _nt("Camera", ["id", "model", "width", "height", "params"])
_ImgRec = _nt("Image", ["id", "qvec", "tvec", "camera_id", "name"])
# model_id -> (name, num_params); only the models our scenes use are needed.
_CAMERA_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4),
                  2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5), 4: ("OPENCV", 8)}


def read_cameras_binary(path):
    cams = {}
    with open(path, "rb") as f:
        (n,) = _struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cam_id, model_id, w, h = _struct.unpack("<iiQQ", f.read(24))
            name, npar = _CAMERA_MODELS[model_id]
            params = np.array(_struct.unpack("<" + "d" * npar, f.read(8 * npar)))
            cams[cam_id] = _CamRec(cam_id, name, w, h, params)
    return cams


def read_images_binary(path):
    imgs = {}
    with open(path, "rb") as f:
        (n,) = _struct.unpack("<Q", f.read(8))
        for _ in range(n):
            props = _struct.unpack("<idddddddi", f.read(64))
            img_id = props[0]
            qvec = np.array(props[1:5]); tvec = np.array(props[5:8])
            cam_id = props[8]
            name = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c
            (npts,) = _struct.unpack("<Q", f.read(8))
            f.read(24 * npts)  # skip the 2D points (x, y, point3D_id)
            imgs[img_id] = _ImgRec(img_id, qvec, tvec, cam_id, name.decode())
    return imgs

CONFIGS = {
    "hcm0034": REPO / "runs/phase1/exp004_backend_ablation/antialiased/train_staging_dense/splatfacto/2026-07-05_153154/config.yml",
    "hcm0031": REPO / "runs/phase1/exp004_backend_ablation/hcm0031_antialiased",
    "HCM0181": REPO / "runs/phase1/exp004_backend_ablation/HCM0181_antialiased",
    "HCM0193": REPO / "runs/phase1/exp004_backend_ablation/HCM0193_antialiased",
    "HCM0204": REPO / "runs/phase1/exp004_hcm0204_fill/HCM0204",
    # --- 8 private scenes (exp005 fleet; Kaggle-trained, fix_paths rewrites output_dir) ---
    "HCM0249": REPO / "runs/phase1/exp005_antialiased_dense/HCM0249",
    "HCM0254": REPO / "runs/phase1/exp005_antialiased_dense/HCM0254",
    "HCM0276": REPO / "runs/phase1/exp005_antialiased_dense/HCM0276",
    "HCM1439": REPO / "runs/phase1/exp005_antialiased_dense/HCM1439",
    "HNI0366": REPO / "runs/phase1/exp005_antialiased_dense/HNI0366",
    "HNI0437": REPO / "runs/phase1/exp005_antialiased_dense/HNI0437",
    "HNI0131": REPO / "runs/phase1/exp005_antialiased_dense/HNI0131",  # k=-0.115: DIBR out_k needs expanded canvas (see 08)
    "HNI0265": REPO / "runs/phase1/exp005_antialiased_dense/HNI0265",  # k=-0.115: DIBR out_k needs expanded canvas (see 08)
}


def find_config(scene: str) -> Path:
    # Round-2 backbones are not enumerated in CONFIGS: they are produced by
    # scripts/phase_run.sh into a predictable runs/round2/phase_locked/<scene>/
    # tree, so discover them instead of hand-maintaining 7 more entries (the 5
    # drone scenes land there as their fleet finishes).
    p = CONFIGS.get(scene)
    if p is None:
        p = REPO / "runs" / "round2" / "phase_locked" / scene
        if not p.exists():
            raise SystemExit(
                f"no backbone for {scene!r}: not in CONFIGS and {p} does not exist")
    if p.name == "config.yml":
        return p
    hits = sorted(p.glob("**/config.yml"))
    assert hits, f"no config.yml under {p}"
    return hits[-1]


_FLOW_MOD = None


def _flow_align_mod():
    """Lazy-load Analysis/17_flow_align.py (leading digit -> import by path)."""
    global _FLOW_MOD
    if _FLOW_MOD is None:
        import importlib.util
        p = Path(__file__).resolve().parent / "17_flow_align.py"
        spec = importlib.util.spec_from_file_location("flow_align17", p)
        _FLOW_MOD = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_FLOW_MOD)
    return _FLOW_MOD


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


class Warper:
    def __init__(self, scene: str, config_path=None, ss: int = 1, sample: str = "bilinear",
                 depth_source=None, holdout_names=None, render_override=None,
                 target_depth_source=None, train_dir=None):
        """ss: supersample factor for the target 3DGS render (2 = render the
        fallback/depth canvas at 2x and sample it on the finer grid — sharper
        fallback + more accurate depth edges; ss=1 is the original behavior).
        sample: 'bilinear' (original) or 'cubic' — interpolation used to gather
        real train-image pixels (and the ss fallback rgb). config_path: override
        the CONFIGS backbone checkpoint (e.g. exp006 splatfacto-big).
        depth_source: exp041 (Wave 2) — a directory of externally-rendered
        per-train-view depth maps ({image_name}.npy, metres, same H/W and camera
        convention as our pinhole render) to use for the occlusion z-test
        INSTEAD of the 3DGS expected depth. Populate it with
        Analysis/18_import_depth.py (RaDe-GS / PGSR). Only the OCCLUSION TEST
        changes; the fallback RGB still comes from our backbone, so this is a
        clean depth-only A/B."""
        self.scene = scene
        self.ss = int(ss)
        self.sample = sample
        self.scene_dir = scene_raw(scene) / scene
        # train_dir: directory holding {sparse/0/{cameras,images}.bin, images/}.
        # Default = the raw scene's train/. COLMAP-native points this at the
        # UNDISTORTED (pinhole) scene SSS/RaDe-GS actually trained on, so the
        # camera model is SIMPLE_PINHOLE (k=0, distortion is a no-op), the poses
        # match the SSS renders, and train_img() serves the pinhole GT.
        self.train_dir = Path(train_dir) if train_dir else (self.scene_dir / "train")
        # target_depth_source (COLMAP-native mode): a directory of externally
        # rendered per-TEST-pose depth maps ({image_name}.npy, raw COLMAP units,
        # the same z-forward OpenCV convention as depth_source). When set, the
        # Warper drops nerfstudio/splatfacto ENTIRELY: depth_T comes from this
        # dir, rgb_T from render_override (SSS), and the world transform is the
        # identity at scale 1 (SSS + RaDe-GS both live in raw COLMAP world
        # units, so no dataparser_transform/scale is applied). This is the
        # "SSS render backbone + RaDe-GS all-depth, no splatfacto" refiner path.
        self.target_depth_source = Path(target_depth_source) if target_depth_source else None
        self.colmap_native = self.target_depth_source is not None
        if self.colmap_native:
            if not self.target_depth_source.is_dir():
                raise SystemExit(f"target_depth_source {self.target_depth_source} is not a directory")
            if render_override is None:
                raise SystemExit("colmap_native mode requires --render-override (SSS renders)")
            if depth_source is None:
                raise SystemExit("colmap_native mode requires --depth-source (RaDe-GS train depth)")
            if holdout_names:
                raise SystemExit("colmap_native mode does not support --val-holdout")
            if int(ss) != 1:
                raise SystemExit("colmap_native mode requires ss=1 (depth_T is an "
                                 "externally-rendered map at native test resolution)")
        # neighbor-depth cache must not be shared across backbones (the tag does NOT achieve
        # that on its own -- see the .owner stamp in the depth getter)
        self.cache_tag = "_bb" if config_path is not None else ""
        self._cfg_path = None
        self.backbone_staging = None
        self.depth_source = Path(depth_source) if depth_source else None
        if self.depth_source is not None and not self.depth_source.is_dir():
            raise SystemExit(f"depth_source {self.depth_source} is not a directory")
        # render_override (SSS-backbone gate-2): a dir of externally-rendered
        # per-view RGB ({image_name}[.png/.jpg], same H/W + camera convention as
        # our pinhole render). When set, synthesize(override_name=...) REPLACES
        # the 3DGS rgb_T (the refiner's render channel AND the DIBR fallback/
        # exposure reference) with that image, while depth_T stays from the 3DGS
        # backbone. Lets a non-nerfstudio backbone (SSS) drive the render channel.
        self.render_override_dir = Path(render_override) if render_override else None
        if self.render_override_dir is not None and not self.render_override_dir.is_dir():
            raise SystemExit(f"render_override {self.render_override_dir} is not a directory")
        if self.render_override_dir is not None:
            self.cache_tag += "_ro"  # never share a cache with the 3DGS render channel
        if self.colmap_native:
            # No nerfstudio backbone: raw COLMAP world frame, identity transform.
            # colmap_pose_to_c2w still applies its OpenCV->OpenGL CAMERA flip
            # (which the warp math in synthesize() requires); the identity
            # dataparser transform + scale 1 leave the WORLD in raw COLMAP units,
            # matching the RaDe-GS depths (also raw COLMAP z-forward).
            self.transform = np.eye(4)[:3, :4]
            self.scale = 1.0
            self.pipeline = None
            self.device = "cpu"
        else:
            self._setup_nerfstudio_backbone(config_path)

        cams = read_cameras_binary(self.train_dir / "sparse/0/cameras.bin")
        cam = list(cams.values())[0]
        # Phase-1 (drone) scenes are SIMPLE_RADIAL with a real k; the round-2
        # indoor scenes (bonsai, chair) are SIMPLE_PINHOLE and carry NO
        # distortion term. k=0 is the correct value there, and it makes the
        # whole distortion path a no-op (out_k=0 -> identity remap, no canvas
        # margin), which is exactly the required "bypass remap" behaviour.
        if cam.model == "SIMPLE_RADIAL":
            self.f, self.cx, self.cy, self.k = cam.params
        elif cam.model == "SIMPLE_PINHOLE":
            self.f, self.cx, self.cy = cam.params
            self.k = 0.0
        elif cam.model == "PINHOLE":
            # undistorted scenes are PINHOLE (fx, fy, cx, cy); fx==fy after
            # undistortion, no distortion term -> remap is a no-op.
            fx, fy, self.cx, self.cy = cam.params
            self.f = fx
            self.k = 0.0
        else:
            raise SystemExit(f"{scene}: unsupported camera model {cam.model}")
        self.W_tr, self.H_tr = cam.width, cam.height

        ims = read_images_binary(self.train_dir / "sparse/0/images.bin")
        train_names = {p.name for p in (self.train_dir / "images").iterdir()}
        self.holdout = set(holdout_names or ())
        leaked = self.holdout - train_names
        if leaked:
            raise SystemExit(f"{scene}: holdout names not in train/images: "
                             f"{sorted(leaked)[:5]}...")
        self.train = []  # (name, c2w_ns 4x4, center_ns)
        self.holdout_poses = []  # (name, c2w_ns) for the excluded frames
        for im in ims.values():
            if im.name not in train_names:
                continue
            c2w = self._c2w_ns(list(im.qvec), list(im.tvec))
            if im.name in self.holdout:
                self.holdout_poses.append((im.name, c2w))
                continue
            self.train.append((im.name, c2w, c2w[:3, 3].copy()))
        self.holdout_poses.sort()
        missing = self.holdout - {n for n, _ in self.holdout_poses}
        if missing:
            raise SystemExit(f"{scene}: {len(missing)} holdout frames have no pose "
                             f"in images.bin: {sorted(missing)[:5]}...")
        self.centers = np.stack([c for _, _, c in self.train])
        self._depth_cache: dict[str, np.ndarray] = {}
        self._img_cache: dict[str, np.ndarray] = {}
        self._tgt_depth_cache: dict[str, np.ndarray] = {}

    def _setup_nerfstudio_backbone(self, config_path):
        from nerfstudio.utils.eval_utils import eval_setup
        if config_path is not None:
            config_path = Path(config_path)
            if config_path.name != "config.yml":
                hits = sorted(config_path.glob("**/config.yml"))
                assert hits, f"no config.yml under {config_path}"
                config_path = hits[-1]
        cfg_path = config_path if config_path is not None else find_config(self.scene)
        self._cfg_path = cfg_path

        def fix_paths(config):
            # Some configs (e.g. HCM0204/exp004_hcm0204_fill) carry a Kaggle
            # absolute output_dir + null load_dir, so eval_setup looks for the
            # checkpoint under a nonexistent /kaggle path. Rewrite to the local
            # run dir (config.yml's parent chain) and local processed data.
            local_out = cfg_path.parents[3]  # .../<exp>/<scene>/<variant>
            config.output_dir = local_out
            # The run tree encodes which staging trained it
            # (.../<scene>/<staging>/<method>/<ts>/config.yml), so honour that
            # rather than assuming train_staging_dense: the bonsai val-hold-out
            # backbone trains on train_staging_holdout, and pointing it at the
            # full-train staging would silently re-admit the 25 val frames.
            phase = "round2" if is_round2(self.scene) else "phase1"
            proc = REPO / f"data/processed/{phase}/{self.scene}"
            # FIX (2026-07-26): the run tree is only a HEURISTIC for the staging, and it
            # silently mis-fires. `ns-train --output-dir X --experiment-name <scene>` yields
            # .../<scene>/<method>/<ts>/config.yml, so parents[2] is the SCENE, the lookup
            # misses, and the old code fell through to train_staging_dense -- a DIFFERENT,
            # LARGER pose set (chair 205 vs 180). That changes dataparser_transform/scale, so
            # every warp lands misaligned, and on a val-hold-out backbone it re-admits the 25
            # val frames. It cost the whole 2026-07-26 gate series (0.55 vs 0.67, on ANY
            # backbone). The config records the staging it was TRAINED on -- trust that first.
            dm = config.pipeline.datamanager
            # Which of these three carries the staging depends on how ns-train was invoked:
            # E1's config has an EMPTY dataparser.data and the real path on datamanager.data.
            # An empty PosixPath is truthy and its .name is "", which would resolve to the
            # scene root -- so drop empty names instead of trusting the first attribute.
            stored = [getattr(config, "data", None), getattr(dm, "data", None),
                      getattr(getattr(dm, "dataparser", None), "data", None)]
            cands = [n for n in (Path(str(s)).name for s in stored if s is not None) if n]
            cands += [cfg_path.parents[2].name, "train_staging_dense"]
            local_data = next((proc / c for c in cands if (proc / c).exists()), None)
            if local_data is not None:
                if local_data.name != cands[0]:
                    print(f"WARNING: backbone staging {cands[0]!r} not found locally; using "
                          f"{local_data.name!r} -- verify this is the set that trained it, "
                          f"a mismatch silently corrupts every warp", flush=True)
                config.pipeline.datamanager.data = local_data
                if hasattr(dm, "dataparser") and hasattr(dm.dataparser, "data"):
                    dm.dataparser.data = local_data
                self.backbone_staging = local_data
            return config

        config, self.pipeline, _, _ = eval_setup(cfg_path, update_config_callback=fix_paths)
        dp = self.pipeline.datamanager.train_dataparser_outputs
        # Leave evidence in every log: which pose set defined the world transform. A silent
        # substitution here is invisible in the metrics and ruins them (see fix_paths above).
        st = getattr(self, "backbone_staging", None)
        print(f"BACKBONE staging={st.name if st else '<config default>'} "
              f"poses={len(dp.image_filenames)} scale={float(dp.dataparser_scale):.6f}", flush=True)
        self.transform = dp.dataparser_transform.cpu().numpy()
        self.scale = float(dp.dataparser_scale)
        self.device = self.pipeline.device

    def _c2w_ns(self, qvec, tvec) -> np.ndarray:
        c2w = colmap_pose_to_c2w(qvec, tvec)
        m = apply_dataparser_transform(c2w, self.transform, self.scale)
        out = np.eye(4)
        out[:3, :4] = m
        return out

    def render(self, c2w_ns: np.ndarray, fx, fy, cx, cy, W, H):
        from nerfstudio.cameras.cameras import Cameras, CameraType
        cam = Cameras(
            camera_to_worlds=torch.tensor(c2w_ns[:3, :4], dtype=torch.float32).unsqueeze(0),
            fx=torch.tensor([fx], dtype=torch.float32), fy=torch.tensor([fy], dtype=torch.float32),
            cx=torch.tensor([cx], dtype=torch.float32), cy=torch.tensor([cy], dtype=torch.float32),
            width=torch.tensor([W], dtype=torch.long), height=torch.tensor([H], dtype=torch.long),
            camera_type=CameraType.PERSPECTIVE,
        ).to(self.device)
        with torch.no_grad():
            out = self.pipeline.model.get_outputs_for_camera(cam)
        return (out["rgb"].clamp(0, 1).cpu().numpy(),
                out["depth"][..., 0].cpu().numpy(),
                out["accumulation"][..., 0].cpu().numpy())

    def train_depth(self, idx: int) -> np.ndarray:
        name, c2w, _ = self.train[idx]
        if self.depth_source is not None:
            if name not in self._depth_cache:
                fp = self.depth_source / (name + ".npy")
                # loud, not lazy: silently re-rendering 3DGS depth here would
                # make an "external depth" A/B secretly measure the baseline.
                if not fp.exists():
                    raise SystemExit(f"depth_source missing {fp.name} — reimport "
                                     f"{self.scene} with Analysis/18_import_depth.py")
                d = np.load(fp).astype(np.float32)
                if d.shape != (self.H_tr, self.W_tr):
                    raise SystemExit(f"{fp.name}: depth is {d.shape}, expected "
                                     f"{(self.H_tr, self.W_tr)} (train-image size)")
                self._depth_cache[name] = d
            return self._depth_cache[name]
        if name not in self._depth_cache:
            cache_dir = OUT / self.scene / f"depth_cache{self.cache_tag}"
            cache_dir.mkdir(parents=True, exist_ok=True)
            # The tag is "_bb" for EVERY backbone, so two arms on one scene share this
            # directory and silently read each other's depths. Not worth re-keying (it would
            # invalidate the shipped warm caches), but it must not stay silent: stamp the
            # owner and refuse a foreign one. Separate REPO roots remain the way to run
            # concurrent same-scene arms.
            owner = cache_dir / ".owner"
            me = str(getattr(self, "backbone_staging", None) or "") + "|" + str(self._cfg_path or "")
            if owner.exists():
                assert owner.read_text() == me, (
                    f"depth cache {cache_dir} belongs to another backbone:\n"
                    f"  cache: {owner.read_text()}\n  this run: {me}\n"
                    f"Use a separate REPO root (or delete the cache) -- reusing it corrupts "
                    f"every warp and the metrics will look plausible anyway.")
            else:
                owner.write_text(me)
            fp = cache_dir / (name + ".npy")
            if fp.exists():
                self._depth_cache[name] = np.load(fp)
            else:
                _, d, _ = self.render(c2w, self.f, self.f, self.cx, self.cy, self.W_tr, self.H_tr)
                np.save(fp, d.astype(np.float16))
                self._depth_cache[name] = d
        return self._depth_cache[name]

    def _target_depth(self, name: str, H: int, W: int) -> np.ndarray:
        """COLMAP-native depth_T: load the RaDe-GS test-pose depth for view
        `name` ({image_name}.npy, raw COLMAP z-forward). Loud on a miss, like
        train_depth()."""
        if name not in self._tgt_depth_cache:
            fp = self.target_depth_source / (name + ".npy")
            if not fp.exists():
                raise SystemExit(f"target_depth_source missing {fp.name} — render it "
                                 f"with radegs/render_test_depth_csv.py")
            d = np.load(fp).astype(np.float32)
            if d.shape != (H, W):
                raise SystemExit(f"{fp.name}: depth is {d.shape}, expected {(H, W)} "
                                 f"(test-pose size)")
            self._tgt_depth_cache[name] = d
        return self._tgt_depth_cache[name]

    def _sample_rgb(self, img: np.ndarray, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
        """RGB gather honoring self.sample ('bilinear' = original numpy path,
        'cubic' = cv2.remap INTER_CUBIC, clipped against overshoot)."""
        if self.sample == "cubic":
            import cv2
            out = cv2.remap(img.astype(np.float32),
                            us.astype(np.float32), vs.astype(np.float32),
                            interpolation=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REPLICATE)
            return np.clip(out, 0.0, 1.0)
        return bilinear(img, us, vs).astype(np.float32)

    def train_img(self, idx: int) -> np.ndarray:
        name = self.train[idx][0]
        if name not in self._img_cache:
            self._img_cache[name] = np.asarray(
                Image.open(self.train_dir / "images" / name).convert("RGB"),
                dtype=np.float32) / 255.0
        return self._img_cache[name]

    # ---- the core warp ----
    def _load_override_rgb(self, name, H, W):
        """Load the external (SSS) render for view `name`, as HxWx3 float [0,1]
        resized to (H, W). Used to replace the 3DGS rgb_T when render_override
        is set. `name` may or may not carry an extension."""
        import cv2
        d = self.render_override_dir
        cands = [d / name] + [d / f"{Path(name).stem}{e}" for e in (".png", ".jpg", ".jpeg", ".JPG")]
        p = next((c for c in cands if c.exists()), None)
        if p is None:
            raise SystemExit(f"override render for {name!r} not found in {d}")
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"cv2 failed to read override render {p}")
        img = img[:, :, ::-1]  # BGR -> RGB
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(img).astype(np.float32) / 255.0

    def synthesize(self, c2w_T: np.ndarray, fx, fy, cx, cy, W, H,
                   K=3, exclude_names=(), tol=0.03, min_w=1e-6, out_k=None,
                   guard=None, canvas_margin=0, rel_tol=None, flow_align=None,
                   exposure=False, override_name=None, k_blend=None, fill_only=False):
        """out_k: if set (SIMPLE_RADIAL k), the OUTPUT image is synthesized in
        distorted geometry (matching raw DJI GT, per X4): each output pixel's ray
        is the undistorted direction, and the pinhole-rendered fallback/depth
        maps are bilinearly sampled at the corresponding pinhole position.
        canvas_margin: with negative k the undistorted rays fall BEYOND the
        pinhole FOV, so the fallback/depth maps must be rendered on a
        (W+2m, H+2m) canvas (same focal, principal point shifted by m) or the
        periphery edge-replicates (the streaking script 08 fixed). Applies only
        when out_k is set; m=0 is the original behavior for positive-k scenes.
        rel_tol: exp036 (IBGS relative depth-consistency filter, arXiv 2511.14357).
        When set, the occlusion z-test becomes the scale-invariant per-pixel
        |z_neigh - z_cam| / (z_neigh + z_cam) < rel_tol instead of the absolute
        band tol*z + 1e-4. It is a per-pixel z-margin: near geometry gets a
        tighter tolerance, far geometry a looser one, so distant background does
        not leak while thin near structures are kept. None = original tol path.
        flow_align: exp039 (Wave 1). None disables. A dict
        {backend, max_px, searaft_ckpt} enabling flow-residual alignment of each
        warped neighbour to the 3DGS render before the guard/blend, recovering
        few-pixel-displaced real texture the guard would otherwise reject
        (train-free GADA-offset approximation, see Analysis/17_flow_align.py).
        exposure: exp040 (IBGS, arXiv 2511.14357). Drone frames are auto-exposed,
        so a neighbour can be globally brighter than the target view; the
        photometric guard then rejects correct texture for a reason that has
        nothing to do with geometry. Fits a per-channel gain+bias per neighbour
        on its own depth-consistent pixels (robust, closed-form) and applies it
        before the guard."""
        m = int(canvas_margin) if out_k is not None else 0
        ss = self.ss
        We, He = W + 2 * m, H + 2 * m
        if self.colmap_native:
            # No nerfstudio render: depth_T is the RaDe-GS test-pose map (pinhole,
            # native res); rgb_T is a placeholder that _load_override_rgb replaces
            # with the SSS render below; acc_T is fully accumulated (opaque).
            if out_k is not None or ss != 1:
                raise SystemExit("colmap_native synthesize requires out_k=None and ss=1")
            if override_name is None:
                raise SystemExit("colmap_native synthesize requires override_name")
            depth_T = self._target_depth(override_name, He, We)
            rgb_T = np.zeros((He, We, 3), np.float32)
            acc_T = np.ones((He, We), np.float32)
        else:
            rgb_T, depth_T, acc_T = self.render(c2w_T, fx * ss, fy * ss, (cx + m) * ss,
                                                (cy + m) * ss, We * ss, He * ss)

        u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
        x = (u - cx) / fx
        y = (v - cy) / fy
        if out_k is not None:
            xd, yd = x, y
            xu, yu = xd.copy(), yd.copy()
            for _ in range(5):
                r2 = xu * xu + yu * yu
                xu = xd / (1 + out_k * r2)
                yu = yd / (1 + out_k * r2)
            x, y = xu, yu
            us_pin, vs_pin = x * fx + cx + m, y * fy + cy + m
            if m and not (us_pin.min() >= 0 and us_pin.max() < We - 1
                          and vs_pin.min() >= 0 and vs_pin.max() < He - 1):
                print(f"  WARN canvas_margin={m} insufficient: "
                      f"u[{us_pin.min():.0f},{us_pin.max():.0f}] "
                      f"v[{vs_pin.min():.0f},{vs_pin.max():.0f}] (edge-clamped)")
        else:
            us_pin, vs_pin = u, v  # identity (m=0 on this path)
        if out_k is not None or ss > 1:
            # pixel-center-correct scaling: index_ss = ss*index_1x + (ss-1)/2
            # keeps the effective sampled ray identical to the ss=1 behavior.
            us_s = us_pin * ss + (ss - 1) / 2.0
            vs_s = vs_pin * ss + (ss - 1) / 2.0
            rgb_T = self._sample_rgb(rgb_T, us_s, vs_s)
            depth_T = bilinear(depth_T[..., None], us_s, vs_s)[..., 0].astype(np.float32)
            acc_T = bilinear(acc_T[..., None], us_s, vs_s)[..., 0].astype(np.float32)
        # SSS-backbone gate-2: swap the 3DGS render for the external render at
        # the output grid. rgb_T now feeds the refiner's render channel AND the
        # DIBR fallback/exposure reference; depth_T stays from the 3DGS backbone.
        if override_name is not None and self.render_override_dir is not None:
            rgb_T = self._load_override_rgb(override_name, rgb_T.shape[0], rgb_T.shape[1])
        d = depth_T.astype(np.float64)
        # OpenGL camera coords: x right, y up, z backward; z-depth is +d forward
        P_cam = np.stack([x * d, -y * d, -d], axis=-1)  # (H,W,3)
        Xw = P_cam @ c2w_T[:3, :3].T + c2w_T[:3, 3]

        center_T = c2w_T[:3, 3]
        cand = np.argsort(np.linalg.norm(self.centers - center_T, axis=1))
        # k_blend (>= K) widens the BLEND only. Measured on bonsai (2026-07-25): in the
        # starved early region 38.7% of pixels fall back to the plain render, and loosening
        # the photometric guard moved that by 0.8pp -- so the loss is geometric COVERAGE
        # (no neighbour sees those pixels), not photometric rejection. More neighbours is
        # the only thing that can fill them. The EVIDENCE stack still packs the nearest K
        # (4K+1 channels), so the refiner's input width is unchanged.
        K_use = max(K, k_blend or K)
        neigh = [i for i in cand if self.train[i][0] not in exclude_names][:K_use]
        # fill_only: extra neighbours (beyond K) may ONLY write pixels the nearest-K blend
        # leaves uncovered. Measured on bonsai 2026-07-25: widening the blend outright raised
        # coverage (starved 38.7->33.7% fallback) but LOST quality in BOTH regions
        # (starved Score -0.0110, dense -0.0076) because 1/distance weighting dilutes good
        # near texture with bad far texture on already-covered pixels. Restricting the extras
        # to uncovered pixels cannot, by construction, damage a covered one.

        num = np.zeros((H, W, 3)); den = np.zeros((H, W, 1))
        num_x = np.zeros((H, W, 3)); den_x = np.zeros((H, W, 1))  # fill_only: extras go here
        # exp039 instrumentation: how many depth-consistent warped samples does
        # the photometric guard throw away? That rejection rate is the mechanism
        # flow alignment is supposed to move (GADA's 33% -> 79% analogue), so we
        # measure it directly rather than inferring it from the Score.
        n_depth_ok = n_guard_kept = n_flow_applied = 0
        ev_cols, ev_confs = [], []  # exp040 per-neighbour evidence
        for j, i in enumerate(neigh):
            name, c2w_N, cen_N = self.train[i]
            w2c = np.linalg.inv(c2w_N)
            Pc = Xw @ w2c[:3, :3].T + w2c[:3, 3]
            xc, yc, zc = Pc[..., 0], -Pc[..., 1], -Pc[..., 2]  # OpenCV coords
            valid = zc > 1e-6
            zs = np.where(valid, zc, 1.0)
            xn, yn = xc / zs, yc / zs
            # undistorted pinhole coords (for the neighbor's rendered depth)
            uu = xn * self.f + self.cx
            vu = yn * self.f + self.cy
            # SIMPLE_RADIAL distorted coords (for the RAW jpg)
            r2 = xn * xn + yn * yn
            ud = xn * (1 + self.k * r2) * self.f + self.cx
            vd = yn * (1 + self.k * r2) * self.f + self.cy
            inb = (valid & (uu >= 1) & (uu < self.W_tr - 2) & (vu >= 1) & (vu < self.H_tr - 2)
                   & (ud >= 1) & (ud < self.W_tr - 2) & (vd >= 1) & (vd < self.H_tr - 2))
            dN = self.train_depth(i)
            zn = bilinear(dN[..., None], uu, vu)[..., 0]
            if rel_tol is not None:
                # IBGS relative depth-consistency test (exp036): scale-invariant
                rel = np.abs(zn - zc) / (zn + zc + 1e-6)
                visible = inb & (rel < rel_tol)
                agree = 1.0 - rel / rel_tol
            else:
                band = tol * zc + 1e-4
                visible = inb & (np.abs(zn - zc) < band)
                agree = 1.0 - np.abs(zn - zc) / np.maximum(band, 1e-9)
            n_depth_ok += int(visible.sum())
            col = self._sample_rgb(self.train_img(i), ud, vd)
            if exposure:
                col = _fit_exposure(col, rgb_T, visible)
            if flow_align is not None:
                # align the warped neighbour to the 3DGS render (few-px offset
                # recovery) BEFORE the guard, so displaced-but-valid texture is
                # snapped back into agreement instead of rejected.
                fa = _flow_align_mod()
                col, applied = fa.align_to_reference(
                    col, rgb_T, max_px=flow_align.get("max_px", 7.0),
                    backend=flow_align.get("backend", "dis"),
                    searaft_ckpt=flow_align.get("searaft_ckpt"))
                n_flow_applied += int((applied > 0.5)[visible].sum())
            if guard is not None:
                # photometric guard: reject warped samples that disagree with the
                # (aligned, if blurry) 3DGS render — kills thin-structure ghosting
                # from unreliable expected-depth. Biases toward the 3DGS baseline,
                # making DIBR >= baseline by construction.
                visible = visible & (np.abs(col - rgb_T).mean(axis=-1) < guard)
            n_guard_kept += int(visible.sum())
            wgt = 1.0 / (np.linalg.norm(cen_N - center_T) + 1e-6)
            wmap = (visible.astype(np.float64) * wgt)[..., None]
            if fill_only and j >= K:
                num_x += col * wmap      # extras stay quarantined until the blend below
                den_x += wmap
            else:
                num += col * wmap
                den += wmap
            if getattr(self, "_return_evidence", False) and j < K:
                # exp040: hand the refiner the UNBLENDED per-neighbour evidence
                # (aligned warp + its depth-agreement confidence) instead of only
                # the collapsed weighted mean, so the net can learn WHICH
                # neighbour to trust per pixel rather than inheriting our
                # hand-tuned 1/distance weighting.
                ev_cols.append(col.astype(np.float32))
                ev_confs.append((visible * np.clip(agree, 0, 1)).astype(np.float32))

        if fill_only:
            # only where the nearest-K blend produced nothing does the extra pool get a vote
            gap = (den[..., 0] <= min_w) & (den_x[..., 0] > min_w)
            num = np.where(gap[..., None], num_x, num)
            den = np.where(gap[..., None], den_x, den)
        warped = num / np.maximum(den, min_w)
        have = (den[..., 0] > min_w) & (acc_T > 0.5)
        # feather the fallback boundary to avoid hard seams
        import cv2
        alpha = cv2.blur(have.astype(np.float32), (7, 7))[..., None]
        out = alpha * np.where(have[..., None], warped, rgb_T) + (1 - alpha) * rgb_T
        out = np.clip(out, 0, 1)
        self.last_stats = {
            "depth_ok_frac": n_depth_ok / max(H * W * max(len(neigh), 1), 1),
            "guard_reject_frac": (1 - n_guard_kept / n_depth_ok) if n_depth_ok else 0.0,
            "flow_applied_frac": (n_flow_applied / n_depth_ok) if n_depth_ok else 0.0,
        }
        if getattr(self, "_return_evidence", False):
            ev = self._pack_evidence(ev_cols, ev_confs, depth_T, K, H, W)
            return out, float(1 - have.mean()), rgb_T, have.astype(np.float32), ev
        if getattr(self, "_return_mask", False):
            return out, float(1 - have.mean()), rgb_T, have.astype(np.float32)
        return out, float(1 - have.mean()), rgb_T

    @staticmethod
    def _pack_evidence(cols, confs, depth_T, K, H, W):
        """exp040 evidence channels: [warp_0..warp_{K-1} (3 each) |
        conf_0..conf_{K-1} (1 each) | normalised 3DGS depth (1)] = 4K+1.
        Neighbours are distance-ordered by `neigh`, so slot i means the same
        thing (i-th nearest) on every view -- the net can learn a per-slot prior.
        Scenes with fewer than K usable neighbours get zero-filled slots, which
        read as zero-confidence evidence."""
        ch = []
        for i in range(K):
            ch.append(cols[i] if i < len(cols) else np.zeros((H, W, 3), np.float32))
        for i in range(K):
            ch.append((confs[i] if i < len(confs)
                       else np.zeros((H, W), np.float32))[..., None])
        # depth is metric and scene-scale-dependent; normalise by the view median
        # so one refiner hyper-parameter set transfers across scenes.
        d = depth_T.astype(np.float32)
        med = float(np.median(d[d > 0])) if (d > 0).any() else 1.0
        ch.append(np.clip(d / (2 * med + 1e-9), 0, 1)[..., None])
        return np.concatenate(ch, axis=-1).astype(np.float32)


def _fit_exposure(col, ref, mask, min_px=512, max_gain=1.25):
    """exp040/IBGS: per-channel affine photometric alignment of a warped
    neighbour to the 3DGS render, fitted ONLY on `mask` (the depth-consistent
    pixels) so occluded/wrong-depth junk cannot drag the fit.

    Deliberately global-affine and clamped, not per-pixel: a per-pixel fit would
    just reproduce the render (erasing the real texture we warped in) and make
    the guard vacuous. Gain is clamped to max_gain and the fit is skipped on thin
    evidence — both keep a bad fit from being worse than no correction."""
    m = mask if mask.dtype == bool else mask > 0.5
    if int(m.sum()) < min_px:
        return col
    out = col.copy()
    for c in range(col.shape[-1]):
        x, y = col[..., c][m], ref[..., c][m]
        vx = float(x.var())
        if vx < 1e-6:
            continue
        g = float(np.cov(x, y)[0, 1] / vx)
        g = float(np.clip(g, 1.0 / max_gain, max_gain))
        b = float(y.mean() - g * x.mean())
        out[..., c] = np.clip(g * col[..., c] + b, 0, 1)
    return out


def bilinear(img: np.ndarray, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
    H, W = img.shape[:2]
    u0 = np.clip(np.floor(us).astype(int), 0, W - 2)
    v0 = np.clip(np.floor(vs).astype(int), 0, H - 2)
    du = np.clip(us - u0, 0, 1)[..., None]
    dv = np.clip(vs - v0, 0, 1)[..., None]
    return (img[v0, u0] * (1 - du) * (1 - dv) + img[v0, u0 + 1] * du * (1 - dv)
            + img[v0 + 1, u0] * (1 - du) * dv + img[v0 + 1, u0 + 1] * du * dv)


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(((a - b) ** 2).mean(), 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="hcm0034")
    ap.add_argument("--mode", choices=["traincheck", "test"], default="traincheck")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--tol", type=float, default=0.03)
    ap.add_argument("--rel-tol", type=float, default=None,
                    help="exp036 IBGS relative depth-consistency filter "
                         "|zn-zc|/(zn+zc) < rel_tol (e.g. 1e-3); overrides --tol")
    ap.add_argument("--flow-align", choices=["off", "dis", "searaft"], default="off",
                    help="exp039 flow-residual alignment of warped neighbours "
                         "before the guard (dis=classical/weightless, searaft=needs ckpt)")
    ap.add_argument("--flow-max-px", type=float, default=7.0,
                    help="clamp on the alignment flow magnitude (GADA sigma_max=7)")
    ap.add_argument("--searaft-ckpt", default=None, help="SEA-RAFT checkpoint (flow-align=searaft)")
    ap.add_argument("--n-check", type=int, default=5)
    ap.add_argument("--no-distort", action="store_true",
                    help="output pinhole geometry instead of SIMPLE_RADIAL (pre-X4 behavior)")
    ap.add_argument("--guard", type=float, default=None,
                    help="photometric guard threshold (mean-abs channel diff vs 3DGS render); "
                         "e.g. 0.18. None disables.")
    ap.add_argument("--canvas-margin", type=int, default=0,
                    help="expand the 3DGS fallback canvas by this many px per side (out_k only); "
                         "0 auto-bumps to 128 for strongly-negative k (HNI0131/HNI0265).")
    ap.add_argument("--ss", type=int, default=1, help="supersample factor for the 3DGS canvas")
    ap.add_argument("--sample", choices=["bilinear", "cubic"], default="bilinear",
                    help="train-pixel gather interpolation")
    ap.add_argument("--config", default=None, help="override backbone checkpoint dir/config.yml")
    ap.add_argument("--exposure", action="store_true",
                    help="exp040/IBGS per-neighbour affine exposure correction "
                         "before the guard (auto-exposed drone frames)")
    ap.add_argument("--depth-source", default=None,
                    help="exp041: dir of imported per-train-view depth .npy "
                         "(RaDe-GS/PGSR) to use for the occlusion z-test instead "
                         "of 3DGS expected depth; see Analysis/18_import_depth.py")
    ap.add_argument("--vtag", default="", help="extra output-dir tag for A/B variants")
    args = ap.parse_args()

    flow_align = None if args.flow_align == "off" else {
        "backend": args.flow_align, "max_px": args.flow_max_px,
        "searaft_ckpt": args.searaft_ckpt}
    if args.flow_align == "searaft" and not args.searaft_ckpt:
        ap.error("--flow-align searaft requires --searaft-ckpt")

    w = Warper(args.scene, config_path=args.config, ss=args.ss, sample=args.sample,
               depth_source=args.depth_source)
    out_k = None if args.no_distort else w.k
    cmargin = args.canvas_margin
    if out_k is not None and out_k < -0.05 and cmargin == 0:
        cmargin = 128  # negative-k FOV expansion (see script 08)
        print(f"[canvas_margin auto-set to {cmargin} for k={out_k:.4f}]")

    if args.mode == "traincheck":
        # hold out every ~48th train view, warp its neighbors into it, score center crop
        rows = []
        for idx in range(10, len(w.train), max(1, len(w.train) // args.n_check))[:args.n_check]:
            name, c2w, _ = w.train[idx]
            gt = w.train_img(idx)
            out, fallback_frac, rgb_T = w.synthesize(
                c2w, w.f, w.f, w.cx, w.cy, w.W_tr, w.H_tr,
                K=args.K, exclude_names={name}, tol=args.tol, out_k=out_k,
                guard=args.guard, canvas_margin=cmargin, rel_tol=args.rel_tol,
                flow_align=flow_align, exposure=args.exposure)
            H, W_ = gt.shape[:2]
            sl = (slice(int(H * .2), int(H * .8)), slice(int(W_ * .2), int(W_ * .8)))
            p_warp, p_3dgs = psnr(out[sl], gt[sl]), psnr(rgb_T[sl], gt[sl])
            st = w.last_stats
            rows.append((name, p_warp, p_3dgs, fallback_frac, st))
            print(f"{name}: center-PSNR warp={p_warp:.2f} vs 3dgs={p_3dgs:.2f} "
                  f"(fallback {fallback_frac*100:.1f}%, guard-reject "
                  f"{st['guard_reject_frac']*100:.1f}%, flow-applied "
                  f"{st['flow_applied_frac']*100:.1f}%)")
            d = OUT / args.scene / "traincheck"
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray((out * 255).astype(np.uint8)).save(d / f"warp_{name}")
        mean_w = np.mean([r[1] for r in rows]); mean_g = np.mean([r[2] for r in rows])
        print(f"\nMEAN center-PSNR: warp={mean_w:.2f} vs 3dgs={mean_g:.2f} "
              f"({'WARP WINS' if mean_w > mean_g else 'warp loses'})")
        print(f"MEAN guard-reject={np.mean([r[4]['guard_reject_frac'] for r in rows])*100:.1f}% "
              f"flow-applied={np.mean([r[4]['flow_applied_frac'] for r in rows])*100:.1f}% "
              f"(flow_align={args.flow_align})")
        return

    # test mode: full 60 views + scoring
    from src.metrics import compute_metrics
    rows = load_test_poses(w.scene_dir / "test/test_poses.csv")
    tag = (f"_g{args.guard}" if args.guard is not None else "")
    if args.rel_tol is not None:
        tag += f"_rt{args.rel_tol:g}"
    if args.K != 3:
        tag += f"_K{args.K}"
    if flow_align is not None:
        tag += f"_fa{args.flow_align}{args.flow_max_px:g}"
    if args.depth_source:
        tag += "_dsrc"
    tag += args.vtag
    rdir = OUT / args.scene / f"renders{tag}"
    rdir.mkdir(parents=True, exist_ok=True)
    fb, grj = [], []
    for r in rows:
        c2w = w._c2w_ns(r["qvec"], r["tvec"])
        out, fallback_frac, _ = w.synthesize(c2w, r["fx"], r["fy"], r["cx"], r["cy"],
                                             r["width"], r["height"], K=args.K, tol=args.tol,
                                             out_k=out_k, guard=args.guard, canvas_margin=cmargin,
                                             rel_tol=args.rel_tol, flow_align=flow_align,
                                             exposure=args.exposure)
        fb.append(fallback_frac)
        grj.append(w.last_stats["guard_reject_frac"])
        Image.fromarray((out * 255).astype(np.uint8)).save(rdir / r["image_name"], quality=98)
        print(f"{r['image_name']} fallback={fallback_frac*100:.1f}%")
    gt_dir = w.scene_dir / "test/images"
    if not gt_dir.exists():
        # private scene: no test GT to score against; renders are the deliverable
        (OUT / args.scene / f"metrics{tag}.json").write_text(json.dumps(
            {"mean": None, "mean_fallback_frac": float(np.mean(fb)),
             "mean_guard_reject_frac": float(np.mean(grj)),
             "K": args.K, "tol": args.tol, "guard": args.guard,
             "rel_tol": args.rel_tol, "flow_align": args.flow_align,
             "flow_max_px": args.flow_max_px,
             "note": "private scene, no GT"}, indent=2))
        print(f"\n{args.scene} DIBR: {len(rows)} views rendered to {rdir} "
              f"(no GT; mean fallback {np.mean(fb)*100:.1f}%)")
        return
    res = compute_metrics(rdir, gt_dir, "vgg", 50.0)
    m = res["mean"]
    (OUT / args.scene / f"metrics{tag}.json").write_text(json.dumps(
        {"mean": m, "mean_fallback_frac": float(np.mean(fb)),
         "mean_guard_reject_frac": float(np.mean(grj)),
         "K": args.K, "tol": args.tol, "guard": args.guard,
         "rel_tol": args.rel_tol, "flow_align": args.flow_align,
         "flow_max_px": args.flow_max_px}, indent=2))
    print(f"\n{args.scene} DIBR: PSNR={m['psnr']:.3f} SSIM={m['ssim']:.4f} "
          f"LPIPS={m['lpips']:.4f} Score={m['score']:.4f} "
          f"(mean fallback {np.mean(fb)*100:.1f}%, "
          f"guard-reject {np.mean(grj)*100:.1f}%)")


if __name__ == "__main__":
    main()
