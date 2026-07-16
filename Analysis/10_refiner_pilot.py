"""Analysis 10: P2 per-scene neural blending refiner — PILOT (hcm0034/HCM0181).

Deep-Blending/SVS-style recipe, fully compliant (trains only on provided train
views + our own 3DGS model; no test images, no external data):

  input  (7ch): [ F1-remap 3DGS render (3) | DIBR blended output (3) | visibility mask (1) ]
  target (3ch): the real RAW/distorted train image
  net         : small U-Net, residual on top of the DIBR output (starts = DIBR)
  loss        : the grader's own objective  0.4*LPIPS_vgg + 0.3*(1-SSIM) + 0.3*L1
  supervision : leave-one-out over train views (neighbors exclude the target view),
                so every training pair mimics a novel test view.

Stage 1 caches per-train-view (input,target) tiles to disk; stage 2 trains; stage 3
applies to the TEST poses and (public scenes) scores vs GT to compare against DIBR.

Run: conda run -n airace python Analysis/10_refiner_pilot.py --scene hcm0034 --iters 3000
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "Analysis"

# import the DIBR Warper from 04 (module name starts with a digit -> load by path)
_spec = importlib.util.spec_from_file_location("dibr04", ANALYSIS / "04_x3_dibr_pilot.py")
dibr04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dibr04)
Warper, load_test_poses = dibr04.Warper, dibr04.load_test_poses

OUT = ANALYSIS / "X5_refiner"


# ----------------------------- differentiable SSIM -----------------------------
def _gauss_window(ch, ws=11, sigma=1.5, device="cpu"):
    coords = torch.arange(ws, dtype=torch.float32) - ws // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    w = (g[:, None] @ g[None, :])[None, None]
    return w.expand(ch, 1, ws, ws).contiguous().to(device)


def ssim(a, b, window):
    ch = a.shape[1]
    mu_a = F.conv2d(a, window, padding=5, groups=ch)
    mu_b = F.conv2d(b, window, padding=5, groups=ch)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = F.conv2d(a * a, window, padding=5, groups=ch) - mu_a2
    sb = F.conv2d(b * b, window, padding=5, groups=ch) - mu_b2
    sab = F.conv2d(a * b, window, padding=5, groups=ch) - mu_ab
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_ab + c1) * (2 * sab + c2)) / ((mu_a2 + mu_b2 + c1) * (sa + sb + c2))
    return s.mean()


# ----------------------------------- U-Net -------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU(),
            nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU())

    def forward(self, x):
        return self.net(x)


class _LayerNorm2d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.g = nn.Parameter(torch.ones(c))
        self.b = nn.Parameter(torch.zeros(c))

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + 1e-6)
        return x * self.g[None, :, None, None] + self.b[None, :, None, None]


def _sg(x):  # SimpleGate: split channels in half, multiply
    a, b = x.chunk(2, dim=1)
    return a * b


class _NAFBlock(nn.Module):
    """One NAFNet block (Chen et al., ECCV22): activation-free, SimpleGate +
    simplified channel attention, two residual sub-blocks (spatial + FFN)."""
    def __init__(self, c, dw_expand=2, ffn_expand=2):
        super().__init__()
        dc, fc = c * dw_expand, c * ffn_expand
        self.norm1 = _LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dc, 1)
        self.conv2 = nn.Conv2d(dc, dc, 3, padding=1, groups=dc)  # depthwise
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(dc // 2, dc // 2, 1))
        self.conv3 = nn.Conv2d(dc // 2, c, 1)
        self.norm2 = _LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, fc, 1)
        self.conv5 = nn.Conv2d(fc // 2, c, 1)
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x):
        p = _sg(self.conv2(self.conv1(self.norm1(x))))
        y = self.conv3(p * self.sca(p))
        x = x + y * self.beta
        z = self.conv5(_sg(self.conv4(self.norm2(x))))
        return x + z * self.gamma


class NAFDoubleBlock(nn.Module):
    """exp040 refiner v3 backbone block: two stacked NAFNet blocks at width `co`.
    Drop-in for DoubleConv (same ci->co contract) so the UNet skeleton + skip
    connections are unchanged."""
    def __init__(self, ci, co, dw_expand=2, ffn_expand=2):
        super().__init__()
        self.proj_in = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()
        self.blocks = nn.ModuleList([_NAFBlock(co, dw_expand, ffn_expand)
                                     for _ in range(2)])

    def forward(self, x):
        x = self.proj_in(x)
        for blk in self.blocks:
            x = blk(x)
        return x


def _block(kind):
    return NAFDoubleBlock if kind == "naf" else DoubleConv


class UNet(nn.Module):
    def __init__(self, ci=7, co=3, base=32, blocks="conv"):
        super().__init__()
        B = _block(blocks)
        self.blocks_kind = blocks
        self.ci = ci
        self.base = base
        self.d1 = B(ci, base)
        self.d2 = B(base, base * 2)
        self.d3 = B(base * 2, base * 4)
        self.bott = B(base * 4, base * 4)
        self.u3 = B(base * 4 + base * 4, base * 2)
        self.u2 = B(base * 2 + base * 2, base)
        self.u1 = B(base + base, base)
        self.head = nn.Conv2d(base, co, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        b = self.bott(self.pool(c3))
        b = F.interpolate(b, size=c3.shape[-2:], mode="nearest")
        u3 = self.u3(torch.cat([b, c3], 1))
        u3 = F.interpolate(u3, size=c2.shape[-2:], mode="nearest")
        u2 = self.u2(torch.cat([u3, c2], 1))
        u2 = F.interpolate(u2, size=c1.shape[-2:], mode="nearest")
        u1 = self.u1(torch.cat([u2, c1], 1))
        return torch.tanh(self.head(u1))  # residual in [-1,1]


def save_refiner(net: "UNet", path):
    """Wrapped checkpoint carrying architecture meta so any ci/base/blocks
    combination (v3 NAFNet, extra evidence channels) reloads unambiguously."""
    torch.save({"sd": net.state_dict(), "ci": net.ci, "base": net.base,
                "blocks": net.blocks_kind, "fmt": "refiner-v3"}, path)


def load_refiner(path, device):
    """Loads both wrapped v3 checkpoints and legacy raw-state_dict v1/v2 ones
    (conv blocks, base inferred from d1.net.0.weight, ci from its in-channels)."""
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and obj.get("fmt") == "refiner-v3":
        net = UNet(ci=obj["ci"], base=obj["base"], blocks=obj["blocks"]).to(device)
        net.load_state_dict(obj["sd"])
        return net
    # legacy raw state_dict (conv U-Net)
    w = obj["d1.net.0.weight"]
    net = UNet(ci=w.shape[1], base=w.shape[0], blocks="conv").to(device)
    net.load_state_dict(obj)
    return net


# ------------------------------ stage 1: cache pairs ---------------------------
def stack_channels(rgb_T, dibr, mask, evidence=None):
    """Refiner input stack. v2 = [render 3 | DIBR 3 | have-mask 1] = 7ch.
    v3 (exp040) appends the per-neighbour evidence (4K+1 ch). The v2 prefix is
    kept verbatim -- _net_apply takes the residual base from channels 3:6, so
    DIBR must stay there, and a v2 checkpoint keeps reading the same channels."""
    ch = [rgb_T, dibr, mask[..., None]]
    if evidence is not None:
        ch.append(evidence)
    return np.concatenate(ch, axis=-1).astype(np.float16)


def build_pairs(scene, K=3, tol=0.03, guard=0.18, warper_kw=None, variant="",
                max_pairs=0, evidence=False, rel_tol=None, flow_align=None):
    cache = OUT / scene / f"pairs{variant}"
    cache.mkdir(parents=True, exist_ok=True)
    w = Warper(scene, **(warper_kw or {}))
    w._return_mask = True
    w._return_evidence = evidence
    # max_pairs: cap the number of leave-one-out pairs (weak-CPU hosts, e.g.
    # Kaggle). Stride-sampled over the flight order so the crops still cover
    # the whole trajectory rather than one corner of it.
    keep = None
    if max_pairs and max_pairs < len(w.train):
        idxs = np.linspace(0, len(w.train) - 1, max_pairs).round().astype(int)
        keep = {w.train[i][0] for i in idxs}
    out_k = w.k
    cmargin = 128 if out_k < -0.05 else 0
    made = 0
    for idx in range(len(w.train)):
        name, c2w, _ = w.train[idx]
        if keep is not None and name not in keep:
            continue
        fp = cache / f"{name}.npz"
        if fp.exists():
            continue
        res = w.synthesize(
            c2w, w.f, w.f, w.cx, w.cy, w.W_tr, w.H_tr,
            K=K, exclude_names={name}, tol=tol, out_k=out_k,
            guard=guard, canvas_margin=cmargin, rel_tol=rel_tol,
            flow_align=flow_align)
        (dibr, _, rgb_T, mask), ev = (res[:4], res[4] if evidence else None)
        tgt = w.train_img(idx)  # real distorted train image, HxWx3 float [0,1]
        inp = stack_channels(rgb_T, dibr, mask, ev)
        np.savez_compressed(fp, inp=inp, tgt=tgt.astype(np.float16))
        made += 1
        if made % 20 == 0:
            print(f"  {scene}: cached {made} pairs")
    names = sorted(p.stem for p in cache.glob("*.npz"))
    if keep is not None:
        names = [n for n in names if n in keep]
    print(f"{scene}: {len(names)} training pairs ready ({made} new)")
    return cache, names, w


# ------------------------------- stage 2: train --------------------------------
def train(scene, cache, names, device, iters=3000, crop=256, bs=4, lr=2e-4, val_frac=0.15,
          base=32, ema=0.0, suffix="", seed=0, blocks="conv"):
    import lpips
    # The train/val SPLIT is seeded separately (fixed) from the weight init +
    # batch order (seed): a seed-ensemble (reserve R2) must average nets that
    # saw the SAME held-out val set, so their val_loss is comparable and no
    # member leaks a val crop into fit. Only init/sampling vary across members.
    split_rng = np.random.default_rng(0)
    order = split_rng.permutation(len(names))
    n_val = max(2, int(len(names) * val_frac))
    val_names = [names[i] for i in order[:n_val]]
    fit_names = [names[i] for i in order[n_val:]]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    def load(nm):
        d = np.load(cache / f"{nm}.npz")
        return d["inp"].astype(np.float32), d["tgt"].astype(np.float32)

    fit = [load(n) for n in fit_names]
    val = [load(n) for n in val_names]

    # channel count comes from the cache, not a constant: v2 pairs are 7ch and
    # exp040 evidence pairs are 7+4K+1, and the net must match whatever the
    # variant cache actually holds.
    ci = fit[0][0].shape[-1]
    net = UNet(ci=ci, base=base, blocks=blocks).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=lr * 0.05)
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)
    win = _gauss_window(3, device=device)

    def grader_loss(pred, tgt):
        l1 = F.l1_loss(pred, tgt)
        s = ssim(pred, tgt, win)
        lp = lpips_fn(pred * 2 - 1, tgt * 2 - 1).mean()
        return 0.4 * lp + 0.3 * (1 - s) + 0.3 * l1, (lp.item(), s.item(), l1.item())

    def sample_batch(pool, rng_):
        xs, ys = [], []
        for _ in range(bs):
            inp, tgt = pool[rng_.integers(len(pool))]
            H, W = tgt.shape[:2]
            y0, x0 = rng_.integers(0, H - crop), rng_.integers(0, W - crop)
            xi = inp[y0:y0 + crop, x0:x0 + crop].transpose(2, 0, 1)
            yi = tgt[y0:y0 + crop, x0:x0 + crop].transpose(2, 0, 1)
            if rng_.random() < 0.5:  # h-flip aug
                xi, yi = xi[:, :, ::-1].copy(), yi[:, :, ::-1].copy()
            xs.append(xi); ys.append(yi)
        return (torch.from_numpy(np.stack(xs)).to(device),
                torch.from_numpy(np.stack(ys)).to(device))

    # deterministic val batches: same crops every eval -> best-ckpt selection is
    # signal, not sampling noise (v1 resampled random crops each eval).
    vrng = np.random.default_rng(1234)
    val_batches = [sample_batch(val, vrng) for _ in range(8)]
    val_batches = [(x.cpu(), y.cpu()) for x, y in val_batches]

    ema_state = ({k: v.detach().clone() for k, v in net.state_dict().items()}
                 if ema > 0 else None)

    def eval_val(state=None):
        backup = None
        if state is not None:
            backup = {k: v.detach().clone() for k, v in net.state_dict().items()}
            net.load_state_dict(state)
        net.eval()
        with torch.no_grad():
            vl = []
            for xv, yv in val_batches:
                xv, yv = xv.to(device), yv.to(device)
                pv = (xv[:, 3:6] + net(xv)).clamp(0, 1)
                vl.append(grader_loss(pv, yv)[0].item())
        if backup is not None:
            net.load_state_dict(backup)
        return float(np.mean(vl))

    best_val, best_state = 1e9, None
    for it in range(1, iters + 1):
        net.train()
        x, y = sample_batch(fit, rng)
        dibr = x[:, 3:6]
        pred = (dibr + net(x)).clamp(0, 1)
        loss, _ = grader_loss(pred, y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if ema_state is not None:
            with torch.no_grad():
                msd = net.state_dict()
                for k in ema_state:
                    ema_state[k].mul_(ema).add_(msd[k].float(), alpha=1 - ema)
        if it % 250 == 0 or it == iters:
            vloss = eval_val(ema_state)
            print(f"  it {it}: train_loss {loss.item():.4f}  val_loss {vloss:.4f}")
            if vloss < best_val:
                src = ema_state if ema_state is not None else net.state_dict()
                best_val, best_state = vloss, {k: v.detach().cpu().clone()
                                               for k, v in src.items()}
    net.load_state_dict(best_state)
    torch.save(net.state_dict(), OUT / scene / f"refiner{suffix}.pt")
    print(f"{scene}: best val_loss {best_val:.4f} -> saved refiner{suffix}.pt")
    return net


# ---------------------- stage 3: apply to test + score -------------------------
def _net_apply(net, inp, device, tta=False):
    """inp: (C,H,W) float. Residual-on-DIBR forward, optional hflip TTA."""
    _, H, W = inp.shape
    ph, pw = (-H) % 8, (-W) % 8  # pad to a multiple of 8 for the U-Net
    x = torch.from_numpy(inp[None].astype(np.float32)).to(device)
    x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    with torch.no_grad():
        pred = (x[:, 3:6] + net(x)).clamp(0, 1)
        if tta:
            xf = torch.flip(x, dims=[3])
            pf = (xf[:, 3:6] + net(xf)).clamp(0, 1)
            pred = 0.5 * (pred + torch.flip(pf, dims=[3]))
    return pred[0, :, :H, :W].cpu().numpy().transpose(1, 2, 0)


def _net_apply_ensemble(nets, inp, device, tta=False):
    """Reserve R2: average the residual-on-DIBR outputs of several seed members.
    All members share the DIBR base (x[:,3:6]); averaging their residuals is the
    apply-time-only cost. Equivalent to averaging the final RGB since the base
    is common."""
    _, H, W = inp.shape
    ph, pw = (-H) % 8, (-W) % 8
    x = torch.from_numpy(inp[None].astype(np.float32)).to(device)
    x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    base = x[:, 3:6]
    with torch.no_grad():
        res = torch.stack([n(x) for n in nets], 0).mean(0)
        pred = (base + res).clamp(0, 1)
        if tta:
            xf = torch.flip(x, dims=[3])
            resf = torch.stack([n(xf) for n in nets], 0).mean(0)
            pf = (xf[:, 3:6] + resf).clamp(0, 1)
            pred = 0.5 * (pred + torch.flip(pf, dims=[3]))
    return pred[0, :, :H, :W].cpu().numpy().transpose(1, 2, 0)


def apply_test(scene, net, device, K=3, tol=0.03, guard=0.18,
               warper_kw=None, variant="", suffix="", tta=False, png=False,
               evidence=False, rel_tol=None, flow_align=None):
    from src.metrics import compute_metrics
    scene_dir = dibr04.scene_raw(scene) / scene
    rows = load_test_poses(scene_dir / "test/test_poses.csv")
    icache = OUT / scene / f"test_inputs{variant}"
    icache.mkdir(parents=True, exist_ok=True)
    rdir = OUT / scene / f"renders_refined{suffix}"
    rdir.mkdir(parents=True, exist_ok=True)
    net.eval()
    # build (or reuse) the warp inputs; Warper is only constructed on a miss
    w = None
    for r in rows:
        fp = icache / (r["image_name"] + ".npz")
        if fp.exists():
            continue
        if w is None:
            w = Warper(scene, **(warper_kw or {}))
            w._return_mask = True
            w._return_evidence = evidence
        out_k = w.k
        cmargin = 128 if out_k < -0.05 else 0
        c2w = w._c2w_ns(r["qvec"], r["tvec"])
        res = w.synthesize(
            c2w, r["fx"], r["fy"], r["cx"], r["cy"], r["width"], r["height"],
            K=K, tol=tol, out_k=out_k, guard=guard, canvas_margin=cmargin,
            rel_tol=rel_tol, flow_align=flow_align)
        (dibr, _, rgb_T, mask), ev = (res[:4], res[4] if evidence else None)
        np.savez_compressed(fp, inp=stack_channels(rgb_T, dibr, mask, ev))
    for r in rows:
        inp = np.load(icache / (r["image_name"] + ".npz"))["inp"].astype(np.float32)
        if inp.shape[-1] != net.ci:
            raise SystemExit(
                f"{scene}: cache {icache.name} holds {inp.shape[-1]}ch but the net "
                f"wants {net.ci}ch — stale cache for this variant. Delete it or "
                f"pass the --variant/--evidence the checkpoint was trained with.")
        img = _net_apply(net, inp.transpose(2, 0, 1), device, tta=tta)
        pil = Image.fromarray((img * 255).astype(np.uint8))
        if png:  # lossless hand-off; the submission builder single-encodes JPEG
            import os as _os
            pil.save(rdir / (_os.path.splitext(r["image_name"])[0] + ".png"))
        else:
            pil.save(rdir / r["image_name"], quality=98)
    gt = scene_dir / "test/images"
    if gt.exists() and png:
        print(f"\n{scene}: PNG renders written to {rdir} (GT present but scoring "
              f"needs JPG names — score via the builder or rerun without --png)")
        return None
    if gt.exists():
        m = compute_metrics(rdir, gt, "vgg", 50.0)["mean"]
        (OUT / scene / f"metrics_refined{suffix}.json").write_text(json.dumps(m, indent=2))
        print(f"\n{scene} REFINER{suffix}: PSNR={m['psnr']:.3f} SSIM={m['ssim']:.4f} "
              f"LPIPS={m['lpips']:.4f} Score={m['score']:.4f}")
        return m
    print(f"\n{scene}: refined renders written to {rdir} (private, no GT)")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="hcm0034")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--stage", choices=["all", "pairs", "train", "apply"], default="all")
    ap.add_argument("--base", type=int, default=32, help="U-Net width")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay (0 = off, e.g. 0.999)")
    ap.add_argument("--seed", type=int, default=0,
                    help="weight-init/batch seed for a seed-ensemble member (R2); "
                         "train/val split is fixed regardless")
    ap.add_argument("--tta", action="store_true", help="hflip self-ensemble at apply time")
    ap.add_argument("--load", action="store_true",
                    help="skip training; load refiner{suffix}.pt (or refiner.pt)")
    ap.add_argument("--suffix", default="", help="tag for refiner.pt/renders/metrics outputs")
    ap.add_argument("--ss", type=int, default=1, help="Warper supersample factor")
    ap.add_argument("--sample", choices=["bilinear", "cubic"], default="bilinear")
    ap.add_argument("--config", default=None, help="override backbone checkpoint dir")
    ap.add_argument("--variant", default=None,
                    help="input-cache tag; default auto from ss/sample/config")
    ap.add_argument("--max-pairs", type=int, default=0,
                    help="cap leave-one-out training pairs (0 = all); stride-sampled")
    ap.add_argument("--png", action="store_true",
                    help="write lossless PNG test renders (for the single-encode builder)")
    ap.add_argument("--blocks", choices=["conv", "naf"], default="conv",
                    help="exp040 A/B: plain double-conv blocks or NAFNet blocks "
                         "(arXiv 2204.04676) at matched params")
    ap.add_argument("--evidence", action="store_true",
                    help="exp040 refiner v3: feed per-neighbour aligned warps + "
                         "confidence maps + 3DGS depth (7+4K+1 ch) instead of 7ch")
    ap.add_argument("--rel-tol", type=float, default=None,
                    help="exp036 IBGS relative depth-consistency filter (passed to the Warper)")
    ap.add_argument("--flow-align", choices=["off", "dis", "searaft"], default="off",
                    help="exp039 flow-residual alignment of warped neighbours")
    ap.add_argument("--flow-max-px", type=float, default=7.0)
    ap.add_argument("--searaft-ckpt", default=None)
    args = ap.parse_args()
    flow_align = None if args.flow_align == "off" else {
        "backend": args.flow_align, "max_px": args.flow_max_px,
        "searaft_ckpt": args.searaft_ckpt}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (OUT / args.scene).mkdir(parents=True, exist_ok=True)

    warper_kw = {"ss": args.ss, "sample": args.sample, "config_path": args.config}
    variant = args.variant
    if variant is None:
        variant = ""
        if args.ss > 1:
            variant += f"_ss{args.ss}"
        if args.sample == "cubic":
            variant += "_cub"
        if args.config:
            variant += "_bb"
        # anything that changes what the npz CONTAINS must change the cache tag,
        # or a stale 7ch cache silently gets reused for a v3 run (and vice versa)
        if args.evidence:
            variant += "_ev"
        if args.rel_tol is not None:
            variant += f"_rt{args.rel_tol:g}"
        if flow_align is not None:
            variant += f"_fa{args.flow_align}{args.flow_max_px:g}"

    if args.load:
        ckpt = OUT / args.scene / f"refiner{args.suffix}.pt"
        if not ckpt.exists():
            ckpt = OUT / args.scene / "refiner.pt"
        net = load_refiner(ckpt, device)
        print(f"loaded {ckpt} (ci={net.ci} base={net.base} blocks={net.blocks_kind})")
    else:
        cache, names, _ = build_pairs(args.scene, warper_kw=warper_kw, variant=variant,
                                      max_pairs=args.max_pairs, evidence=args.evidence,
                                      rel_tol=args.rel_tol, flow_align=flow_align)
        if args.stage == "pairs":
            return
        net = train(args.scene, cache, names, device, iters=args.iters, bs=args.bs,
                    base=args.base, ema=args.ema, suffix=args.suffix, seed=args.seed,
                    blocks=args.blocks)
        if args.stage == "train":
            return
    apply_test(args.scene, net, device, warper_kw=warper_kw, variant=variant,
               suffix=args.suffix, tta=args.tta, png=args.png, evidence=args.evidence,
               rel_tol=args.rel_tol, flow_align=flow_align)


if __name__ == "__main__":
    main()
