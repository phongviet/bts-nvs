"""Analysis 25 (§3.4): is the encode knapsack's objective the wrong shape?

`16_encode_knapsack.recoverable_score` allocates bytes by
    0.3*SSIM + 0.3*min(PSNR,50)/50
i.e. the grader MINUS its largest term. The grader is
    0.4*(1-LPIPS) + 0.3*SSIM + 0.3*PSNR/50
so 40 % of the objective -- the perceptual term -- is invisible to the
allocator. The docstring's citation (arXiv 2510.10970, "LPIPS-tuned tables do
not help") is about the QUANTIZATION TABLES, not about the allocation
objective, so this is a genuinely unmeasured lever.

This script measures it end to end, locally, with no submission slot:

 1. Encode every graded frame at every rung once (shared by both arms).
 2. Score each rung TWICE: arm A = the shipped SSIM+PSNR proxy, arm B = the
    full grader shape including LPIPS. Same bytes, same rungs, same budget,
    same scene weights -- the objective is the only variable.
 3. Run the same greedy allocation for each and diff the allocations.
 4. Judge both with the REAL grader (full-res, vgg LPIPS, PSNR_MAX 50) against
    the lossless renders, averaged PER SCENE like the leaderboard.

Step 4 is the ruler. JPEG can only degrade the render, so the quantity that
matters is encode-induced Score loss vs the lossless source; that needs no GT
and is therefore exactly measurable on the private set's own renders.

Usage:
  conda run -n airace python Analysis/25_grader_shaped_knapsack.py \
      --suffix _v7a --budget-mib 348 --qfloor 98 --qmax 100
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load_by_path(mod: str, rel: str):
    spec = importlib.util.spec_from_file_location(mod, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod] = m
    spec.loader.exec_module(m)
    return m


knap16 = _load_by_path("knap16", "Analysis/16_encode_knapsack.py")

SCENES = ["HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674", "bonsai", "chair"]


# --------------------------- the two objectives --------------------------------
def score_ssim_psnr(dec_small, ref_small):
    """Arm A: exactly what 16_encode_knapsack.recoverable_score computes."""
    return (0.3 * knap16.ssim_np(dec_small, ref_small)
            + 0.3 * min(knap16.psnr_np(dec_small, ref_small), 50.0) / 50.0)


class GraderShaped:
    """Arm B: arm A plus the grader's 0.4*(1-LPIPS) term.

    LPIPS runs on the same downsampled copy the other terms use -- the proxy
    only has to RANK rungs, and a full-res perceptual eval per rung per frame
    would cost more than the whole build. alex (not vgg) on purpose here: it is
    ~4x cheaper and the rung ORDER is what matters. The final judgement in
    stage 4 uses the real vgg grader at full res.
    """

    def __init__(self, device):
        import lpips
        self.net = lpips.LPIPS(net="alex").to(device).eval()
        self.device = device

    def __call__(self, dec_small, ref_small):
        base = score_ssim_psnr(dec_small, ref_small)
        a = torch.from_numpy(dec_small).permute(2, 0, 1)[None].to(self.device) * 2 - 1
        b = torch.from_numpy(ref_small).permute(2, 0, 1)[None].to(self.device) * 2 - 1
        with torch.no_grad():
            lp = float(self.net(a, b).item())
        return base + 0.4 * (1.0 - lp), lp


# ------------------------------ shared curves ----------------------------------
def build_curves(pooled, qualities, cache: Path, device, maxside=640):
    """Encode once, score twice. Returns {key: [(q, size, scoreA, scoreB)]} and
    writes the JPEG bytes to `cache` so the two arms ship identical files."""
    grader = GraderShaped(device)
    cache.mkdir(parents=True, exist_ok=True)
    curves, t0 = {}, time.time()
    for i, (key, path) in enumerate(pooled, 1):
        im = Image.open(path).convert("RGB")
        ref_small = knap16._downsample(im, maxside)
        pts = []
        for q in qualities:
            data = knap16.encode(im, q, subsampling=0, tune="ms-ssim")
            fp = cache / f"{key.replace('/', '__')}.q{q}.jpg"
            fp.write_bytes(data)
            dec_small = knap16._downsample(knap16._decode(data), maxside)
            sa = score_ssim_psnr(dec_small, ref_small)
            sb, lp = grader(dec_small, ref_small)
            pts.append((q, len(data), sa, sb, lp))
        curves[key] = pts
        if i % 25 == 0 or i == len(pooled):
            el = time.time() - t0
            print(f"    curves {i}/{len(pooled)}  ({el:.0f}s, "
                  f"{el / i * (len(pooled) - i):.0f}s left)", flush=True)
        im.close()
    return curves


def greedy(curves, budget, weights, arm):
    """The same greedy as knap16.allocate, but reading a precomputed curve and
    selecting the score column by arm ('A' -> index 2, 'B' -> index 3)."""
    col = 2 if arm == "A" else 3
    # Pareto-prune on THIS arm's score: a rung only survives if more bytes buy
    # more of the objective being maximised.
    pruned = {}
    for k, pts in curves.items():
        seq, best = [], -1e9
        for p in sorted(pts, key=lambda t: t[1]):
            if p[col] > best:
                seq.append(p)
                best = p[col]
        pruned[k] = seq
    chosen = {k: 0 for k in pruned}
    total = sum(pruned[k][0][1] for k in pruned)
    if total > budget:
        return chosen, pruned, total, False
    while True:
        best = None
        for k, ci in chosen.items():
            rungs = pruned[k]
            if ci + 1 >= len(rungs):
                continue
            dsz = rungs[ci + 1][1] - rungs[ci][1]
            dsc = rungs[ci + 1][col] - rungs[ci][col]
            if dsz <= 0 or total + dsz > budget:
                continue
            ratio = dsc * (weights.get(k, 1.0) if weights else 1.0) / dsz
            if best is None or ratio > best[0]:
                best = (ratio, k, dsz)
        if best is None:
            break
        _, k, dsz = best
        chosen[k] += 1
        total += dsz
    return chosen, pruned, total, True


def materialise(chosen, pruned, cache: Path, out: Path):
    """Write the selected rung's bytes into out/<scene>/<name>."""
    if out.exists():
        shutil.rmtree(out)
    for key, ci in chosen.items():
        q = pruned[key][ci][0]
        scene, name = key.split("/", 1)
        (out / scene).mkdir(parents=True, exist_ok=True)
        src = cache / f"{key.replace('/', '__')}.q{q}.jpg"
        shutil.copy2(src, out / scene / name)


def real_grader(enc_root: Path, ref: dict, device):
    """Full-res vgg LPIPS + skimage SSIM/PSNR of decoded JPEG vs the LOSSLESS
    render, averaged per scene then over scenes -- the leaderboard's shape."""
    import lpips
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim
    net = lpips.LPIPS(net="vgg").to(device).eval()
    per_scene = {}
    for scene, items in ref.items():
        rows = []
        for name, src in items:
            a = np.asarray(Image.open(enc_root / scene / name).convert("RGB"),
                           np.float32) / 255.0
            b = np.asarray(Image.open(src).convert("RGB"), np.float32) / 255.0
            psnr = float(sk_psnr(b, a, data_range=1.0))
            ssim = float(sk_ssim(b, a, data_range=1.0, channel_axis=2, win_size=11))
            ta = torch.from_numpy(a).permute(2, 0, 1)[None].to(device) * 2 - 1
            tb = torch.from_numpy(b).permute(2, 0, 1)[None].to(device) * 2 - 1
            with torch.no_grad():
                lp = float(net(ta, tb).item())
            rows.append((psnr, ssim, lp,
                         0.4 * (1 - lp) + 0.3 * ssim + 0.3 * min(psnr / 50.0, 1.0)))
        arr = np.array(rows)
        per_scene[scene] = dict(psnr=arr[:, 0].mean(), ssim=arr[:, 1].mean(),
                                lpips=arr[:, 2].mean(), score=arr[:, 3].mean(),
                                n=len(rows))
    del net
    torch.cuda.empty_cache()
    return per_scene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_v7a", help="render set to encode")
    ap.add_argument("--budget-mib", type=int, default=348)
    ap.add_argument("--qfloor", type=int, default=98)
    ap.add_argument("--qmax", type=int, default=100)
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: first N frames per scene (budget scales with it)")
    ap.add_argument("--work", default=None, help="scratch dir for encoded rungs")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    work = Path(args.work) if args.work else Path("/tmp/knap34")
    print(f"device={device} backend={'mozjpeg' if knap16.MOZJPEG else 'pillow'} "
          f"work={work}")

    # --- gather the lossless render set, exactly as the builder sees it -------
    ref, pooled = {}, []
    for s in args.scenes:
        d = REPO / f"Analysis/X5_refiner/{s}/renders_refined{args.suffix}"
        assert d.exists(), f"missing {d}"
        files = sorted(d.glob("*.png"))
        assert files, f"no PNGs in {d}"
        if args.limit:
            files = files[:args.limit]
        items = [((f.stem + ".JPG") if s.startswith("HCM") else (f.stem + ".jpg"), f)
                 for f in files]
        ref[s] = items
        pooled += [(f"{s}/{n}", p) for n, p in items]
    n = len(pooled)
    print(f"  {n} frames over {len(args.scenes)} scenes "
          f"({', '.join(f'{s}:{len(ref[s])}' for s in args.scenes)})")

    budget = args.budget_mib * 2 ** 20
    if args.limit:                       # keep the budget proportional in debug
        full = sum(len(list((REPO / f'Analysis/X5_refiner/{s}/renders_refined'
                             f'{args.suffix}').glob('*.png'))) for s in args.scenes)
        budget = int(budget * n / full)
        print(f"  DEBUG budget scaled to {budget/2**20:.1f} MiB ({n}/{full} frames)")

    # scene weights, identical to 24_build_round2_submission
    raw = {f"{s}/{nm}": 1.0 / (len(args.scenes) * len(ref[s]))
           for s in args.scenes for nm, _ in ref[s]}
    mw = sum(raw.values()) / len(raw)
    weights = {k: v / mw for k, v in raw.items()}

    qualities = list(range(args.qfloor, args.qmax + 1))
    print(f"  rungs q{qualities[0]}..q{qualities[-1]}, budget {budget/2**20:.1f} MiB")
    curves = build_curves(pooled, qualities, work / "rungs", device)

    # How much signal does arm B actually add? The objective difference between
    # rungs is what the greedy ranks on, so compare the ACROSS-RUNG spread of the
    # LPIPS term (0.4*dLPIPS) against the spread of the shipped SSIM+PSNR term.
    lp_span = np.array([max(p[4] for p in pts) - min(p[4] for p in pts)
                        for pts in curves.values()])
    a_span = np.array([max(p[2] for p in pts) - min(p[2] for p in pts)
                       for pts in curves.values()])
    b_term = 0.4 * lp_span
    print(f"\n  across-rung spread per frame (this is all the greedy can rank on):")
    print(f"    arm A term (0.3*SSIM+0.3*PSNR/50): mean {a_span.mean():.6f}  "
          f"max {a_span.max():.6f}")
    print(f"    LPIPS term (0.4*dLPIPS)          : mean {b_term.mean():.6f}  "
          f"max {b_term.max():.6f}")
    print(f"    raw dLPIPS q{qualities[0]}..q{qualities[-1]}      : mean "
          f"{lp_span.mean():.6f}  max {lp_span.max():.6f}")
    print(f"    => the added term is {b_term.mean()/max(a_span.mean(),1e-12):.1%} "
          f"the size of the existing one")

    res = {}
    for arm in ("A", "B"):
        chosen, pruned, total, fit = greedy(curves, budget, weights, arm)
        qs = [pruned[k][ci][0] for k, ci in chosen.items()]
        hist = {q: qs.count(q) for q in sorted(set(qs))}
        print(f"\n  arm {arm} ({'SSIM+PSNR proxy' if arm == 'A' else 'grader-shaped'}): "
              f"{total/2**20:.1f} MiB, fit={fit}, histogram {hist}")
        out = work / f"enc_{arm}"
        materialise(chosen, pruned, work / "rungs", out)
        res[arm] = dict(chosen=chosen, pruned=pruned, total=total, hist=hist, out=out)

    diff = [k for k in res["A"]["chosen"]
            if res["A"]["pruned"][k][res["A"]["chosen"][k]][0]
            != res["B"]["pruned"][k][res["B"]["chosen"][k]][0]]
    print(f"\n  frames allocated DIFFERENTLY: {len(diff)}/{n} "
          f"({100*len(diff)/n:.1f}%)")
    if not diff:
        # Identical allocation => identical bytes => the graded delta is exactly
        # 0.000000 by construction. Running the 30-min full-res grader to print a
        # zero would be theatre, not evidence.
        print("  => the objective change is a NO-OP at this budget/rung set: both "
              "arms select the SAME rung for every frame, so the encoded bytes are "
              "byte-identical and the graded delta is exactly 0 by construction.")
        print("     Skipping stage 4 (it can only print zeros). §3.4 is REFUTED "
              "at this rung set; see the spread numbers above for why.")
        (work / "result.json").write_text(json.dumps(
            {"verdict": "no-op", "frames": n, "diff_frames": 0,
             "hist": {str(k): v for k, v in res["A"]["hist"].items()},
             "mib": res["A"]["total"] / 2 ** 20,
             "a_span_mean": float(a_span.mean()), "a_span_max": float(a_span.max()),
             "lpips_term_mean": float(b_term.mean()),
             "lpips_term_max": float(b_term.max()),
             "dlpips_mean": float(lp_span.mean()),
             "dlpips_max": float(lp_span.max())}, indent=2))
        print(f"wrote {work/'result.json'}")
        return

    print("\n--- stage 4: real grader (full-res vgg) vs the lossless renders ---")
    for arm in ("A", "B"):
        res[arm]["graded"] = real_grader(res[arm]["out"], ref, device)

    print(f"\n{'scene':10s} {'A score':>9s} {'B score':>9s} {'delta':>9s} "
          f"{'A lpips':>9s} {'B lpips':>9s}")
    for s in args.scenes:
        a, b = res["A"]["graded"][s], res["B"]["graded"][s]
        print(f"{s:10s} {a['score']:9.6f} {b['score']:9.6f} "
              f"{b['score']-a['score']:+9.6f} {a['lpips']:9.5f} {b['lpips']:9.5f}")
    ma = np.mean([res["A"]["graded"][s]["score"] for s in args.scenes])
    mb = np.mean([res["B"]["graded"][s]["score"] for s in args.scenes])
    print(f"{'SCENE MEAN':10s} {ma:9.6f} {mb:9.6f} {mb-ma:+9.6f}")
    print(f"\nLB-equivalent delta: {100*(mb-ma):+.5f} points")
    print("Ship arm B only if this is positive AND above the project's "
          "+-0.002 Score noise bar (= +-0.2 LB points).")

    (work / "result.json").write_text(json.dumps(
        {arm: {"hist": {str(k): v for k, v in res[arm]["hist"].items()},
               "mib": res[arm]["total"] / 2 ** 20,
               "graded": {s: {k: float(v) for k, v in res[arm]["graded"][s].items()}
                          for s in args.scenes}} for arm in ("A", "B")}, indent=2))
    print(f"wrote {work/'result.json'}")


if __name__ == "__main__":
    main()
