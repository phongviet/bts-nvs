"""exp022 read-out: did top-biased periphery loss weighting actually reduce the
frame-corner error (exp020 finding (a)), and at what cost to the global Score?

Compares the two trained variants' test renders (control = stock splatfacto vs
prw_boost1 = top/corner-weighted L1) against GT, reusing exp020's 6x4 tile grid.
Reports per-variant global metrics + corner severity + top/bottom asymmetry, and
the deltas. Gate (plan): +0.002 Score to promote; the mechanism check is whether
corner_severity / top-corner error drops even if global Score is flat.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch, lpips
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn

# repo root = two levels up from this file (results/analysis_exp022_periphery/), so the
# script runs unchanged on any host (local or Kaggle /kaggle/working/bts-nvs).
REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "runs/phase1/exp022_periphery_weighted/hcm0034"
GT = REPO / "data/raw/phase1/public_set/hcm0034/test/images"
OUT = REPO / "results/analysis_exp022_periphery"
VARIANTS = ["control", "prw_boost1"]
PSNR_MAX, NY, NX = 50.0, 4, 6
CORNERS = [(0, 0), (0, NX-1), (NY-1, 0), (NY-1, NX-1)]
TOP, BOT = [(0, 0), (0, NX-1)], [(NY-1, 0), (NY-1, NX-1)]

dev = "cuda" if torch.cuda.is_available() else "cpu"
lp_fn = lpips.LPIPS(net="vgg").to(dev).eval()
rgb = lambda p: np.asarray(Image.open(p).convert("RGB"))
def t(a): return (torch.from_numpy(a).float().permute(2,0,1)[None]/127.5-1).to(dev)

results = {}
for var in VARIANTS:
    rdir = RUN / var / "renders_test"
    if not rdir.exists():
        print(f"!! {var}: {rdir} missing (training/render not done)"); continue
    files = sorted(p.name for p in rdir.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png"))
    ps, ss, ls, tile_maps, worst = [], [], [], [], []
    for fn in files:
        g, r = rgb(GT/fn), rgb(rdir/fn)
        if r.shape != g.shape:
            r = np.asarray(Image.fromarray(r).resize((g.shape[1], g.shape[0]), Image.LANCZOS))
        ps.append(psnr_fn(g, r, data_range=255))
        ss.append(ssim_fn(g, r, channel_axis=2, data_range=255))
        with torch.no_grad(): ls.append(lp_fn(t(g), t(r)).item())
        h, w = g.shape[:2]; diff = (g.astype(np.float32)-r.astype(np.float32))**2
        tm = np.array([[diff[iy*h//NY:(iy+1)*h//NY, ix*w//NX:(ix+1)*w//NX].mean()
                        for ix in range(NX)] for iy in range(NY)])
        tile_maps.append(tm); worst.append(tuple(np.unravel_index(tm.argmax(), tm.shape)))
    psnr, ssim, lpv = np.mean(ps), np.mean(ss), np.mean(ls)
    score = 0.4*(1-lpv) + 0.3*ssim + 0.3*min(max(psnr/PSNR_MAX,0),1)
    mmap = np.mean(tile_maps, axis=0)
    sev = float(np.mean([mmap[c] for c in CORNERS]) / mmap.mean())
    asym = float(np.mean([mmap[c] for c in TOP]) / np.mean([mmap[c] for c in BOT]))
    ftop = float(np.mean([w in TOP for w in worst]))
    results[var] = dict(n=len(files), psnr=float(psnr), ssim=float(ssim), lpips=float(lpv),
                        score=float(score), corner_severity=sev, top_bot_asym=asym,
                        frac_worst_top_corner=ftop)

print("="*70); print("exp022: periphery-weighted L1 read-out (hcm0034)"); print("="*70)
hdr = f"{'variant':11} {'PSNR':>6} {'SSIM':>6} {'LPIPS':>6} {'Score':>7} {'cornSev':>8} {'top/bot':>8} {'worst=top':>9}"
print(hdr)
for v in VARIANTS:
    if v not in results: continue
    r = results[v]
    print(f"{v:11} {r['psnr']:6.3f} {r['ssim']:6.4f} {r['lpips']:6.4f} {r['score']:7.4f} "
          f"{r['corner_severity']:8.3f} {r['top_bot_asym']:8.3f} {r['frac_worst_top_corner']*100:8.0f}%")
if all(v in results for v in VARIANTS):
    c, p = results["control"], results["prw_boost1"]
    dS = p["score"]-c["score"]; dSev = p["corner_severity"]-c["corner_severity"]
    print("-"*70)
    print(f"Delta Score        : {dS:+.4f}   (gate +0.0020 to promote)")
    print(f"Delta corner_sev   : {dSev:+.3f}   (negative = corners improved, mechanism worked)")
    print(f"Delta PSNR/SSIM/LPIPS: {p['psnr']-c['psnr']:+.3f} / {p['ssim']-c['ssim']:+.4f} / {p['lpips']-c['lpips']:+.4f}")
    print("-"*70)
    verdict = ("PROMOTE" if dS >= 0.002 else
               "MECHANISM-WORKS-BUT-BELOW-GATE" if dSev < -0.05 and dS > -0.001 else
               "DROP")
    print(f"VERDICT: {verdict}")
json.dump(results, open(OUT/"corner_comparison.json","w"), indent=2)
print(f"\nWrote {OUT/'corner_comparison.json'}")
