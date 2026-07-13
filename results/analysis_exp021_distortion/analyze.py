"""exp021: COLMAP distortion-residual diagnostic (from exp020 finding (a)).

Question: is the ~2.5x frame-corner error (measured in exp020) driven by
lens-distortion residual, or by a multi-view-overlap / frustum-edge effect?

Method (analysis only, no retrain):
  1. Read each public scene's COLMAP distortion coefficient (SIMPLE_RADIAL k).
  2. Recompute per-scene corner-error severity from exp020's tile_errors.json
     (mean of the 4 corner tiles / mean of all tiles).
  3. Compute the top-vs-bottom corner asymmetry.
  4. Correlate corner severity against k across the 5 scenes.

Verdict rule (pre-registered):
  - DISTORTION mechanism  => corner severity POSITIVELY correlates with k AND the
    effect is roughly top/bottom symmetric (radial distortion is symmetric).
  - OVERLAP/FRUSTUM       => corner severity does NOT track k AND the effect is
    top-corner biased (asymmetric). This is the exp022 trigger.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr, spearmanr
from nerfstudio.data.utils.colmap_parsing_utils import read_cameras_binary

REPO = Path("/home/phong/Viettel_AI_Race_2026/bts-nvs")
TILE_JSON = REPO / "results/analysis_exp020_weakness/tile_errors.json"
OUT = REPO / "results/analysis_exp021_distortion"
OUT.mkdir(parents=True, exist_ok=True)

SCENES = ["hcm0034", "hcm0031", "HCM0181", "HCM0193", "HCM0204"]

# tile grid used in exp020: ny=4 rows, nx=6 cols
NY, NX = 4, 6
CORNERS = [(0, 0), (0, NX - 1), (NY - 1, 0), (NY - 1, NX - 1)]
TOP_CORNERS = [(0, 0), (0, NX - 1)]
BOT_CORNERS = [(NY - 1, 0), (NY - 1, NX - 1)]


def read_distortion(scene):
    cams = read_cameras_binary(str(REPO / f"data/processed/phase1/{scene}/colmap_dense_init/cameras.bin"))
    c = next(iter(cams.values()))
    assert c.model == "SIMPLE_RADIAL", f"{scene}: unexpected model {c.model}"
    f, cx, cy, k = c.params
    # normalized radial distortion at the image corner: predicted pixel displacement
    # magnitude ~ k * r^3 where r is the max normalized radius (corner) in focal units.
    r_corner = np.hypot(max(cx, c.width - cx), max(cy, c.height - cy)) / f
    corner_disp_px = f * k * r_corner ** 3  # radial model x_d = x(1+k r^2)
    return dict(k=float(k), f=float(f), r_corner=float(r_corner),
                corner_disp_px=float(corner_disp_px), width=c.width, height=c.height)


# --- load exp020 tile errors, aggregate per scene ---
records = json.load(open(TILE_JSON))
by_scene = {s: [] for s in SCENES}
for r in records:
    if r["scene"] in by_scene:
        by_scene[r["scene"]].append(np.array(r["tile_errs"]))

rows = []
for s in SCENES:
    tiles = np.stack(by_scene[s])  # (N, NY, NX)
    mean_map = tiles.mean(axis=0)  # (NY, NX)
    overall = mean_map.mean()
    corner_vals = [mean_map[iy, ix] for iy, ix in CORNERS]
    top_vals = [mean_map[iy, ix] for iy, ix in TOP_CORNERS]
    bot_vals = [mean_map[iy, ix] for iy, ix in BOT_CORNERS]
    corner_severity = float(np.mean(corner_vals) / overall)
    top_bot_asym = float(np.mean(top_vals) / np.mean(bot_vals))
    # fraction of images whose single worst tile is a corner, and specifically a top corner
    worst = [tuple(np.unravel_index(t.argmax(), t.shape)) for t in tiles]
    frac_corner = float(np.mean([w in CORNERS for w in worst]))
    frac_top_corner = float(np.mean([w in TOP_CORNERS for w in worst]))
    d = read_distortion(s)
    rows.append(dict(scene=s, n_imgs=len(tiles), corner_severity=corner_severity,
                     top_bot_asym=top_bot_asym, frac_worst_is_corner=frac_corner,
                     frac_worst_is_top_corner=frac_top_corner, **d))

# --- correlations across the 5 scenes ---
k_arr = np.array([r["k"] for r in rows])
disp_arr = np.array([r["corner_disp_px"] for r in rows])
sev_arr = np.array([r["corner_severity"] for r in rows])

def safe_corr(fn, a, b):
    try:
        c, p = fn(a, b)
        return float(c), float(p)
    except Exception:
        return float("nan"), float("nan")

pear_k = safe_corr(pearsonr, k_arr, sev_arr)
spear_k = safe_corr(spearmanr, k_arr, sev_arr)
pear_disp = safe_corr(pearsonr, disp_arr, sev_arr)

summary = dict(
    per_scene=rows,
    corr_severity_vs_k_pearson=dict(r=pear_k[0], p=pear_k[1]),
    corr_severity_vs_k_spearman=dict(rho=spear_k[0], p=spear_k[1]),
    corr_severity_vs_corner_disp_pearson=dict(r=pear_disp[0], p=pear_disp[1]),
    mean_top_bot_asym=float(np.mean([r["top_bot_asym"] for r in rows])),
)
json.dump(summary, open(OUT / "distortion_diagnostic.json", "w"), indent=2)

# --- report ---
print("=" * 78)
print("exp021: COLMAP distortion-residual diagnostic")
print("=" * 78)
hdr = f"{'scene':9} {'k':>9} {'disp_px':>8} {'corner_sev':>11} {'top/bot':>8} {'worst=corner':>13} {'=top':>6}"
print(hdr)
for r in rows:
    print(f"{r['scene']:9} {r['k']:9.5f} {r['corner_disp_px']:8.3f} "
          f"{r['corner_severity']:11.3f} {r['top_bot_asym']:8.3f} "
          f"{r['frac_worst_is_corner']*100:11.0f}% {r['frac_worst_is_top_corner']*100:5.0f}%")
print("-" * 78)
print(f"corner_severity vs k     : Pearson r={pear_k[0]:+.3f} (p={pear_k[1]:.3f}), "
      f"Spearman rho={spear_k[0]:+.3f} (p={spear_k[1]:.3f})")
print(f"corner_severity vs disp  : Pearson r={pear_disp[0]:+.3f} (p={pear_disp[1]:.3f})")
print(f"mean top/bottom corner asymmetry: {summary['mean_top_bot_asym']:.3f} "
      f"(1.0 = symmetric; >1 = top-biased)")
print("-" * 78)

# --- pre-registered verdict ---
asym = summary["mean_top_bot_asym"]
r_k = pear_k[0]
distortion_signal = (r_k > 0.5) and (0.8 < asym < 1.25)
overlap_signal = (abs(r_k) < 0.5) and (asym > 1.25)
print("VERDICT:")
if distortion_signal:
    print("  -> DISTORTION-dominated: severity tracks k and effect is ~symmetric.")
    print("     Fix path: improve undistortion / use full radial model. exp022 (periphery")
    print("     loss weighting) is NOT the right lever -- reconsider before spending compute.")
elif overlap_signal:
    print("  -> OVERLAP/FRUSTUM-dominated: severity does not track k and effect is top-biased.")
    print("     exp022 (per-view periphery loss weighting) is the indicated fix. PROCEED.")
else:
    print("  -> MIXED / inconclusive on 5 points. k-range is narrow (%.5f-%.5f), so the"
          % (k_arr.min(), k_arr.max()))
    print("     correlation test is weak; lean on the asymmetry signal (top/bot=%.2f)." % asym)
    if asym > 1.15:
        print("     Asymmetry favors OVERLAP/FRUSTUM -> exp022 is still the better bet.")
    else:
        print("     Asymmetry is weak too -> neither mechanism cleanly wins; gather more scenes.")
print("=" * 78)
print(f"\nWrote {OUT/'distortion_diagnostic.json'}")
