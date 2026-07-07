"""Week-3 Day-5 per-scene enhancer gate.

Reads results/week3_enhancer_ablation.csv
    (scene, stage in {render_only, offtheshelf, finetuned}, lpips, ssim, psnr, score)
and results/test_pose_coverage_summary.csv, and decides per scene whether the
enhancer ships, writing configs/scene_overrides/<scene>.yaml.

Rules (plan_execution_v3 W3 D5):
  - pick the best enhanced stage per scene (finetuned if present else offtheshelf)
  - apply only if score delta vs render_only >= threshold:
        0.003 normally, 0.001 if the scene is under-covered
        (regime=extrapolative OR frac_uncovered > --uncovered-frac)
  - AND no severe regression: PSNR drop <= 0.3 dB, SSIM drop <= 0.005.

Val-split scores for private scenes / public-GT scores for public scenes go
into the same CSV; the gate doesn't care which, it compares stages within a
scene.
"""
import argparse
import csv
from pathlib import Path

GATE_NORMAL = 0.003
GATE_LOOSE = 0.001


def decide_scene(stages: dict[str, dict], under_covered: bool,
                 gate_normal: float = GATE_NORMAL, gate_loose: float = GATE_LOOSE) -> dict:
    """stages: {stage_name: {lpips, ssim, psnr, score}} for one scene."""
    base = stages.get("render_only")
    if base is None:
        return {"enhancer": "off", "reason": "no render_only baseline row"}
    candidates = [(s, m) for s, m in stages.items() if s != "render_only"]
    if not candidates:
        return {"enhancer": "off", "reason": "no enhanced rows"}
    stage, m = max(candidates, key=lambda kv: kv[1]["score"])
    delta = m["score"] - base["score"]
    thresh = gate_loose if under_covered else gate_normal
    psnr_drop = base["psnr"] - m["psnr"]
    ssim_drop = base["ssim"] - m["ssim"]
    if delta < thresh:
        return {"enhancer": "off",
                "reason": f"best stage {stage} delta {delta:+.4f} < {thresh} "
                          f"({'loose' if under_covered else 'normal'} gate)"}
    if psnr_drop > 0.3 or ssim_drop > 0.005:
        return {"enhancer": "off",
                "reason": f"{stage} wins score ({delta:+.4f}) but regresses "
                          f"psnr {psnr_drop:+.2f} dB / ssim {ssim_drop:+.4f} -- severe-regression rule"}
    return {"enhancer": stage, "reason": f"delta {delta:+.4f} >= {thresh}, no severe regression",
            "score_delta": round(delta, 5)}


def load_ablation(path: Path) -> dict[str, dict[str, dict]]:
    scenes: dict[str, dict[str, dict]] = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            scenes.setdefault(r["scene"], {})[r["stage"]] = {
                "lpips": float(r["lpips"]), "ssim": float(r["ssim"]),
                "psnr": float(r["psnr"]), "score": float(r["score"])}
    return scenes


def load_under_covered(path: Path, uncovered_frac: float) -> set[str]:
    out = set()
    if not path.exists():
        print(f"WARN: {path} missing -- treating all scenes as well-covered (normal gate)")
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["regime"] == "extrapolative" or float(r["frac_uncovered"]) > uncovered_frac:
                out.add(r["scene"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-csv", type=Path, default=Path("results/week3_enhancer_ablation.csv"))
    ap.add_argument("--coverage-csv", type=Path, default=Path("results/test_pose_coverage_summary.csv"))
    ap.add_argument("--overrides-dir", type=Path, default=Path("configs/scene_overrides"))
    ap.add_argument("--uncovered-frac", type=float, default=0.05)
    args = ap.parse_args()

    scenes = load_ablation(args.ablation_csv)
    under = load_under_covered(args.coverage_csv, args.uncovered_frac)
    args.overrides_dir.mkdir(parents=True, exist_ok=True)

    for scene, stages in sorted(scenes.items()):
        decision = decide_scene(stages, scene in under)
        path = args.overrides_dir / f"{scene}.yaml"
        lines = [f"# gate_enhancer.py decision ({args.ablation_csv})"]
        if scene in under:
            lines.append("# scene is under-covered -> loose gate 0.001")
        lines += [f"enhancer: {decision['enhancer']}", f"reason: \"{decision['reason']}\""]
        prev = ""
        if path.exists():
            prev = "".join(l for l in path.read_text().splitlines(keepends=True)
                           if not (l.startswith("#") or l.startswith("enhancer:") or l.startswith("reason:")))
        path.write_text("\n".join(lines) + "\n" + prev)
        print(f"{scene}: enhancer={decision['enhancer']} -- {decision['reason']}")


if __name__ == "__main__":
    main()
