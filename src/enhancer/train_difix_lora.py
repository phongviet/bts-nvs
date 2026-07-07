"""exp016: LoRA-finetune Difix on our own render/real pairs (Week 3 Day 3-4).

Data: data/processed/phase1/enhancer_pairs/manifest.csv (built by
build_enhancer_pairs.py -- pairs come only from organizer-provided imagery).
Held out BY SCENE: --holdout-scenes are never trained on; their LPIPS curve
is the early-stopping signal (plan: train on 10 scenes, validate on 3).

Loss: perceptual-heavy by design -- lpips_w * LPIPS_vgg + l1_w * L1, with
transient masks (where present) zeroing masked pixels on both sides.
Random 512px crops, batch via grad accumulation.

FIRST-RUN VERIFICATION (rented GPU, before any real sweep -- record in
experiment_log.md):
  - print(pipe) and confirm components: unet, vae, scheduler, text_encoder.
    This script reimplements the one-step img2img forward (encode -> unet at
    t=199 -> scheduler x0 -> decode) so gradients flow; diff ONE image's
    output against pipe(...) inference -- if they disagree beyond fp tolerance
    the hub pipeline does something custom (e.g. skip-connected decoder) and
    the forward below must be adapted to match (read the hub repo's
    pipeline source; it's downloaded next to the weights).
  - confirm LoRA target module names exist (script errors out otherwise).

Usage:
  python src/enhancer/train_difix_lora.py \
      --pairs-root data/processed/phase1/enhancer_pairs \
      --holdout-scenes HCM0204 HNI0366 HCM0254 \
      --out runs/phase1/exp016_difix_lora \
      [--lpips-w 1.0 --l1-w 0.2] [--steps 4000] [--lr 1e-4] [--rank 16]
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PROMPT = "remove degradation"
TIMESTEP = 199


# ---------------- data ----------------

class PairsDataset(torch.utils.data.Dataset):
    def __init__(self, root: Path, scenes: list[str], crop: int = 512):
        self.root = root
        self.crop = crop
        with open(root / "manifest.csv") as f:
            self.rows = [r for r in csv.DictReader(f) if r["scene"] in scenes]
        if not self.rows:
            raise SystemExit(f"no pairs for scenes {scenes} in {root}/manifest.csv")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        render = np.asarray(Image.open(self.root / r["render"]).convert("RGB"), np.float32) / 255
        real = np.asarray(Image.open(self.root / r["real"]).convert("RGB"), np.float32) / 255
        if r["mask"]:
            mask = np.asarray(Image.open(self.root / r["mask"]).convert("L"), np.float32)[..., None] / 255
        else:
            mask = np.ones_like(render[..., :1])
        h, w = render.shape[:2]
        c = min(self.crop, h, w)
        y = random.randint(0, h - c)
        x = random.randint(0, w - c)
        sl = (slice(y, y + c), slice(x, x + c))
        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a[sl])).permute(2, 0, 1)
        return to_t(render), to_t(real), to_t(mask)


# ---------------- model ----------------

def load_components(model_id: str, device: str):
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(model_id, trust_remote_code=True,
                                             torch_dtype=torch.float32)
    pipe.to(device)
    for name in ("unet", "vae", "scheduler"):
        assert hasattr(pipe, name), f"pipeline has no .{name} -- adapt this script to the hub code"
    return pipe


def add_lora(pipe, rank: int, target_modules: list[str]):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=rank, lora_alpha=rank, target_modules=target_modules,
                     lora_dropout=0.0, bias="none")
    pipe.unet = get_peft_model(pipe.unet, cfg)
    pipe.unet.print_trainable_parameters()
    return pipe


@torch.no_grad()
def encode_prompt(pipe, device):
    tok = pipe.tokenizer(PROMPT, padding="max_length",
                         max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt")
    return pipe.text_encoder(tok.input_ids.to(device))[0]


def one_step_fix(pipe, img: torch.Tensor, prompt_embeds: torch.Tensor) -> torch.Tensor:
    """Differentiable single-step img2img: img in [0,1] (B,3,H,W) -> fixed [0,1].

    Mirrors the SD-Turbo/pix2pix-turbo structure Difix is built on. VERIFY
    against the hub pipeline's own output on first run (see module docstring).
    """
    vae, unet, sched = pipe.vae, pipe.unet, pipe.scheduler
    x = img * 2 - 1
    latents = vae.encode(x).latent_dist.sample() * vae.config.scaling_factor
    t = torch.full((latents.shape[0],), TIMESTEP, device=latents.device, dtype=torch.long)
    noise_pred = unet(latents, t, encoder_hidden_states=prompt_embeds.expand(latents.shape[0], -1, -1)).sample
    # x0 prediction under the scheduler's parameterization
    alphas = sched.alphas_cumprod.to(latents.device)[t].view(-1, 1, 1, 1)
    pred_type = getattr(sched.config, "prediction_type", "epsilon")
    if pred_type == "epsilon":
        x0 = (latents - (1 - alphas).sqrt() * noise_pred) / alphas.sqrt()
    elif pred_type == "v_prediction":
        x0 = alphas.sqrt() * latents - (1 - alphas).sqrt() * noise_pred
    else:
        raise RuntimeError(f"unhandled prediction_type {pred_type} -- adapt one_step_fix()")
    out = vae.decode(x0 / vae.config.scaling_factor).sample
    return ((out + 1) / 2).clamp(0, 1)


# ---------------- train ----------------

def evaluate(pipe, ds, prompt_embeds, lpips_fn, device, n: int = 24) -> float:
    vals = []
    with torch.no_grad():
        for i in np.linspace(0, len(ds) - 1, min(n, len(ds))).round().astype(int):
            render, real, _ = ds[int(i)]
            out = one_step_fix(pipe, render.unsqueeze(0).to(device), prompt_embeds)
            vals.append(float(lpips_fn(out * 2 - 1, (real.unsqueeze(0).to(device)) * 2 - 1).item()))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-root", required=True, type=Path)
    ap.add_argument("--holdout-scenes", required=True, nargs="+")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default="nvidia/difix")
    ap.add_argument("--lpips-w", type=float, default=1.0)
    ap.add_argument("--l1-w", type=float, default=0.2, help="L1 down-weighted per the plan")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=4,
                    help="stop after N evals without held-out improvement (divergence guard)")
    ap.add_argument("--target-modules", nargs="+",
                    default=["to_k", "to_q", "to_v", "to_out.0"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.pairs_root / "manifest.csv") as f:
        all_scenes = sorted({r["scene"] for r in csv.DictReader(f)})
    train_scenes = [s for s in all_scenes if s not in set(args.holdout_scenes)]
    print(f"train scenes ({len(train_scenes)}): {train_scenes}")
    print(f"holdout scenes ({len(args.holdout_scenes)}): {args.holdout_scenes}")

    train_ds = PairsDataset(args.pairs_root, train_scenes, args.crop)
    val_ds = PairsDataset(args.pairs_root, args.holdout_scenes, args.crop)

    pipe = load_components(args.model, device)
    pipe = add_lora(pipe, args.rank, args.target_modules)
    prompt_embeds = encode_prompt(pipe, device)

    import lpips as lpips_mod
    lpips_fn = lpips_mod.LPIPS(net="vgg").to(device)
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    params = [p for p in pipe.unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=True,
                                         num_workers=2, drop_last=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "train_config.json").write_text(json.dumps(vars(args), default=str, indent=2))

    base_val = evaluate(pipe, val_ds, prompt_embeds, lpips_fn, device)
    print(f"step 0: held-out LPIPS {base_val:.4f} (pre-finetune baseline)")
    best_val, best_step, evals_since_best = base_val, 0, 0
    history = [{"step": 0, "val_lpips": base_val}]

    step = 0
    it = iter(loader)
    while step < args.steps:
        opt.zero_grad()
        for _ in range(args.grad_accum):
            try:
                render, real, mask = next(it)
            except StopIteration:
                it = iter(loader)
                render, real, mask = next(it)
            render, real, mask = (t.to(device) for t in (render, real, mask))
            out = one_step_fix(pipe, render, prompt_embeds)
            out_m, real_m = out * mask, real * mask
            loss = (args.l1_w * F.l1_loss(out_m, real_m)
                    + args.lpips_w * lpips_fn(out_m * 2 - 1, real_m * 2 - 1).mean())
            (loss / args.grad_accum).backward()
        opt.step()
        step += 1

        if step % args.eval_every == 0:
            val = evaluate(pipe, val_ds, prompt_embeds, lpips_fn, device)
            history.append({"step": step, "val_lpips": val})
            marker = ""
            if val < best_val:
                best_val, best_step, evals_since_best = val, step, 0
                pipe.unet.save_pretrained(args.out / "best")
                marker = "  <- best, saved"
            else:
                evals_since_best += 1
            print(f"step {step}: held-out LPIPS {val:.4f} (best {best_val:.4f} @ {best_step}){marker}")
            (args.out / "history.json").write_text(json.dumps(history, indent=1))
            if evals_since_best >= args.patience:
                print(f"early stop: no held-out improvement for {args.patience} evals (divergence guard)")
                break

    print(f"done. best held-out LPIPS {best_val:.4f} @ step {best_step} "
          f"(pre-finetune {base_val:.4f}) -> {args.out}/best")
    if best_step == 0:
        print("LoRA never beat the off-the-shelf model on held-out scenes -- per the plan, "
              "ship off-the-shelf (or nothing) and log the kill.")


if __name__ == "__main__":
    main()
