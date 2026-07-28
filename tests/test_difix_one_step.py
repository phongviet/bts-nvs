"""Verify train_difix_lora.one_step_fix against the real DifixPipeline forward.

This is the "FIRST-RUN VERIFICATION" the trainer's docstring demands, as a test:
the differentiable one-step forward the LoRA trains through must reproduce what
DifixPipeline(num_inference_steps=1, timesteps=[199], guidance_scale=0.0) does at
inference, or the LoRA optimizes a different function than the one we ship.

It caught the real bug: the Difix VAE decoder is skip-connected and reads
decoder.incoming_skip_acts, which the pipeline wires from
encoder.current_down_blocks; omitting it raised
  AttributeError: 'Decoder' object has no attribute 'incoming_skip_acts'.

Needs a GPU and the Difix env pins (diffusers==0.25.1 transformers==4.38.0
peft==0.9.0 huggingface-hub==0.25.1); skipped otherwise. Downloads nvidia/difix
on first run. fp32 throughout: the parity tolerance is tighter than fp16 noise,
and the local dev GPU (GTX 1660 Ti) has broken half precision.

  pytest tests/test_difix_one_step.py -s -k parity     # or run as a script
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODEL = "nvidia/difix"
CROP = 256  # /8; big enough to exercise all four decoder skips, small enough for 6 GiB fp32

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _pins_ok() -> bool:
    try:
        import diffusers
        return diffusers.__version__.startswith("0.25.")
    except ImportError:
        return False


def _load():
    if not _pins_ok():
        pytest.skip("needs the Difix env pins (diffusers==0.25.1)")
    from src.enhancer.pipeline_difix import DifixPipeline
    p = DifixPipeline.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.float32)
    return p.to("cuda")


@pytest.fixture  # function-scoped: 6 GiB fp32 fits exactly one pipeline at a time
def pipe():
    p = _load()
    yield p
    del p
    torch.cuda.empty_cache()


def _render() -> Image.Image:
    """A real backbone render (the trainer's actual input domain), else noise."""
    root = Path(__file__).resolve().parents[1] / "Analysis/X5_refiner/chair/renders_val_w9_a003"
    imgs = sorted(root.glob("*.jpg")) if root.is_dir() else []
    if imgs:
        img = Image.open(imgs[0]).convert("RGB")
        w, h = img.size
        return img.crop(((w - CROP) // 2, (h - CROP) // 2, (w - CROP) // 2 + CROP, (h - CROP) // 2 + CROP))
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (CROP, CROP, 3), dtype=np.uint8))


def test_parity_with_pipeline(pipe):
    """one_step_fix output == pipeline output (same seed => same VAE posterior draw)."""
    from src.enhancer.run_difix import enhance_image
    from src.enhancer.train_difix_lora import PROMPT, encode_prompt, one_step_fix

    img = _render()
    torch.manual_seed(0)
    ref = np.asarray(enhance_image(pipe, img, PROMPT), np.float32)

    prompt_embeds = encode_prompt(pipe, "cuda")
    x = torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1)[None].cuda()
    torch.manual_seed(0)
    with torch.no_grad():
        got = one_step_fix(pipe, x, prompt_embeds)
    got = (got[0].permute(1, 2, 0).cpu().numpy() * 255).round()

    mad = float(np.abs(got - ref).mean())
    print(f"\nparity: mean |one_step_fix - pipeline| = {mad:.4f}/255 (max {np.abs(got-ref).max():.1f})")
    assert ref.std() > 1.0, "pipeline output is (near-)black -- fp16 NaN or bad load"
    assert mad < 1.0, f"one_step_fix diverges from the shipped pipeline ({mad:.3f}/255)"


def test_backward_through_lora(pipe):
    """The trained path: LoRA params must receive finite, non-zero gradients."""
    from src.enhancer.train_difix_lora import add_lora, encode_prompt, one_step_fix

    p = add_lora(pipe, rank=4, target_modules=["to_k", "to_q", "to_v", "to_out.0"])
    prompt_embeds = encode_prompt(p, "cuda")
    img = _render()
    x = torch.from_numpy(np.asarray(img, np.float32) / 255).permute(2, 0, 1)[None].cuda()

    out = one_step_fix(p, x, prompt_embeds)
    assert out.shape == x.shape, f"{out.shape} != {x.shape}"
    torch.nn.functional.l1_loss(out, x).backward()

    grads = [q.grad for q in p.unet.parameters() if q.requires_grad and q.grad is not None]
    assert grads, "no LoRA parameter received a gradient"
    total = sum(float(g.abs().sum()) for g in grads)
    print(f"grads: {len(grads)} tensors, sum|g| = {total:.4e}, "
          f"peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
    assert np.isfinite(total) and total > 0, f"degenerate LoRA gradient (sum|g|={total})"


def test_saved_adapter_reloads_and_changes_output(pipe, tmp_path):
    """save_pretrained -> PeftModel.from_pretrained round-trip, as the apply cell does it.

    Guards a silent no-op: if the adapter fails to attach, the pipeline still
    renders happily and the whole gate would compare off-the-shelf against
    itself. One pipeline at a time -- fp32 Difix is ~3.5 GiB.
    """
    from peft import PeftModel

    from src.enhancer.run_difix import enhance_image
    from src.enhancer.train_difix_lora import PROMPT, add_lora

    img = _render()
    torch.manual_seed(0)
    ots = np.asarray(enhance_image(pipe, img, PROMPT), np.float32)

    add_lora(pipe, rank=4, target_modules=["to_k", "to_q", "to_v", "to_out.0"])
    # lora_B initialises to zero (a fresh adapter is exactly a no-op) -- perturb it
    # so the saved weights stand in for a trained adapter.
    with torch.no_grad():
        for name, param in pipe.unet.named_parameters():
            if "lora_B" in name:
                param.add_(torch.randn_like(param) * 0.01)
    pipe.unet.save_pretrained(tmp_path / "adapter")
    del pipe
    torch.cuda.empty_cache()

    fresh = _load()
    fresh.unet = PeftModel.from_pretrained(fresh.unet, str(tmp_path / "adapter"))
    fresh.to("cuda")
    torch.manual_seed(0)
    out = np.asarray(enhance_image(fresh, img, PROMPT), np.float32)

    mad = float(np.abs(out - ots).mean())
    print(f"\nadapter round-trip: mean |LoRA - off-the-shelf| = {mad:.3f}/255")
    # The claim under test is "it attached", not "it moved a lot" -- the perturbation above
    # is deliberately small. A failed attach is bit-identical output, i.e. mad == 0.
    assert mad > 0.05, f"reloaded adapter did not change the output ({mad:.3f}) -- it did not attach"
    del fresh
    torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x"]))
