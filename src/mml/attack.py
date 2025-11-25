import os
from PIL import Image
from .encrypt import render_encrypted
from .prompts import build_prompt


def run_attack(sample, cfg, model):
    # Support either 'text' or 'original_title'
    original = (sample.get("text") or sample.get("original_title") or "").strip()
    if not original:
        raise ValueError(f"No text field found in sample: {sample}")

    methods = cfg["attack"]["encryption"]

    # Generate encrypted image
    img, enc_visible = render_encrypted(original, methods)

    # Save image for inspection
    out_dir = cfg["paths"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    sample_id = sample.get("id") or sample.get("image") or "sample"
    out_path = os.path.join(out_dir, f"{sample_id}.png")
    img.save(out_path)

    # Build model prompt
    prompt = build_prompt(
        cfg["attack"]["use_decryption_hint"],
        original,
        enc_visible,
    )

    # Query model
    resp = model.generate(img, prompt)

    # Compute whether **this sample** is a success
    success_flag = is_success(
        original_text=original,
        model_output=resp,
        refusal_keywords=cfg["eval"]["refusal_keywords"]
    )

    return {
        "id": sample_id,
        "original": original,
        "methods": methods,
        "response": resp,
        "success": int(success_flag),
    }
