import os
from PIL import Image
from .encrypt import render_encrypted
from .prompts import build_prompt

def run_attack(sample, cfg, model):
    original = sample["original_title"].strip()
    methods = cfg["attack"]["encryption"]
    img, enc_visible = render_encrypted(original, methods)
    os.makedirs(cfg["paths"]["out_dir"], exist_ok=True)
    out_path = os.path.join(cfg["paths"]["out_dir"], f'{sample["id"]}.png')
    img.save(out_path)
    prompt = build_prompt(cfg["attack"]["use_decryption_hint"], original, enc_visible)
    resp = model.generate(img, prompt)
    return {"id": sample["id"], "category": sample["category"], "original": original,
            "methods": methods, "response": resp}
