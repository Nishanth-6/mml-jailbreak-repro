import base64
import random
from typing import List, Tuple

import nltk
from PIL import Image, ImageDraw, ImageFont, ImageOps
import qrcode


# Very small, safe replacement vocab
POSITIVE_ADJ = ["happy", "bright", "sweet", "clean", "fresh"]
FOOD_NOUNS = ["pizza", "burger", "pasta", "salad", "sushi"]


def word_replace(text: str) -> str:
    """
    Simple word replacement attack:
    - adjectives -> positive adjectives
    - nouns -> food nouns
    This hides the real semantics while keeping syntax natural.
    """
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)

    out = []
    for tok, pos in tags:
        if pos.startswith("JJ"):  # adjective
            out.append(random.choice(POSITIVE_ADJ))
        elif pos.startswith("NN"):  # noun
            out.append(random.choice(FOOD_NOUNS))
        else:
            out.append(tok)
    return " ".join(out)


def base64_encode_text(text: str) -> str:
    """
    Encode the string as base64.
    """
    return base64.b64encode(text.encode("utf-8")).decode("utf-8")


def build_figstep_text(text: str) -> str:
    """
    FigStep style progressive reveal.
    We create 4 steps that gradually reveal more of the title.
    """
    n = len(text)
    steps = []
    for idx, frac in enumerate([0.25, 0.5, 0.75, 1.0], start=1):
        k = max(1, int(n * frac))
        steps.append(f"Step {idx}: {text[:k]}")
    return "\n".join(steps)


def text_to_typography_img(text: str, width: int = 900, height: int = 300) -> Image.Image:
    """
    Render multi line text onto a white canvas.
    Keeps things very light so it runs on CPU or GPU without issues.
    """
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Use default font to avoid font-path issues
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Simple manual line breaking
    lines = text.split("\n")
    y = 40
    for line in lines:
        draw.text((30, y), line, fill=(0, 0, 0), font=font)
        y += 40
    return img


def text_to_qr_image(text: str, size: int = 512) -> Image.Image:
    """
    QR style attack: encode the (possibly transformed) text as a QR code.
    """
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )
    qr.add_data(text)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    qr_img = qr_img.resize((size, size))
    return qr_img


def mirror(img: Image.Image) -> Image.Image:
    return ImageOps.mirror(img)


def rotate(img: Image.Image) -> Image.Image:
    # 180 degrees rotation, expand=False keeps size
    return img.rotate(180, expand=False)


def render_encrypted(original_title: str, methods: List[str]) -> Tuple[Image.Image, str]:
    """
    Core entry point used by attack.py.

    methods is a list from config.yaml, for example:
      ["word_replace"]
      ["qr"]
      ["figstep"]
      ["base64", "mirror"]
      ["word_replace", "rotate"]

    We apply:
      1) text-level transforms (word_replace, base64, figstep)
      2) choose how to render (qr vs plain typography)
      3) image-level transforms (mirror, rotate)

    Returns:
      (PIL.Image, visible_transformed_text_for_hint)
    """
    text = original_title

    # 1) text-level transforms
    if "word_replace" in methods:
        text = word_replace(text)

    if "base64" in methods:
        text = base64_encode_text(text)

    if "figstep" in methods:
        text = build_figstep_text(text)

    # 2) rendering mode
    if "qr" in methods:
        img = text_to_qr_image(text)
    else:
        img = text_to_typography_img(text)

    # 3) image-level transforms
    if "mirror" in methods:
        img = mirror(img)

    if "rotate" in methods:
        img = rotate(img)

    # visible text that we can optionally show in the prompt hint
    return img, text
