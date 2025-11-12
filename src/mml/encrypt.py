from PIL import Image, ImageOps, ImageDraw, ImageFont
import base64, io, random
import nltk

POSITIVE_ADJ = ["happy","bright","sweet","clean","fresh"]
FOOD_NOUNS   = ["pizza","burger","pasta","salad","sushi"]

def word_replace(text:str):
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    out = []
    for tok, pos in tags:
        if pos.startswith("JJ"):
            out.append(random.choice(POSITIVE_ADJ))
        elif pos.startswith("NN"):
            out.append(random.choice(FOOD_NOUNS))
        else:
            out.append(tok)
    return " ".join(out)

def text_to_typography_img(text:str, width=900, height=300):
    img = Image.new("RGB", (width, height), color=(255,255,255))
    draw = ImageDraw.Draw(img)
    # default font fallback
    draw.text((30, 130), text, fill=(0,0,0))
    return img

def mirror(img:Image.Image): return ImageOps.mirror(img)
def rotate(img:Image.Image): return img.rotate(180, expand=True)

def base64_encode_text(text:str):
    enc = base64.b64encode(text.encode()).decode()
    return enc

def render_encrypted(title:str, methods:list):
    """methods ⊆ {'word_replace','mirror','rotate','base64'}"""
    t = title
    if "word_replace" in methods:
        t = word_replace(t)
    if "base64" in methods:
        t = base64_encode_text(t)
    img = text_to_typography_img(t)
    if "mirror" in methods:
        img = mirror(img)
    if "rotate" in methods:
        img = rotate(img)
    return img, t  # return encrypted-visible text too (for hint)
