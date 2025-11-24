import os, json, subprocess, tempfile
from typing import Optional
from PIL import Image

class BaseModel:
    def generate(self, image:Image.Image, prompt:str)->str:
        raise NotImplementedError

class OllamaVL(BaseModel):
    def __init__(self, model:str="llava:7b"):
        self.model = model
    def generate(self, image:Image.Image, prompt:str)->str:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f.name)
            cmd = ["ollama","run",self.model, json.dumps({"prompt":prompt,"images":[f.name]})]
            out = subprocess.check_output(cmd, text=True)
            return out.strip()

class MockModel(BaseModel):
    def generate(self, image, prompt):
        # cheap, deterministic stub so you can test everything else
        return "MOCK_RESPONSE"

def build_model(engine:str, **kwargs)->BaseModel:
    if engine=="ollama":
        return OllamaVL(model=kwargs.get("model","llava:7b"))
    # elif engine=="openai": ...
    # elif engine=="anthropic": ...
    return MockModel()
