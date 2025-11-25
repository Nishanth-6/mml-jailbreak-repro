import os, json, subprocess, tempfile
from typing import Optional
from PIL import Image


class BaseModel:
    def generate(self, image: Image.Image, prompt: str) -> str:
        raise NotImplementedError


class OllamaVL(BaseModel):
    def __init__(self, model: str = "llava:7b"):
        self.model = model

    def generate(self, image: Image.Image, prompt: str) -> str:
        # Save the image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
            image.save(tmp_path)

        # Build JSON payload
        payload = {"model": self.model, "prompt": prompt, "images": [tmp_path]}

        try:
            # Call the Ollama JSON API via stdin
            proc = subprocess.Popen(
                ["ollama", "run", self.model],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            out, err = proc.communicate(json.dumps(payload))

            if proc.returncode != 0:
                return f"ERROR: {err.strip()}"

            return out.strip()

        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass


class MockModel(BaseModel):
    def generate(self, image, prompt):
        return "I'm sorry, I cannot assist with that."


def build_model(engine: str, **kwargs) -> BaseModel:
    if engine == "ollama":
        return OllamaVL(model=kwargs.get("model", "llava:7b"))
    return MockModel()
