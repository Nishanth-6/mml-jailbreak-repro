import io
from typing import Optional

from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq


class BaseModel:
    def generate(self, image: Image.Image, prompt: str) -> str:
        raise NotImplementedError


class QwenVLVL(BaseModel):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[QwenVLVL] Loading model: {model_name} on {self.device}")

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)

        self.model.eval()

    def generate(self, image: Image.Image, prompt: str) -> str:
        # Qwen2-VL expects the image inside the messages structure, not via images= argument
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor(
            messages,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
            )

        generated = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )[0]

        return generated.strip()


class MockModel(BaseModel):
    def generate(self, image, prompt):
        return "MOCK_RESPONSE"


def build_model(engine: str, **kwargs) -> BaseModel:
    if engine == "qwen_vl":
        return QwenVLVL(
            model_name=kwargs.get("model_name", "Qwen/Qwen2-VL-7B-Instruct"),
            device=kwargs.get("device"),
        )
    return MockModel()
