# src/mml/model_interface.py

import io
from typing import Optional

from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq


class BaseModel:
    def generate(self, image: Image.Image, prompt: str) -> str:
        raise NotImplementedError


class QwenVLVL(BaseModel):
    """
    Vision-language wrapper around Qwen2-VL.

    - Tries to load 7B in bfloat16 on GPU with low_cpu_mem_usage
    - If CUDA OOM happens during load, automatically falls back to 2B
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        max_new_tokens: int = 128,
        device: Optional[str] = None,
    ):
        self.requested_model = model_name
        self.max_new_tokens = max_new_tokens

        # Decide device + dtype once
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        def _load(name: str):
            print(f"[QwenVLVL] Loading model: {name} on {self.device} with dtype={self.dtype}")
            processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
            model = AutoModelForVision2Seq.from_pretrained(
                name,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
            ).to(self.device)
            model.eval()
            return name, processor, model

        try:
            self.model_name, self.processor, self.model = _load(self.requested_model)
        except torch.cuda.OutOfMemoryError:
            # Automatic graceful fallback to 2B
            fallback = "Qwen/Qwen2-VL-2B-Instruct"
            print(
                f"[QwenVLVL] CUDA OOM while loading {self.requested_model}. "
                f"Falling back to smaller model: {fallback}"
            )
            torch.cuda.empty_cache()
            self.model_name, self.processor, self.model = _load(fallback)

    def generate(self, image: Image.Image, prompt: str) -> str:
        # Chat-style message as expected by Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor(
            messages,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
            )

        generated = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]
        return generated.strip()


class MockModel(BaseModel):
    def generate(self, image, prompt):
        return "MOCK_RESPONSE"


def build_model(engine: str, **kwargs) -> BaseModel:
    if engine == "qwen_vl":
        return QwenVLVL(
            model_name=kwargs.get("model_name", "Qwen/Qwen2-VL-7B-Instruct"),
            max_new_tokens=kwargs.get("max_new_tokens", 128),
        )
    return MockModel()
