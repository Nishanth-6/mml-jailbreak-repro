import io
from typing import Optional

from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForCausalLM


class BaseModel:
    def generate(self, image: Image.Image, prompt: str) -> str:
        raise NotImplementedError


class QwenVLVL(BaseModel):
    """
    Vision-language wrapper for Qwen2-VL-7B. Uses the official chat_template +
    (text, image) interface so that image tokens and features line up correctly.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Processor handles both text and images
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )

        # Load model on GPU if available
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)
        self.model.eval()

    def generate(self, image: Image.Image, prompt: str) -> str:
        """
        Build a chat message with an image placeholder, then let the processor
        turn it into text + image tokens. This avoids the
        'Image features and image tokens do not match' error.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Chat template inserts the special image token into the text stream
        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Processor now sees both text and image
        inputs = self.processor(
            text=[chat_text],
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
            )

        generated = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]
        return generated.strip()


class MockModel(BaseModel):
    def generate(self, image, prompt):
        # Dummy model for local testing
        return "MOCK_RESPONSE"


def build_model(engine: str, **kwargs) -> BaseModel:
    if engine == "qwen_vl":
        return QwenVLVL(
            model_name=kwargs.get("model_name", "Qwen/Qwen2-VL-7B-Instruct")
        )
    # Fallback used on your laptop
    return MockModel()
