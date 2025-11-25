import torch
from typing import Optional
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


class BaseModel:
    def generate(self, image: Image.Image, prompt: str) -> str:
        raise NotImplementedError


class QwenVLVL(BaseModel):
    """
    Correct interface for Qwen2-VL models.
    Uses Vision2Seq model class and chat template to align image tokens.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        # Processor handles text + image together
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Correct multimodal model class
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)

        self.model.eval()

    def generate(self, image: Image.Image, prompt: str) -> str:
        # Format as multimodal chat
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Insert <image> token
        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Build multimodal inputs
        inputs = self.processor(
            text=[chat_text],
            images=[image],
            return_tensors="pt",
        ).to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
            )

        # Decode
        output = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )[0]

        return output.strip()


class MockModel(BaseModel):
    def generate(self, image, prompt):
        return "MOCK_RESPONSE"


def build_model(engine: str, **kwargs) -> BaseModel:
    if engine == "qwen_vl":
        return QwenVLVL(
            model_name=kwargs.get("model_name", "Qwen/Qwen2-VL-7B-Instruct")
        )
    return MockModel()
