"""
Script dùng cho mục đích đưa ảnh đầu vào với dạng PIL.Image.Image hoặc là đường dẫn đến file ảnh sau đó tạo ra caption cho ảnh đó.

Có 2 cách để tạo ra caption là sử dụng:
1. Gemini
2. Local model

Nếu sử dụng Gemini thì cần đưa GOOGLE_API_KEY vào trong .env file

Nếu sử dụng local model thì cần tìm model checkpoint trên Hugging Face và thực hiện như sau:

```python
model_checkpoint = "Salesforce/blip-image-captioning-base"

caption = generate_caption_with_local_model(
        checkpoint=model_checkpoint,
        image_path=image_path,
    )
```
Nó sẽ tải mô hình từ Hugging Face và tạo ra caption cho ảnh đầu vào.
"""

import base64
import io
import logging
import os
import sys
from time import time
from typing import Any, Optional, Tuple, Union

import PIL
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.thangquang_config import default_config

logging.basicConfig(level=default_config.log_level)
logger = logging.getLogger(__name__)


class ImageCaptionOutput(BaseModel):
    image_captions: str = Field(description="The generated caption of image")


def get_image_base64(image: Optional[Image.Image] = None) -> Union[str, None]:
    """
    Get image base64
    Args:
        image (PIL.Image): PIL Image object
    Returns:
        str: image base64
    """
    try:
        if image is None:
            logger.error("No image provided")
            return None
        if not isinstance(image, Image.Image):
            logger.error("Image must be a PIL.Image.Image object")
            return None
        image = image.convert("RGB")
        image.thumbnail(default_config.max_image_size, Image.Resampling.LANCZOS)

        buffered = io.BytesIO()
        image.save(buffered, format=default_config.image_format)
        img_byte = buffered.getvalue()
        return base64.b64encode(img_byte).decode("utf-8")
    except Exception as e:
        logger.error(f"Error getting image base64: {e}")
        return None


def load_pil_image(
    image: Optional[Image.Image] = None, image_path: Optional[str] = None
) -> Optional[Image.Image]:
    """
    Load PIL image
    Args:
        image (PIL.Image): PIL Image object
        image_path (str): Path to image file
    Returns:
        PIL.Image: PIL image
    """
    if image is not None:
        if isinstance(image, Image.Image):
            return image
        else:
            logger.error("Image must be a PIL.Image.Image object")
            return None
    elif image_path is not None:
        try:
            if not os.path.exists(image_path):
                logger.error(f"File image not found at path: {image_path}")
                return None
            return Image.open(image_path)
        except FileNotFoundError:  # Thực tế đã check ở trên
            logger.error(f"File image not found at path: {image_path}")
            return None
        except PIL.UnidentifiedImageError:
            logger.error(
                f"Cannot identify image format or file is corrupted: {image_path}"
            )
            return None
        except Exception as e:
            logger.error(f"Error opening image from path: {image_path}")
            return None
    else:
        logger.error("Need to provide 'image' (PIL.Image.Image) or 'image_path' (str).")
        return None


def get_llm_vision_gemini() -> Optional[ChatGoogleGenerativeAI]:
    """
    Get LLM vision Gemini
    Returns:
        ChatGoogleGenerativeAI: LLM vision Gemini
    """
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY not found in environment variables.")
            return None
        llm_vision = ChatGoogleGenerativeAI(
            model=default_config.model_name,
            temperature=default_config.temperature,
            api_key=api_key,
            max_output_tokens=default_config.max_output_tokens,
        )
        return llm_vision
    except Exception as e:
        logger.error(f"Error getting LLM vision: {e}")
        return None


def generate_caption_with_gemini(
    image: Optional[Image.Image] = None,
    image_path: Optional[str] = None,
    prompt: str = default_config.default_prompt,
) -> Union[str, None]:
    """
    Generate caption with Gemini
    Args:
        image (PIL.Image): PIL Image object
        image_path (str): Path to image file
        prompt (str): Prompt for image captioning
    Returns:
        str: Generated caption
    """
    from langchain_core.messages import HumanMessage

    pil_image: Optional[Image.Image] = load_pil_image(
        image=image, image_path=image_path
    )
    if pil_image is None:
        logger.error("Failed to load image")
        return None

    # create llm_vision
    llm_vision = get_llm_vision_gemini()
    if not llm_vision:
        logger.error("Model not found")
        return None
    llm_with_structed_output = llm_vision.with_structured_output(ImageCaptionOutput)

    image_b64 = get_image_base64(image=pil_image)
    if not image_b64:
        logger.error("Failed to get image base64")
        return None

    message_content = [
        {
            "type": "text",
            "text": prompt,
        },
        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"},
    ]

    human_message = HumanMessage(content=message_content)

    # sending request
    start_time = time()
    try:
        logger.info("Sending Request")
        response = llm_with_structed_output.invoke([human_message])
        caption = response.image_captions
        end_time = time()
        logger.info(f"Time taken: {end_time - start_time:.4f} seconds")
        # print(f"Response: {caption}")
        return caption.strip() if isinstance(caption, str) else str(caption).strip()
    except Exception as e:
        logger.error(f"Error: {e}")
        return None


def get_llm_vision_local(
    checkpoint: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Get LLM vision local
    Args:
        checkpoint (str): Model checkpoint on Hugging Face, example: "Salesforce/blip-image-captioning-base"
    Returns:
        tuple: processor and model
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoProcessor,
        BlipForConditionalGeneration,
        BlipProcessor,
    )

    if checkpoint is None:
        logger.error("Checkpoint not found")
        return None, None
    logger.info(f"Loading model from {checkpoint}")
    start_time = time()
    try:
        processor = AutoProcessor.from_pretrained(
            checkpoint, use_fast=True, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(checkpoint)
    except ValueError:
        processor = BlipProcessor.from_pretrained(
            checkpoint, use_fast=True, trust_remote_code=True
        )
        model = BlipForConditionalGeneration.from_pretrained(checkpoint)
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None
    end_time = time()
    logger.info(f"Loading model time taken: {end_time - start_time:.4f} seconds")
    return processor, model


def generate_caption_with_local_model(
    checkpoint: Optional[str] = None,
    image: Optional[Image.Image] = None,
    image_path: Optional[str] = None,
    prompt: str = default_config.default_prompt,
    processor_and_model: Optional[Tuple[Any, Any]] = None,
) -> Union[str, None]:
    """
    Generate caption with local model
    Args:
        checkpoint (str): Model checkpoint on Hugging Face
        image (PIL.Image): PIL Image object
        image_path (str): Path to image file
        prompt (str): Prompt for image captioning
        processor_and_model (tuple): Tuple of processor and model
    Returns:
        str: Generated caption
    """
    from accelerate.test_utils.testing import get_backend

    pil_image: Optional[Image.Image] = load_pil_image(
        image=image, image_path=image_path
    )
    if pil_image is None:
        logger.error("Failed to load image")
        return None

    try:
        if processor_and_model is None:
            logger.warning("Processor and model not found, using default config")
            processor_and_model = get_llm_vision_local(checkpoint=checkpoint)

        processor, model = processor_and_model

        start_time = time()
        device, _, _ = get_backend()
        inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(
            device
        )
        model = model.to(device)
        pixel_values = inputs.pixel_values
        generated_ids = model.generate(
            pixel_values=pixel_values, max_length=default_config.max_output_tokens
        )
        generated_caption = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        end_time = time()
        logger.info(f"Time taken: {end_time - start_time:.4f} seconds")
        return (
            generated_caption.strip()
            if isinstance(generated_caption, str)
            else str(generated_caption).strip()
        )
    except Exception as e:
        logger.error(f"Error generating caption with local model: {e}")
        return None


if __name__ == "__main__":
    image_path = "data/raw/imgs/image.png"  # Paste your image path here

    print("Generating caption with local model")

    caption = generate_caption_with_local_model(
        checkpoint="Salesforce/blip-image-captioning-base",
        image_path=image_path,
    )
    print(caption)
    print("-" * 80)

    print("Generating caption with Gemini")
    caption = generate_caption_with_gemini(
        image_path=image_path,
    )
    print(caption)
    print("-" * 80)
