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
import functools
import io
import logging
import os
import sys
from time import perf_counter
from typing import Any, Callable, Optional, Tuple, Union

import PIL
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from pydantic import BaseModel, Field

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


class DefaultConfig:
    # LLM Vision Model Configuration
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.3
    max_output_tokens: int = 100

    # Image Processing Configuration
    max_image_size: tuple[int, int] = (1024, 1024)
    image_format: str = "JPEG"

    # Default Prompt Configuration
    default_prompt: str = "Describe this image in detail to be used as a caption. Focus on the main subjects, actions, and setting."


class Timer:
    """A context manager for timing code blocks with nested timing support"""

    _timers: dict[str, float] = {}  # Class variable to store all timings

    def __init__(self, name: str, logger_instance: logging.Logger):
        self.name = name
        self.logger = logger_instance
        self.start_time = None
        self.parent = None

    def __enter__(self):
        self.start_time = perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = perf_counter() - self.start_time
            Timer._timers[self.name] = duration
            self.logger.info(f"Step '{self.name}' took {duration:.4f} seconds")

    @classmethod
    def get_timer(cls, name: str) -> float:
        """Get the duration of a specific timer"""
        return cls._timers.get(name, 0.0)

    @classmethod
    def clear_timers(cls):
        """Clear all stored timings"""
        cls._timers.clear()


def time_function(name: Optional[str] = None) -> Callable:
    """Decorator for timing functions"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timer_name = name or func.__name__
            with Timer(timer_name, logger):
                return func(*args, **kwargs)

        return wrapper

    return decorator


default_config = DefaultConfig()


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
    if image is None:
        logger.error("No image provided")
        return None
    if not isinstance(image, Image.Image):
        logger.error("Image must be a PIL.Image.Image object")
        return None
    try:
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
        except FileNotFoundError:
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


@time_function("get_llm_vision_gemini")
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


@time_function("generate_caption_with_gemini")
def generate_caption_with_gemini(
    image: Optional[Image.Image] = None,
    image_path: Optional[str] = None,
    client: Optional[ChatGoogleGenerativeAI] = None,
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
    if client is None:
        llm_vision = get_llm_vision_gemini()
    else:
        llm_vision = client
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
    try:
        logger.info("Sending Request")
        response = llm_with_structed_output.invoke([human_message])
        caption = response.image_captions
        return caption.strip() if isinstance(caption, str) else str(caption).strip()
    except Exception as e:
        logger.error(f"Error: {e}")
        return None


@time_function("get_llm_vision_local")
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
    return processor, model


@time_function("generate_caption_with_local_model")
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
        return (
            generated_caption.strip()
            if isinstance(generated_caption, str)
            else str(generated_caption).strip()
        )
    except Exception as e:
        logger.error(f"Error generating caption with local model: {e}")
        return None


if __name__ == "__main__":
    image_path = "notebooks/Docling/anh-con-cho-51.jpg"  # Paste your image path here

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
