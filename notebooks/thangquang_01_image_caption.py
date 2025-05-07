import base64
import io
import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from time import time
from typing import Union

import PIL
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from PIL import Image
from pydantic import BaseModel, Field

from configs.thangquang_config import default_config

logging.basicConfig(level=default_config.log_level)
logger = logging.getLogger(__name__)


def get_llm_vision():
    try:
        llm_vision = ChatGoogleGenerativeAI(
            model=default_config.model_name,
            temperature=default_config.temperature,
            api_key=os.environ.get("GOOGLE_API_KEY"),
            max_output_tokens=default_config.max_output_tokens,
        )
        return llm_vision
    except Exception as e:
        logger.error(f"Error getting LLM vision: {e}")
        return None


def get_image_base64(
    image: PIL.Image.Image = None, image_path: str = None
) -> Union[str, None]:
    try:
        if image is None and image_path is None:
            raise ValueError("Either image or image_path must be provided")
        if image is None:
            image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.thumbnail(default_config.max_image_size)
        buffered = io.BytesIO()
        image.save(buffered, format=default_config.image_format)
        img_byte = buffered.getvalue()
        return base64.b64encode(img_byte).decode("utf-8")
    except FileNotFoundError:
        logger.error(f"File not found: {image_path}")
        return None
    except Exception as e:
        logger.error(f"Error getting image base64: {e}")
        return None


def generate_caption_with_gemini(
    image: PIL.Image.Image = None,
    image_path: str = None,
    prompt: str = default_config.default_prompt,
) -> Union[str, None]:
    if image is None and image_path is None:
        raise ValueError("Either image or image_path must be provided")
    if image is None:
        image = Image.open(image_path)

    class ImageCaption(BaseModel):
        image_captions: str = Field(description="The generated caption of image")

    # create llm_vision
    llm_vision = get_llm_vision()
    if not llm_vision:
        logger.error("Model not found")
        return None
    llm_with_structed_output = llm_vision.with_structured_output(ImageCaption)

    image_b64 = get_image_base64(image=image)
    if not image_b64:
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
        logger.info(f"Time taken: {end_time - start_time} seconds")
        # print(f"Response: {caption}")
        return caption.strip() if isinstance(caption, str) else str(caption).strip()
    except Exception as e:
        logger.error(f"Error: {e}")
        return None


if __name__ == "__main__":
    image_path = "data/raw/imgs/image.png"
    caption = generate_caption_with_gemini(image_path=image_path)
    print(caption)
