from typing import Optional
from pydantic import BaseModel, Field

class ImageCaptionConfig(BaseModel):
    # LLM Vision Model Configuration
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.3
    max_output_tokens: int = 100
    
    # Image Processing Configuration
    max_image_size: tuple[int, int] = (1024, 1024)
    image_format: str = "JPEG"
    
    # Default Prompt Configuration
    default_prompt: str = "Describe this image in detail to be used as a caption. Focus on the main subjects, actions, and setting."
    
    # Logging Configuration
    log_level: str = "INFO"

# Create default configuration instance
default_config = ImageCaptionConfig()
