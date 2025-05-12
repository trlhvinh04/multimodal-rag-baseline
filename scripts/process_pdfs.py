#!/usr/bin/env python
# coding: utf-8

import os
import torch
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from colpali_engine.models import ColPali, ColPaliProcessor
from qdrant_client import QdrantClient
from qdrant_client.http import models
from utils.pdfs import get_images_from_pdf

# Configure logger
logger.remove()  # Remove default handler
logger.add(
    "process_pdfs.log",
    rotation="10 MB",
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
logger.add(
    lambda msg: tqdm.write(msg, end=""), 
    colorize=True, 
    format="<green>{time:HH:mm:ss}</green> <level>{level: <8}</level> {message}",
    level="INFO"
)

def initialize_qdrant_client():
    """Initialize Qdrant client."""
    logger.info("Initializing Qdrant client...")
    client = QdrantClient(
        ":memory:"  # Use ":memory:" for in-memory database or "path/to/db" for persistent storage
    )
    logger.success("Qdrant client initialized")
    return client

def load_colpali_model_and_processor():
    """Load ColPali model and processor."""
    logger.info("Loading ColPali model and processor...")
    model = ColPali.from_pretrained(
        "vidore/colpali-v1.2",
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # or "mps" if on Apple Silicon 
    )
    processor = ColPaliProcessor.from_pretrained("vidore/colpaligemma-3b-pt-448-base")
    logger.success("ColPali model and processor loaded")
    return model, processor

def create_collection(qdrant_client, collection_name):
    """Create a Qdrant collection."""
    logger.info(f"Creating collection: {collection_name}")
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        on_disk_payload=True,
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=10
        ),
        vectors_config=models.VectorParams(
            size=128,
            distance=models.Distance.COSINE,
            multivector_config=models.MultiVectorConfig(
                comparator=models.MultiVectorComparator.MAX_SIM
            ),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True,
                ),
            ),
        ),
    )
    logger.success(f"Collection '{collection_name}' created successfully")

def process_pdf(pdf_path, model, processor, qdrant_client, collection_name, start_idx=0):
    """Process a single PDF file."""
    logger.info(f"Processing PDF: {pdf_path}")
    
    # Convert PDF to images
    images = get_images_from_pdf(pdf_path)
    if not images:
        logger.warning(f"No images found in {pdf_path}")
        return
    
    # Process images in batches
    batch_size = 6
    points = []
    
    with tqdm(total=len(images), desc=f'Processing {Path(pdf_path).name}') as pbar:
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            
            # Process and encode images
            with torch.no_grad():
                processed_images = processor.process_images(batch_images).to(model.device)
                image_embeddings = model(**processed_images)
            
            # Prepare points for QDrant
            for j, embedding in enumerate(image_embeddings):
                # Convert the embedding to a list vectors
                multivector = embedding.cpu().float().numpy().tolist()
                points.append(
                    models.PointStruct(
                        id=start_idx + i + j,
                        vector=multivector,
                        payload={
                            'source': str(pdf_path),
                            'page': i + j + 1
                        },
                    )
                )
            
            pbar.update(len(batch_images))
    
    # Upload points to Qdrant
    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
        logger.success(f"Successfully indexed {len(points)} pages from {pdf_path}")
    except Exception as e:
        logger.error(f"Error during upsert for {pdf_path}: {e}")

def main():
    # Initialize components
    qdrant_client = initialize_qdrant_client()
    model, processor = load_colpali_model_and_processor()
    
    # Create collection
    collection_name = 'pdf_collection'
    create_collection(qdrant_client, collection_name)
    
    # Get list of PDF files
    pdf_dir = Path('data')
    pdf_files = list(pdf_dir.glob('**/*.pdf'))
    
    if not pdf_files:
        logger.warning("No PDF files found in data directory")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    # Process each PDF
    current_idx = 0
    for pdf_file in pdf_files:
        process_pdf(
            pdf_file,
            model,
            processor,
            qdrant_client,
            collection_name,
            start_idx=current_idx
        )
        # Update index for next PDF
        current_idx += len(get_images_from_pdf(pdf_file))

    logger.success("Completed processing all PDFs")

if __name__ == "__main__":
    main() 