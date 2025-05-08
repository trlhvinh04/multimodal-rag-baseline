#!/usr/bin/env python
# coding: utf-8

import os
import torch
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm
from datasets import load_dataset as hf_load_dataset
import stamina
from colpali_engine.models import ColPali, ColPaliProcessor
from rich import print as r_print
from PIL import Image
from loguru import logger

# Configure logger
logger.remove()  # Remove default handler
logger.add(
    "demo_colpali_qdrant.log",
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

# Set environment variables
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def load_dataset():
    """Load dataset from Hugging Face."""
    logger.info("Loading dataset...")
    dataset = hf_load_dataset("hwyin04/pdf-test")
    logger.success(f"Dataset loaded: {dataset}")
    return dataset

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

@stamina.retry(on=Exception, attempts=3)
def upsert_to_qdrant(qdrant_client, collection_name, points):
    """Upsert points to Qdrant with retry capabilities."""
    try:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points,
            wait=False,
        )
    except Exception as e:
        logger.error(f"Error during upsert: {e}")
        return False
    return True

def index_images(qdrant_client, collection_name, dataset, colpali_model, colpali_processor, batch_size=6):
    """Index images in Qdrant."""
    logger.info(f"Indexing images with batch size {batch_size}...")
    with tqdm(total=len(dataset), desc='Indexing Progress') as pbar:
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            images = batch["image"]
            
            # Process and encode images
            logger.debug(f"Processing batch {i//batch_size + 1}/{(len(dataset) + batch_size - 1)//batch_size}")
            with torch.no_grad():
                batch_images = colpali_processor.process_images(images).to(
                    colpali_model.device
                )
                image_embeddings = colpali_model(**batch_images)
            
            # Prepare points for QDrant
            points = []
            for j, embedding in enumerate(image_embeddings):
                # Convert the embedding to a list vectors
                multivector = embedding.cpu().float().numpy().tolist()
                points.append(
                    models.PointStruct(
                        id=i+j,  # index as ID
                        vector=multivector,
                        payload={
                            'source': 'internet archive'
                        },  # can add other metadata
                    )
                )
            
            # Upload Points to Qdrant
            try:
                upsert_to_qdrant(qdrant_client, collection_name, points)
                logger.debug(f"Upserted batch {i//batch_size + 1} ({len(points)} points)")
            except Exception as e:
                logger.error(f"Error during upsert: {e}")
                continue
            
            pbar.update(min(batch_size, len(batch)))
    
    logger.success("Indexing Complete")

def search_images_by_text(qdrant_client, collection_name, colpali_model, colpali_processor, query_text, top_k=5):
    """Search images by text query."""
    logger.info(f"Searching for: '{query_text}' (top {top_k})")
    # Process and encode the text query
    with torch.no_grad():
        batch_query = colpali_processor.process_queries([query_text]).to(
            colpali_model.device
        )
        query_embedding = colpali_model(**batch_query)
    
    # Convert the query embedding to a list of vectors
    multivector_query = query_embedding[0].cpu().float().numpy().tolist()
    
    # Search in Qdrant
    search_result = qdrant_client.query_points(
        collection_name=collection_name, 
        query=multivector_query, 
        limit=top_k
    )
    
    logger.debug(f"Found {len(search_result.points)} results")
    return search_result

def search_by_text_and_return_images(qdrant_client, collection_name, dataset, colpali_model, colpali_processor, query_text, top_k=5):
    """Search by text and return corresponding images."""
    results = search_images_by_text(qdrant_client, collection_name, colpali_model, colpali_processor, query_text, top_k)
    row_ids = [r.id for r in results.points]
    logger.debug(f"Getting images for IDs: {row_ids}")
    return dataset.select(row_ids), results

def display_collection_info(qdrant_client, collection_name):
    """Display collection information."""
    logger.info(f"Getting information for collection '{collection_name}'")
    collection = qdrant_client.get_collection(collection_name)
    logger.debug(f"Collection info: {collection}")
    r_print(collection)

def run_example_search(qdrant_client, collection_name, dataset, colpali_model, colpali_processor):
    """Run an example search and display results."""
    # Example usage
    logger.info("Running example search")
    query_text = "2023 ballot"
    logger.info(f"Query: '{query_text}'")
    
    results_ds, results = search_by_text_and_return_images(
        qdrant_client, collection_name, dataset, colpali_model, colpali_processor, query_text
    )
    
    # Print search results
    for i, result in enumerate(results.points):
        logger.info(f"Result {i+1}: ID={result.id}, Score={result.score}")
    
    # Save retrieved images
    logger.info("Saving retrieved images")
    for i, row in enumerate(results_ds):
        img_path = f"result_{i+1}.png"
        logger.info(f"Image {i+1}: ID={row['id']} -> {img_path}")
        row["image"].save(img_path)

def main():
    """Main function to orchestrate the demo."""
    logger.info("Starting ColPali + Qdrant demo")
    
    # Setup
    dataset = load_dataset()
    qdrant_client = initialize_qdrant_client()
    colpali_model, colpali_processor = load_colpali_model_and_processor()
    
    # Collection
    collection_name = 'ufo'
    create_collection(qdrant_client, collection_name)
    
    # Index images
    index_images(qdrant_client, collection_name, dataset, colpali_model, colpali_processor)
    
    # Display collection info
    display_collection_info(qdrant_client, collection_name)
    
    # Run example search
    run_example_search(qdrant_client, collection_name, dataset, colpali_model, colpali_processor)
    
    logger.success("Demo completed successfully")

if __name__ == "__main__":
    logger.info("=== Starting ColPali + Qdrant Demo ===")
    main()
    logger.info("=== Demo Finished ===") 