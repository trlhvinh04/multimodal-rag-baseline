from pathlib import Path
from loguru import logger
from pdf2image import convert_from_path
from collections import Counter

def get_images_from_pdf(pdf_path, max_pages=None,dpi_resolution=144,
                        save_dir='/tmp',save_images=False,save_type='png'):
    """
    Convert PDF pages to images
    """
    pdf_path = Path(pdf_path)
    assert pdf_path.exists(), f'PDF file {pdf_path} does not exist'
    
    pdf_fname = pdf_path.name
    
    images = convert_from_path(pdf_path,dpi=dpi_resolution)
    
    # PIL.PpmImagePlugin.PpmImageFile  -> PIL.Image.Image
    images = [img.convert('RGB') for img in images]
    
    # Resizing to the most common image size so that we can stack in pytorch tensor
    
    # 1) find the most common image size
    img_size_counter = Counter()
    for img in images:
        img_size_counter[img.size] += 1
    common_img_size, common_img_size_count = img_size_counter.most_common(1)[0]
    
    # 2) if pages are not the same size, resize to the most common image size
    if len(images) != common_img_size_count:
        logger.info(f'Resizing {len(images)} images to {common_img_size}')
        images = [img.resize(common_img_size) for img in images]
    
    # 3) save images
    if save_images:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True,exist_ok=True)
        
        for page_idx, img in enumerate(images):
            img_path = save_dir / f'{pdf_fname}_{page_idx+1}.{save_type}'
            if not img_path.exists():
                img.save(img_path)
                logger.info(f'Saved image {img_path}')
            else:
                logger.info(f'Image {img_path} already exists')
    
    return images

if __name__ == '__main__':
    get_images_from_pdf(pdf_path='/home/tiamo/multimodal-rag-baseline/data/c0849d57260bcc2e2f8a399cb1eeb8e7.pdf',save_dir='/home/tiamo/multimodal-rag-baseline/data/images',save_images=True)
    