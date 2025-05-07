import glob, sys, fitz
import os
import logging
from typing import List, Union
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

class PDF2ImgConverter:
    def __init__(self, pdf_path: str, output_dir: str):
        self.pdf_path = pdf_path
        self.output_dir = output_dir

    def convert_specific_file(self, filename: Union[str, List[str]] = None, output_dir: str = None):
        if output_dir is None:
            output_dir = self.output_dir
            
        if isinstance(filename, str):
            filename = [filename]
        for f in filename:
            zoom_x = 2.0  # horizontal zoom
            zoom_y = 2.0  # vertical zoom
            mat = fitz.Matrix(zoom_x, zoom_y) 

            doc = fitz.open(f)  # open document
            for page in doc:
                pix = page.get_pixmap(matrix=mat)  # render page to an image
                output_path = os.path.join(output_dir, f"page-{page.number:04d}.png")
                pix.save(output_path)  # store image as a PNG
                # logging.info(f"Saved page {page.number} of {filename}")

    def convert_all_files(self):
        all_files = glob.glob(self.pdf_path + "*.pdf")
        for f in tqdm(all_files):
            logging.info(f"Converting {f}")
            # store with the file name in prefix
            base_name = os.path.splitext(os.path.basename(f))[0]
            output_dir = os.path.join(self.output_dir, base_name)
            os.makedirs(output_dir, exist_ok=True)
            self.convert_specific_file(f, output_dir)
            

if __name__ == "__main__":
    converter = PDF2ImgConverter(pdf_path='data/', output_dir='imgs')
    converter.convert_all_files()