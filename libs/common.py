import os
import json
import shutil
import requests

from sentence_transformers import SentenceTransformer
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
import fitz  # PyMuPDF
from PIL import Image
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv,find_dotenv
import pytesseract
from datasets import load_dataset
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"