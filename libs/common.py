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


import os
import json
from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
# from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field
from typing import List, Optional
from google.oauth2 import service_account
import re