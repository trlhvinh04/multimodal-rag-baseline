# multimodal-rag-baseline
## Intruction
This project serves as a baseline implementation for a multimodal retrieval-augmented generation (RAG) system. It integrates multiple data modalities and provides a framework for efficient information retrieval and generation.

## Setup Instructions
1. Clone the repository:
```
   git clone <repository_url>
   cd multimodal-rag-baseline
```
2. Install dependencies
```
conda create -f environment.yml
# or pip install -r requirements.txt
```
3. Create a .env file in the root directory with the following format:
```
HUGGINGE_FACE_TOKEN=<your_api_key>
MONGO_URI=<your_database_url>
GEMINI_API_KEY=<your_model_name>
```