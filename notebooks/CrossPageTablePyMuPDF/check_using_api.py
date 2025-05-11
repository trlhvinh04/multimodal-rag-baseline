# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
import json
from google import genai
from google.genai import types
## ------- COLAB CODE -------
# from google.colab import userdata
# GEMINI_API_KEY = userdata.get('GOOGLE_API_KEY')

## ------- KAGGLE CODE -------
# from kaggle_secrets import UserSecretsClient
# user_secrets = UserSecretsClient()
# GEMINI_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")

## ------- LOCAL CODE -------
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

def generate(prompt):
    model = "gemini-2.0-flash-lite"
    client = genai.Client(
        api_key=GEMINI_API_KEY,
    )
    system_prompt = """You are an AI assistant specialized in analyzing and grouping data tables for vertical concatenation. I will provide you with information for multiple tables. Each table will include:

Table index (e.g., \"Table 0\").
Explicit headers (e.g., \"Headers: ColA,ColB,ColC\") if this table introduces a new header structure. For subsequent tables with the same structure and headers, the \"Headers:\" line can be omitted.
A few lines of sample data. Each data line contains values separated by the \"=\" character.
The number of columns.
A list of data types for the columns.
The tables will be presented sequentially.

Your task is to identify groups of tables that can be vertically concatenated. Tables are considered concatenable if they share the following characteristics:

The same number of columns.
Compatible data types for corresponding columns.
Identical or very similar headers. If headers are not explicitly provided for a table, assume it shares the headers of the immediately preceding table (or the closest table with explicit headers) if their number of columns and data types match.
Below is the table data:"""
    content = system_prompt + prompt
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0,
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type = genai.types.Type.OBJECT,
            required = ["concatable_tables"],
            properties = {
                "concatable_tables": genai.types.Schema(
                    type = genai.types.Type.ARRAY,
                    items = genai.types.Schema(
                        type = genai.types.Type.OBJECT,
                        required = ["table_index"],
                        properties = {
                            "table_index": genai.types.Schema(
                                type = genai.types.Type.ARRAY,
                                items = genai.types.Schema(
                                    type = genai.types.Type.NUMBER,
                                ),
                            ),
                        },
                    ),
                ),
            },
        ),
        system_instruction=[
            types.Part.from_text(text=system_prompt),
        ],
    )
    input_tokens = client.models.count_tokens(model=model, contents=[content])
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_content_config,
    )
    output_tokens = client.models.count_tokens(model=model, contents=response.text)

    json_response = json.loads(response.text)
    json_response.update({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    })
    return json_response
