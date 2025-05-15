# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
import json
from google import genai
from google.genai import types
# ------- COLAB CODE -------
# from google.colab import userdata
# GEMINI_API_KEY = userdata.get('GOOGLE_API_KEY')

# ------- KAGGLE CODE -------
# from kaggle_secrets import UserSecretsClient
# user_secrets = UserSecretsClient()
# GEMINI_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")

# ------- LOCAL CODE -------
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")


def generate(prompt):
    model = "gemini-2.0-flash-lite"
    client = genai.Client(
        api_key=GEMINI_API_KEY,
    )
    system_prompt = """You are an AI assistant specialized in analyzing and grouping data tables for vertical concatenation. I will provide you with information for multiple tables. Each table entry might include:

Context before table: A line indicating the subject matter preceding the table. This context can provide clues about the table's topic.
Table index: (e.g., Table 0).
Sample data: A few lines of data. Values in data lines are separated by the = character. The first line of sample data may or may not be an explicit header row containing column names.
Number of columns.
A list of data types for the columns.

The tables will be presented sequentially.

Your tasks are:

First, for each table, meticulously examine its first line of sample data. Determine if this first line explicitly serves as a header row containing distinct column names, rather than being a row of actual data. Identify the table indexes of only those tables that possess such an explicit, pre-existing header row in their sample data. If a table's first row contains data values, even if structured, it should not be considered as having an explicit pre-existing header in this step. It is possible that no tables, or only some, have explicit pre-existing headers.

Second, identify groups of tables that can be vertically concatenated. Tables are considered concatenable if they share the following characteristics:

Relevant Context: The Context before table information, if provided, suggests the tables belong to the same logical section or topic. Use this context as a strong indicator for grouping.
Same Number of Columns.
Compatible Data Types for corresponding columns.
Consistent Header Structure (either pre-existing or to be inferred): If a table in a potential group has an explicit pre-existing header (as identified in the first task), other tables considered for this group should ideally align with this structure. For tables without explicit pre-existing headers, their structure must be compatible with the anticipated common header structure for the group.
Third, for every table, you must provide a definitive list of column headers. Also, for each table, you must determine if these provided headers were pre-existing in the sample data or if they were inferred/predicted by you.
If a table was identified in the first task as having an explicit pre-existing header, you must use those exact pre-existing headers for that table. The determination of whether these headers were guessed or not should reflect this initial finding (i.e., not guessed).
If a table was not identified in the first task as having an explicit pre-existing header, you are required to infer or predict appropriate column headers for it. When inferring headers for a table that is part of a concatenation group, prioritize using or adapting the explicit pre-existing headers from another table within the same group, if such a table exists. If the table is independent or no table in its group has an explicit pre-existing header, infer the headers based on its data content, data types, and any available context. For any table where headers are inferred or predicted by you, this fact must be indicated.

Finally, when constructing the information about headers for all tables, pay attention to consolidation. If multiple table indexes, particularly those grouped together for concatenation as identified in the second task, share the exact same list of column headers and the same status regarding whether those headers were guessed or not, you should consolidate them into a single descriptive entry. This single entry should list all relevant table indexes in its table_index array. For tables that do not fit into such a consolidated group (e.g., independent tables with unique headers, or groups with varying header statuses), create individual descriptive entries as appropriate.

Group the tables based on these criteria for concatenation.

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
        max_output_tokens=1000,
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=["concatable_tables", "has_headers", "headers_info"],
            properties={
                "concatable_tables": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        required=["table_index"],
                        properties={
                            "table_index": genai.types.Schema(
                                type=genai.types.Type.ARRAY,
                                items=genai.types.Schema(
                                    type=genai.types.Type.NUMBER,
                                ),
                            ),
                        },
                    ),
                ),
                "has_headers": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.NUMBER,
                    ),
                ),
                "headers_info": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        required=["table_index", "headers",
                                  "is_header_guessed"],
                        properties={
                            "table_index": genai.types.Schema(
                                type=genai.types.Type.ARRAY,
                                items=genai.types.Schema(
                                    type=genai.types.Type.NUMBER,
                                ),
                            ),
                            "headers": genai.types.Schema(
                                type=genai.types.Type.ARRAY,
                                items=genai.types.Schema(
                                    type=genai.types.Type.STRING,
                                ),
                            ),
                            "is_header_guessed": genai.types.Schema(
                                type=genai.types.Type.BOOLEAN,
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
    output_tokens = client.models.count_tokens(
        model=model, contents=response.text)

    json_response = json.loads(response.text)
    json_response.update({"input_tokens": input_tokens,
                         "output_tokens": output_tokens})
    return json_response
