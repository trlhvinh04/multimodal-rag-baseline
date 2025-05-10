# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
import random

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


def prepare_df_for_prompt(df1, df2, n_rows=7):
    def get_input_df(df):
        columns = df.columns.tolist()
        # Check if any columns start with "Unnamed" or are numeric indices
        unnamed_cols = [col for col in columns if str(col).startswith("Unnamed")]
        # check columns is all digits
        numeric_cols = all(
            isinstance(col, int) or (isinstance(col, str) and col.isdigit())
            for col in columns
        )
        # check if NAN exists in columns
        nan_cols = any(pd.isna(col) for col in df.columns)
        header = not (
            bool(unnamed_cols) or numeric_cols or nan_cols
        )  # Don't show header if unnamed or numeric columns or has NaN columns

        if len(df) < n_rows:
            if header:
                return f"Headers: {','.join(columns)}\n" + df.to_csv(
                    index=False, header=False, sep="="
                )
            return df.to_csv(index=False, header=False, sep="=")
        else:
            # Calculate number of rows for each section
            n_edge = n_rows // 3  # Number of rows to take from start and end
            n_random = n_rows - (2 * n_edge)  # Remaining rows will be random

            # Get first and last n_edge rows
            first_rows = df.head(n_edge)
            last_rows = df.tail(n_edge)
            # Get random rows from middle section
            middle_section = df.iloc[n_edge:-n_edge]
            random_rows = (
                middle_section.sample(n=n_random)
                if len(middle_section) >= n_random
                else middle_section
            )

            concated_df = pd.concat([first_rows, last_rows, random_rows])
            if header:
                return f"Headers: {','.join(columns)}\n" + concated_df.to_csv(
                    index=False, header=False, sep="="
                )
            return concated_df.to_csv(index=False, header=False, sep="=")

    input_df1 = get_input_df(df1)
    input_df2 = get_input_df(df2)

    # Get number of columns for each dataframe
    n1 = len(df1.columns)
    n2 = len(df2.columns)

    # Get data types for each column, handling all-NaN columns
    def get_column_types(df):
        types = []
        for col in df.columns:
            if df[col].isna().all():
                types.append("unknown")
            else:
                dtype = df[col].dtype
                types.append(
                    "numeric" if pd.api.types.is_numeric_dtype(dtype) else str(dtype)
                )
        return types

    data_type_df1 = get_column_types(df1)
    data_type_df2 = get_column_types(df2)

    prompt = f"""
You are skilled at analyzing fragmented tables. Given two table fragments (`table_fragment_1`, `table_fragment_2`) in plain text (CSV) but the separator is "=" perform the following:

Tasks:
1. Check if `table_fragment_2` is a continuation of `table_fragment_1`. Consider:
    Column count must match.
    Data types must be consistent across columns.
    Logical flow (softer criterion).
    If a header is provided, you should assess the semantic alignment of the data with that header.
    If a column has an 'unknown' data type, you should ignore it when checking for data type consistency.

if they have the same number of columns, please check step by step
2. If not a continuation:
    Identify a potential header row in `table_fragment_2`:
    A header row is only valid if all cells are non-empty after trimming.
    If valid:
      Cells should describe the data below.
      The pattern should differ from data rows.
    If no valid header is found, `second_table_header` must be an empty list `[]`.
    
### table_fragment_1
{input_df1}

Note that:
- Number of columns is: {n1} 
- Data type of each column is: {data_type_df1}

### table_fragment_2
{input_df2}

Note that:
- Number of columns is: {n2} 
- Data type of each column is: {data_type_df2}
"""
    return prompt


def handle_response(response):
    import json

    try:
        # Parse the response text as JSON
        result = json.loads(response)
        return result
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON response from LLM")
    except KeyError as e:
        raise ValueError(f"Missing required field in response: {e}")


def generate(prompt):
    # Initialize client and model constants
    MODEL = "gemini-2.0-flash-lite"
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    # Prepare request content
    content = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])

    # Define response schema
    schema = genai.types.Schema(
        type=genai.types.Type.OBJECT,
        required=["is_continuation", "second_table_header"],
        properties={
            "is_continuation": genai.types.Schema(type=genai.types.Type.BOOLEAN),
            "second_table_header": genai.types.Schema(
                type=genai.types.Type.ARRAY,
                items=genai.types.Schema(type=genai.types.Type.STRING),
            ),
        },
    )

    # Configure generation parameters
    config = types.GenerateContentConfig(
        temperature=0, response_mime_type="application/json", response_schema=schema
    )

    # Get input token count
    input_tokens = client.models.count_tokens(model=MODEL, contents=[content])

    # Generate response
    response = client.models.generate_content(
        model=MODEL, contents=[content], config=config
    )

    # Process response
    result = handle_response(response.text)
    result.update(
        {
            "input_tokens": input_tokens,
            "output_tokens": client.models.count_tokens(
                model=MODEL, contents=response.text
            ),
        }
    )

    return result
