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
    model = "gemini-2.0-flash"
    client = genai.Client(
        api_key=GEMINI_API_KEY,
    )
    system_prompt = """You are an AI assistant specialized in analyzing and grouping data tables for vertical concatenation. Your primary goal is to identify sets of tables that can be meaningfully combined based on their context, structure, and headers. You will be provided with information for multiple tables, presented sequentially.

**Input Format:**

Each table entry provides several pieces of information:
1.  `Context before table` (CBT): (Optional) Text preceding the table, often indicating its subject matter, title, or the section it belongs to. This context is **critically important** for grouping.
2.  `Table index`: A numerical identifier (e.g., Table 0).
3.  `Explicit header definition line` (EHDL): (Optional) A line formatted as `ColumnName1=ColumnName2=ColumnName3...` that explicitly defines the column names. This line, if present, appears immediately after the `Context before table` (if any) and before any `Sample data` rows. If multiple lines formatted this way appear consecutively in this position (i.e., between CBT and the first actual data row), the **last such line immediately preceding the first data row is considered the EHDL.**
4.  `Sample data`: One or more lines of table content. Values in data lines are separated by the `=` character.
5.  `Number of columns` (NOC): The count of columns in the table.
6.  `Column types`: A list of data types for the columns.

**Your Tasks (leading to a structured JSON output):**

**First: Identify Explicit Headers**
For each table, determine if it possesses an `Explicit header definition line (EHDL)` as defined in the "Input Format" section.
* Carefully check the line(s) between the CBT and the `Sample data`.
* The first actual data row is NOT the EHDL, even if it contains values that might seem like headers, unless it is specifically the EHDL identified as per the rule above.
* Compile a list of table indexes that have a true EHDL. This information will be used for the `has_headers` field in the output.

**Second: Identify Concatable Table Groups**
Your main task is to group tables that can be vertically concatenated. This process must result in every table index being assigned to a group in the `concatable_tables` output, even if a group contains only a single table due to its unique characteristics. Apply the following criteria hierarchically:

1.  **Primary Separator - `Context before table` (CBT):** This is the **most important criterion for separating groups.**
    * When a table is encountered, compare its CBT (if present) with the CBT of the active or immediately preceding table/group.
    * A new or significantly different CBT (e.g., a new title, a change from "Anime" to "Animation", or "Film" to "Video games") **must initiate a new table group.** This rule takes precedence over structural similarities. Tables with no CBT are generally considered continuations of the current CBT group, provided other criteria (NOC, Headers) match.

2.  **Structural Compatibility (within a CBT-defined group):** Once tables are tentatively associated by CBT (or lack of new CBT), they must also be structurally compatible to belong to the same concatenation group:
    * **Number of Columns (NOC):** All tables within a single concatenation group **must** have the exact same `Number of columns`. A change in NOC, even if the CBT implies continuation, forces a new group.
    * **Consistent Header Structure:**
        * **Tables with an EHDL:** Parse the column names from their EHDL. For tables to be in the same group, their parsed column names (and their order) must be **strictly identical after applying any specified column-name-specific normalizations (like the 'Source' rule detailed below).**
            * A difference in any column name not covered by a specific normalization rule (e.g., "Crew role, notes" vs. "Notes") makes the headers incompatible, forcing a new group.
            * **Normalization Rule (Specific Cases):** The only current normalization is for 'Source' columns: header names like `Source[13]`, `Source[XYZ]`, `Source[13][35]` etc., should be treated as a canonical "Source" when comparing headers for this specific column name. This rule applies *only* to columns named 'Source' or its direct variants. It does *not* make entire header sets compatible if other column names differ (e.g., `['Year', 'Title', 'Role', 'Crew role, notes', 'Source']` is NOT compatible with `['Year', 'Title', 'Role', 'Notes', 'Source']` because the fourth column names differ, and this difference is not addressed by the 'Source' normalization).
        * **Tables without an EHDL:** If a table lacks an EHDL but its CBT and NOC align with an active group, it can be part of that group only if its implicit structure is consistent with the group's established headers. It inherits these headers. The group's headers are typically set by the first table with an EHDL in that group, or are consistently inferred if no table in the group has an EHDL (see Task Third).

**Third: Determine Headers and "Guessed" Status for All Tables**
For every table, provide its definitive column headers and whether these headers were pre-existing or inferred. This will inform the `headers_info` field in the output.

* **Tables with an EHDL:**
    * `headers`: Use the exact column names parsed from its EHDL (applying the 'Source' normalization only if that column name appears and is part of the EHDL).
    * `is_header_guessed`: `false`.
* **Tables without an EHDL:**
    * `headers`:
        * If it's part of a group where headers are established by another table's EHDL (typically the first such table in that group), it inherits those (potentially normalized) headers.
        * If it's part of a group where **no table possesses an EHDL** (e.g., the group starts due to a CBT change and the first table in this new group lacks an EHDL, and subsequent tables also lack EHDLs but match on CBT and NOC), then all tables in this group will use consistently applied generic headers like `["Column 1", "Column 2", ...]` based on the group's NOC.
        * If it forms a new group by itself (e.g., due to a NOC change not aligning it with any EHDL-defined group, or it's a standalone table with unique CBT and no EHDL), and no EHDL defines headers for this new group, use generic headers as above based on its NOC.
    * `is_header_guessed`: `true` (because headers are inherited or generic).

**Finally: Consolidate Header Information Output**
For the `headers_info` part of the JSON output: if multiple table indexes (particularly those within the same `concatable_tables` group) share the exact same list of `headers` AND the same `is_header_guessed` status, consolidate them into a single `headers_info` entry. This entry's `table_index` field will be an array listing all such table indexes. Tables not fitting this consolidation pattern will have their own `headers_info` entries.

**Critical Guidelines for JSON Output Generation:**
When generating the final JSON output, pay strict attention to the following to ensure its validity:
* **String Formatting:** All string values (e.g., column names in the `headers` list) must be correctly formatted.
    * Enclose all strings in double quotes (`"`).
    * Inside any string value, special characters must be escaped:
        * A literal double quote (`"`) must be written as `\"`.
        * A literal backslash (`\\`) must be written as `\\`.
        * A newline character must be written as `\n`.
        * A tab character must be written as `\t`.
        * A carriage return character must be written as `\r`.
        * A backspace character must be written as `\b`.
        * A form feed character must be written as `\f`.
    * Ensure every string opened with a double quote is properly terminated with a closing double quote. Adhering to this will prevent `Unterminated string` errors and ensure the JSON is parsable.

Below is the table data:\n\n"""
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
