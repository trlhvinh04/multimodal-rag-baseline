import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd
from check_using_api import generate
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import TableItem, TextItem
from docling_core.types.doc.labels import DocItemLabel


class PDFConverter:
    def __init__(
        self,
        do_table_structure: bool = True,
        do_ocr: bool = False,
        do_cell_matching: bool = False,
        mode: TableFormerMode = TableFormerMode.ACCURATE,
        generate_page_images: bool = False,
        generate_picture_images: bool = False,
        images_scale: float = 1.0,
        # ocr_options=EasyOcrOptions(force_full_page_ocr=True, lang=["en"]),  # Use EasyOCR for OCR
        # ocr_options=TesseractOcrOptions(force_full_page_ocr=True, lang=["eng"]),  # Uncomment to use Tesseract for OCR
        # ocr_options = OcrMacOptions(force_full_page_ocr=True, lang=['en-US']),
    ):
        pipeline_options = PdfPipelineOptions(
            do_table_structure=do_table_structure,  # Enable table structure detection
            do_ocr=do_ocr,  # Enable OCR
            # full page ocr and language selection
            # ocr_options=EasyOcrOptions(force_full_page_ocr=True, lang=["en"]),  # Use EasyOCR for OCR
            # ocr_options=TesseractOcrOptions(force_full_page_ocr=True, lang=["eng"]),  # Uncomment to use Tesseract for OCR
            # ocr_options = OcrMacOptions(force_full_page_ocr=True, lang=['en-US']),
            table_structure_options=dict(
                # Use text cells predicted from table structure model
                do_cell_matching=do_cell_matching,
                mode=mode,  # Use more accurate Tableformer model
            ),
            generate_page_images=generate_page_images,  # Enable page image generation
            # Enable picture image generation
            generate_picture_images=generate_picture_images,
            # Set image resolution scale (scale=1 corresponds to a standard 72 DPI image)
            images_scale=images_scale,
        )
        format_options = {
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
        self.converter = DocumentConverter(format_options=format_options)

    def convert(self, sources: Union[str, Sequence[str]]) -> Dict[str, Any]:
        """
        Convert PDF files to Docling Document objects.

        Args:
            sources: A single PDF file path or a list of PDF file paths.

        Returns:
            A dictionary mapping source file paths to their corresponding Docling Document objects.
        """
        if isinstance(sources, str):
            sources = [sources]
        start_time = time.time()
        conversions = {
            source: self.converter.convert(source).document for source in sources
        }
        end_time = time.time()
        logging.info(f"Time taken: {end_time - start_time:.4f}")
        return conversions


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess DataFrame to handle common header issues:
    - If column names are numbers, use first row as header
    - If first data row is numbers (0-based or 1-based), remove it

    Args:
        df: A pandas DataFrame to preprocess.

    Returns:
        A preprocessed pandas DataFrame.
    """
    if df.empty:
        return df

    df_processed = df.copy()

    # Handle case where column names are numbers
    if all(str(col).isdigit() for col in df_processed.columns):
        if not df_processed.empty and df_processed.shape[0] > 0:
            new_cols = df_processed.iloc[0].astype(str).values
            df_processed = df_processed.iloc[1:].reset_index(drop=True)
            df_processed.columns = new_cols

    # Handle case where first data row is a sequence of numbers
    elif not df_processed.empty and df_processed.shape[0] > 0:
        first_row_values = df_processed.iloc[0]
        try:
            numeric_first_row = pd.to_numeric(first_row_values)
            is_0_based_sequence = all(
                numeric_first_row.iloc[i] == i for i in range(len(numeric_first_row))
            )
            is_1_based_sequence = all(
                numeric_first_row.iloc[i] == i + 1
                for i in range(len(numeric_first_row))
            )

            if is_0_based_sequence or is_1_based_sequence:
                df_processed = df_processed.iloc[1:].reset_index(drop=True)
        except (ValueError, TypeError):
            pass

    return df_processed


def solve_non_header_table(df: pd.DataFrame, target_headers: List[str]) -> pd.DataFrame:
    """
    Convert headerless DataFrame by moving current column names to first data row,
    then assign target_headers

    Args:
        df: A pandas DataFrame to preprocess.
        target_headers: A list of target headers to assign to the DataFrame.

    Returns:
        A preprocessed pandas DataFrame.
    """
    if not isinstance(target_headers, list):
        logging.warning(
            "Warning: target_headers is not a list. Returning original DataFrame."
        )
        return df.copy()

    df_copy = df.copy()

    first_row_data_values = []
    for col_original_name in df_copy.columns:
        if isinstance(col_original_name, str) and col_original_name.startswith(
            "Unnamed:"
        ):
            first_row_data_values.append(np.nan)
        else:
            first_row_data_values.append(str(col_original_name))

    if len(first_row_data_values) != len(target_headers):
        if len(first_row_data_values) > len(target_headers):
            first_row_data_values = first_row_data_values[: len(
                target_headers)]
        else:
            first_row_data_values.extend(
                [np.nan] * (len(target_headers) - len(first_row_data_values))
            )

    new_first_row_df = pd.DataFrame(
        [first_row_data_values], columns=target_headers)

    try:
        df_copy.columns = target_headers
    except ValueError as e:
        logging.error(
            f"Critical error: Cannot assign target_headers. "
            f"DataFrame columns ({len(df_copy.columns)}) don't match target_headers length ({len(target_headers)}). Error: {e}"
        )
        return new_first_row_df

    result_df = pd.concat([new_first_row_df, df_copy], ignore_index=True)

    return result_df.reset_index(drop=True)


def get_column_types(df: pd.DataFrame) -> List[str]:
    """
    Determine data types for each column

    Args:
        df: A pandas DataFrame to get column types from.

    Returns:
        A list of data types for each column.
    """
    types = []
    if df.columns.empty:
        return types

    for col_name in df.columns:
        current_column_object = df[col_name]

        is_truly_all_na = False
        effective_dtype = None

        if isinstance(current_column_object, pd.DataFrame):
            is_truly_all_na = pd.isna(current_column_object).all().all()
            if not is_truly_all_na:
                effective_dtype = current_column_object.iloc[:, 0].dtype
        elif isinstance(current_column_object, pd.Series):
            is_truly_all_na = pd.isna(current_column_object).all()
            if not is_truly_all_na:
                effective_dtype = current_column_object.dtype
        else:
            types.append("error_unexpected_type")
            continue

        if is_truly_all_na:
            types.append("unknown")
        else:
            if effective_dtype is not None:
                types.append(
                    "numeric"
                    if pd.api.types.is_numeric_dtype(effective_dtype)
                    else str(effective_dtype)
                )
            else:
                types.append("unknown_error_in_logic")
    return types


def get_input_df(
    df: pd.DataFrame, n_rows: int = 10, sep: str = "=", max_tokens: int = 15
) -> str:
    """
    Get input DataFrame with sampling and formatting for LLM processing

    Args:
        df: A pandas DataFrame to get input DataFrame from.
        n_rows: The number of rows to sample from the DataFrame.
        sep: The separator to use for the CSV string.
        max_tokens: The maximum number of tokens to use for the CSV string.

    Returns:
        A string representation of the DataFrame.
    """
    if df.empty:
        return ""

    df_copy = df.copy()
    df_copy = solve_non_header_table(
        df_copy, [str(x) for x in range(len(df_copy.columns))]
    )

    column_data_types = get_column_types(df_copy)

    for dtype, col_name in zip(column_data_types, df_copy.columns):
        if dtype == "object":
            if isinstance(df_copy[col_name], pd.Series):
                current_series = df_copy[col_name]
                try:
                    df_copy.loc[:, col_name] = current_series.astype(str).apply(
                        lambda x: " ".join(x.split()[:max_tokens])
                    )
                except AttributeError:
                    df_copy.loc[:, col_name] = current_series.apply(
                        lambda x: " ".join(str(x).split()[:max_tokens])
                    )
                pass

    num_original_rows = len(df_copy)
    sampled_df_parts = []

    if num_original_rows <= n_rows:
        sampled_df_parts.append(df_copy)
    else:
        n_head = (n_rows + 1) // 2
        n_tail = n_rows - n_head

        sampled_df_parts.append(df_copy.head(n_head))
        if n_tail > 0:
            sampled_df_parts.append(df_copy.tail(n_tail))

    if not sampled_df_parts:
        return ""

    final_df_to_sample = pd.concat(sampled_df_parts)
    final_df_to_sample = final_df_to_sample.drop_duplicates().reset_index(drop=True)

    return final_df_to_sample.to_csv(index=False, header=False, sep=sep)


def _get_item_from_cref(docling_document: Any, cref: str) -> Optional[Union[TextItem, TableItem]]:
    """
    Get an item from a Docling Document object using a cref.

    Args:
        docling_document: A Docling Document object.
        cref: A cref to get an item from.

    Returns:
        An item from the Docling Document object.
    """
    try:
        parts = cref.strip("#/").split("/")
        item_type = parts[0]  # 'texts' hoặc 'tables'
        item_index = int(parts[1])
        if item_type == "texts" and item_index < len(docling_document.texts):
            return docling_document.texts[item_index]
        elif item_type == "tables" and item_index < len(docling_document.tables):
            return docling_document.tables[item_index]
        else:
            logging.warning(
                f"Warning: Cref '{cref}' không hợp lệ hoặc không tìm thấy.")
            return None
    except (IndexError, ValueError, AttributeError) as e:
        logging.error(f"Error parsing cref '{cref}': {e}")
        return None


def extract_raw_tables_from_docling(
    docling_document: Any, n_tokens_previous: int = 20
) -> Tuple[List[pd.DataFrame], str]:
    """
    Extract raw tables from a Docling Document object.

    Args:
        docling_document: A Docling Document object.
        n_tokens_previous: The number of tokens to use for the preceding context.

    Returns:
        A tuple containing a list of DataFrames and a string representation of the prompt.
    """
    tables: List[pd.DataFrame] = []
    full_prompt = ""
    table_index = 0

    doc_items = (
        docling_document.body.children
        if hasattr(docling_document, "body")
        and hasattr(docling_document.body, "children")
        else []
    )

    for i, item_ref in enumerate(doc_items):
        # Kiểm tra xem item hiện tại có phải là một bảng không
        if item_ref.cref.startswith("#/tables/"):
            table_item = _get_item_from_cref(docling_document, item_ref.cref)

            # Chỉ xử lý nếu là TableItem hợp lệ và có label là TABLE
            if isinstance(table_item, TableItem) and table_item.label in [
                DocItemLabel.TABLE
            ]:
                # --- Lấy ngữ cảnh văn bản phía trước ---
                previous_text_snippet = ""
                if i > 0:  # Đảm bảo không phải là item đầu tiên
                    prev_item_ref = doc_items[i - 1]
                    # Kiểm tra xem item trước đó có phải là text không
                    if prev_item_ref.cref.startswith("#/texts/"):
                        prev_text_item = _get_item_from_cref(
                            docling_document, prev_item_ref.cref
                        )
                        if (
                            isinstance(prev_text_item, TextItem)
                            and hasattr(prev_text_item, "text")
                            and prev_text_item.text
                        ):
                            words = prev_text_item.text.split()
                            # Lấy n từ cuối cùng làm ngữ cảnh
                            previous_text_snippet = " ".join(
                                words[-n_tokens_previous:])

                # --- Xử lý DataFrame của bảng ---
                df = preprocess_df(table_item.export_to_dataframe())
                if df.empty:
                    continue
                df_string = get_input_df(df, n_rows=7)
                column_types = get_column_types(df=df)
                n_columns = len(df.columns)

                # --- Xây dựng prompt ---
                prompt_part = ""
                if previous_text_snippet:
                    prompt_part += f"Context before table:\n{previous_text_snippet}\n\n"
                prompt_part += f"Table {table_index}:\n{df_string}\nNumber of columns: {n_columns}\nColumn types: {column_types}\n\n"

                full_prompt += prompt_part
                tables.append(df)
                table_index += 1

    logging.info(
        f"{len(tables)} tables extracted and processed with preceding context."
    )
    return tables, full_prompt


def main_concatenation_logic(
    tables: List[pd.DataFrame], concat_json: Dict[str, Any]
) -> List[pd.DataFrame]:
    """
    Concatenate DataFrames based on groups in concat_json.
    Headers are applied based on 'headers_info' from concat_json.
    'is_header_guessed' from 'headers_info' determines if solve_non_header_table is used.

    Args:
        tables: A list of DataFrames to concatenate.
        concat_json: A dictionary containing the concatenation instructions.

    Returns:
        A list of concatenated DataFrames.
    """
    concatenated_results = []
    # Keep track of indices handled by 'concatable_tables'
    processed_indices_from_json = set()

    if not isinstance(tables, list) or not all(
        isinstance(t, pd.DataFrame) for t in tables
    ):
        logging.error("Error: 'tables' must be a list of DataFrames.")
        return []

    # 1. Pre-process headers_info into a usable map
    # header_details_map: key is table_index (int), value is {'headers': list[str], 'is_guessed': bool}
    header_details_map = {}
    if isinstance(concat_json, dict):
        headers_info_list = concat_json.get("headers_info", [])
        if isinstance(headers_info_list, list):
            for info_entry in headers_info_list:
                if (
                    isinstance(info_entry, dict)
                    and "table_index" in info_entry
                    and "headers" in info_entry
                ):
                    table_indices_for_this_info = info_entry["table_index"]
                    headers_for_this_info = info_entry["headers"]
                    # Default is_header_guessed to True if missing, common for LLM-inferred headers
                    is_guessed_for_this_info = info_entry.get(
                        "is_header_guessed", True)

                    # Ensure table_indices_for_this_info is a list for iteration
                    if not isinstance(table_indices_for_this_info, list):
                        if isinstance(table_indices_for_this_info, int):
                            # Convert single int to list
                            table_indices_for_this_info = [
                                table_indices_for_this_info]
                        else:
                            logging.warning(
                                f"Warning: 'table_index' in headers_info is not a list or int: {info_entry}. Skipping entry."
                            )
                            continue

                    if isinstance(headers_for_this_info, list):
                        for idx in table_indices_for_this_info:
                            if isinstance(idx, int):
                                # First-come, first-served for an index to prevent overriding by less specific entries
                                if idx not in header_details_map:
                                    header_details_map[idx] = {
                                        "headers": headers_for_this_info,
                                        "is_guessed": is_guessed_for_this_info,
                                    }
                            else:
                                logging.warning(
                                    f"Warning: Non-integer index '{idx}' found in headers_info.table_index list. Skipping this index."
                                )
                    else:
                        logging.warning(
                            f"Warning: 'headers' in headers_info entry is not a list: {info_entry}. Skipping entry."
                        )
                else:
                    logging.warning(
                        f"Warning: Invalid headers_info entry (e.g., not a dict, or missing keys): {info_entry}. Skipping entry."
                    )
        elif headers_info_list is not None:  # Present but not a list
            logging.warning(
                "Warning: 'headers_info' in concat_json is not a list. No explicit header information will be used from headers_info."
            )
    # If concat_json is not a dict, or headers_info is None/missing, map remains empty.

    if not header_details_map:
        logging.info(
            "Info: 'header_details_map' is empty after processing 'headers_info'. "
            "Tables will be processed using original columns or potentially skipped if in groups."
        )

    # 2. Process concatable_tables (groups)
    instructions_list = []
    if isinstance(concat_json, dict):
        instructions_list = concat_json.get("concatable_tables", [])
        if not isinstance(instructions_list, list):
            logging.warning(
                "Warning: 'concatable_tables' in concat_json is not a list. No concatenation will be performed based on it."
            )
            instructions_list = []
    elif concat_json is not None:  # concat_json exists but is not a dict
        logging.warning(
            "Warning: 'concat_json' is not a dictionary. No concatenation will be performed based on it."
        )

    for group_instruction in instructions_list:
        if (
            not isinstance(group_instruction, dict)
            or "table_index" not in group_instruction
        ):
            logging.warning(
                f"Warning: Invalid group instruction: {group_instruction}. Skipping this group."
            )
            continue

        table_indices_from_json_group = group_instruction.get(
            "table_index", [])
        if (
            not isinstance(table_indices_from_json_group, list)
            or not table_indices_from_json_group
        ):
            logging.warning(
                f"Warning: 'table_index' in group is invalid or empty: {group_instruction}. Skipping this group."
            )
            continue

        valid_indices_in_this_json_group = []
        for index in table_indices_from_json_group:
            if not isinstance(index, int):
                logging.warning(
                    f"Warning: Non-integer index '{index}' in group {table_indices_from_json_group}. Skipping this index."
                )
                continue
            if 0 <= index < len(tables):
                valid_indices_in_this_json_group.append(index)
            else:
                logging.warning(
                    f"Warning: Index {index} out of range for tables list. Skipping this index in group {table_indices_from_json_group}."
                )

        if not valid_indices_in_this_json_group:
            logging.warning(
                f"Warning: No valid DataFrames to process for index group {table_indices_from_json_group} after validation."
            )
            continue

        # Mark all valid indices from this group instruction as "processed" by a concat rule
        for idx in valid_indices_in_this_json_group:
            processed_indices_from_json.add(idx)

        # Determine the header standard for this group
        group_level_header_details = None
        reference_idx_for_group_header = -1

        # Try to find header_info for any table in the group to act as the standard
        for idx_in_group in valid_indices_in_this_json_group:
            if idx_in_group in header_details_map:
                group_level_header_details = header_details_map[idx_in_group]
                reference_idx_for_group_header = idx_in_group
                # logging.info(f"Info: Using header_info from table {reference_idx_for_group_header} as standard for group {valid_indices_in_this_json_group}.")
                break  # Found a standard for the group

        if group_level_header_details is None:
            logging.warning(
                f"Critical Warning: No 'headers_info' found for ANY table in group {valid_indices_in_this_json_group}. "
                "These tables will be added individually if not processed otherwise."
            )
            # Add original tables to results to prevent data loss, they won't be concatenated as a group
            for original_df_idx in valid_indices_in_this_json_group:
                # Check if already added (e.g. by another rule or if logic changes)
                # This check isn't strictly necessary with current flow but good for thought
                # if original_df_idx not in [item.attrs.get('original_index', -1) for item in concatenated_results]: # pseudo-code for check
                concatenated_results.append(
                    tables[original_df_idx].copy()
                )  # Add original
            continue  # Skip to next group instruction

        group_target_headers = group_level_header_details["headers"]
        is_group_header_guessed = group_level_header_details["is_guessed"]

        dfs_to_concat_in_group = []
        for original_df_idx in valid_indices_in_this_json_group:
            df = tables[original_df_idx].copy()

            # All tables in this group will use the same target headers and guessed status
            # derived from the group_level_header_details.

            if len(df.columns) != len(group_target_headers):
                logging.warning(
                    f"Warning: Table at index {original_df_idx} (in group) has {len(df.columns)} columns, "
                    f"but group standard header has {len(group_target_headers)} columns. Skipping this table from group."
                )
                # Optionally, add this skipped table as an independent one later if robust handling is needed
                # For now, it's skipped from this group.
                continue

            if is_group_header_guessed:
                df = solve_non_header_table(df, group_target_headers)
            else:
                df.columns = group_target_headers

            dfs_to_concat_in_group.append(df)

        if dfs_to_concat_in_group:
            if len(dfs_to_concat_in_group) == 1:  # Only one table qualified
                concatenated_results.append(dfs_to_concat_in_group[0])
            else:
                try:
                    final_df_group = pd.concat(
                        dfs_to_concat_in_group, ignore_index=True
                    )
                    concatenated_results.append(final_df_group)
                except Exception as e:
                    logging.error(
                        f"Error concatenating DataFrames in group {valid_indices_in_this_json_group}: {e}. Adding DFs individually."
                    )
                    # Add prepared DFs individually
                    concatenated_results.extend(dfs_to_concat_in_group)
        # Group had valid indices but no DFs made it (e.g., all column mismatches)
        elif valid_indices_in_this_json_group:
            logging.warning(
                f"Warning: No DataFrames in group {valid_indices_in_this_json_group} could be prepared for concatenation."
            )

    # 3. Handle independent tables (those not covered by any 'concatable_tables' group instruction)
    all_available_indices = set(range(len(tables)))
    independent_table_indices = sorted(
        list(all_available_indices - processed_indices_from_json)
    )

    for idx in independent_table_indices:
        # Keep a reference to original
        original_df_for_independent = tables[idx]
        df = original_df_for_independent.copy()

        header_details_for_independent_df = header_details_map.get(idx)

        if header_details_for_independent_df is None:
            logging.info(
                f"Info: No 'headers_info' found for independent table at index {idx}. Adding table as is."
            )
            # Add with original/current headers
            concatenated_results.append(df)
            continue

        target_headers = header_details_for_independent_df["headers"]
        is_guessed = header_details_for_independent_df["is_guessed"]

        if len(df.columns) != len(target_headers):
            logging.warning(
                f"Warning: Column count mismatch for independent table {idx}. Original has {len(df.columns)}, "
                f"target has {len(target_headers)}. Adding table as is (original form)."
            )
            concatenated_results.append(
                original_df_for_independent.copy()
            )  # Add original on error
            continue

        if is_guessed:
            df = solve_non_header_table(df, target_headers)
        else:
            df.columns = target_headers

        concatenated_results.append(df)

    return concatenated_results


def extract_tables_from_sources(sources: List[str]) -> List[pd.DataFrame]:
    """
    Extract connected tables from a list of PDF files.

    Args:
        sources: A list of PDF file paths.

    Returns:
        A list of concatenated DataFrames.
    """
    converter = PDFConverter()
    conversions = converter.convert(sources)
    for source, docling_document in conversions.items():
        tables, full_prompt = extract_raw_tables_from_docling(docling_document)
        concat_json = generate(full_prompt)
        concatenated_results = main_concatenation_logic(tables, concat_json)
    return concatenated_results


def extract_tables_from_docling(docling_document: Any) -> List[pd.DataFrame]:
    """
    Extract tables from a Docling Document object.

    Args:
        docling_document: A Docling Document object.

    Returns:
        A list of concatenated DataFrames.
    """
    tables, full_prompt = extract_raw_tables_from_docling(docling_document)
    concat_json = generate(full_prompt)
    concatenated_results = main_concatenation_logic(tables, concat_json)
    return concatenated_results


def display_tables(tables: List[pd.DataFrame], n_rows: int = 3) -> None:
    """
    Display tables in a Jupyter Notebook.

    Args:
        tables: A list of DataFrames to display.
        n_rows: The number of rows to display.
    """
    from IPython.display import display

    for idx, table in enumerate(tables):
        print(f"Table {idx}:")
        display(table.head(n_rows))
        print(f"Shape: {table.shape}")
        print(f"Columns: {table.columns.tolist()}")
        print("\n")
