import logging
import os
import time
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Set,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
import pandas as pd
import pymupdf
from check_using_api import generate

pd.set_option("future.no_silent_downcasting", True)


class ExtractedDataType(TypedDict):
    """Structure for extracted PDF data"""

    dataframes: List[pd.DataFrame]
    page_numbers: List[int]


def get_pdf_name(source: str) -> str:
    """Get PDF filename without extension"""
    pdf_name = os.path.basename(source)
    file_name_part, file_extension_part = os.path.splitext(pdf_name)
    return file_name_part


def preprocess_df(df):
    """Clean DataFrame by removing empty cols/rows and normalizing column names"""
    if df.empty:
        return df

    df_processed = df.copy()
    col_names_with_prefix = [
        col for col in df_processed.columns if str(col).startswith("Col")
    ]
    for col in col_names_with_prefix:
        if df_processed[col].isnull().all():
            df_processed.drop(col, axis=1, inplace=True)

    df_processed = df_processed.replace([None, ""], np.nan)
    df_processed = df_processed.dropna(how="all")
    df_processed = df_processed.fillna("")

    df_processed_columns = df_processed.columns
    for col in df_processed_columns:
        if str(col).startswith("Col"):
            df_processed.rename(columns={col: ""}, inplace=True)

    return df_processed


def get_limited_text_before_table(
    page: pymupdf.Page,
    current_table_bbox: pymupdf.Rect,
    all_table_bboxes_on_page: List[pymupdf.Rect],
    n_tokens: int,
    search_height_multiplier: float = 2.0,
    min_search_height: int = 30,
    main_content_max_x_ratio: float = 0.65 # Main content usually in left 65% of page width
) -> str:
    """Get limited text before table, avoiding irrelevant columns and focusing on nearest section header"""
    if isinstance(current_table_bbox, tuple):
        if len(current_table_bbox) == 4:
            try:
                current_table_bbox = pymupdf.Rect(current_table_bbox)
            except Exception: return ""
        else: return ""
    elif not isinstance(current_table_bbox, pymupdf.Rect): return ""

    if not current_table_bbox or current_table_bbox.is_empty: return ""

    page_width = page.rect.width
    try:
        table_height = current_table_bbox.height
        if table_height <= 0: table_height = 10
    except AttributeError: return ""

    max_search_height = min(min_search_height, 80)  # Max 80 pixels
    search_area_height = min(table_height * 0.5, max_search_height)
    
    search_rect_y0 = max(0, current_table_bbox.y0 - 1)
    search_rect_y1 = max(0, search_rect_y0 - search_area_height)

    if search_rect_y1 >= search_rect_y0:
        search_area_height = 25  # Only 25 pixels above table
        search_rect_y1 = max(0, search_rect_y0 - search_area_height)
        if search_rect_y1 >= search_rect_y0: 
            return ""

    search_rect = pymupdf.Rect(0, search_rect_y1, page_width, search_rect_y0)
    if search_rect.is_empty or search_rect.height <= 0: return ""

    blocks_in_search_area = page.get_text("blocks", clip=search_rect, sort=True)
    filtered_text_blocks = []

    for block_data in blocks_in_search_area:
        if block_data[6] != 0:  # Not a text block
            continue

        block_rect = pymupdf.Rect(block_data[0], block_data[1], block_data[2], block_data[3])
        block_text_content = block_data[4]

        is_part_of_a_table = False
        for table_bbox_on_pg in all_table_bboxes_on_page:
            test_table_rect = pymupdf.Rect(table_bbox_on_pg) if isinstance(table_bbox_on_pg, tuple) else table_bbox_on_pg
            if not test_table_rect.is_empty:
                if not (block_rect.irect & test_table_rect.irect).is_empty:
                    is_part_of_a_table = True
                    break
        if is_part_of_a_table: continue

        is_horizontally_relevant = False
        effective_table_x0 = current_table_bbox.x0 - (current_table_bbox.width * 0.1) # Allow left offset
        effective_table_x1 = current_table_bbox.x1 + (current_table_bbox.width * 0.1) # Allow right offset

        if max(block_rect.x0, effective_table_x0) < min(block_rect.x1, effective_table_x1):
            is_horizontally_relevant = True
        
        if is_horizontally_relevant and \
           current_table_bbox.x1 < page_width * main_content_max_x_ratio: # Table in main content
            if block_rect.x0 >= page_width * (main_content_max_x_ratio - 0.05): # Block in sidebar
                is_horizontally_relevant = False

        if is_horizontally_relevant:
            cleaned_block_text = block_text_content.replace("\n", " ").strip()
            if cleaned_block_text:
                distance_to_table = current_table_bbox.y0 - block_rect.y1  # Distance from block bottom to table top
                filtered_text_blocks.append({
                    'text': cleaned_block_text,
                    'distance_to_table': distance_to_table,
                    'y_position': block_rect.y0
                })

    if not filtered_text_blocks:
        return ""
    
    filtered_text_blocks.sort(key=lambda x: x['distance_to_table'])
    
    closest_block = filtered_text_blocks[0]
    text_above = closest_block['text']
    text_above = " ".join(text_above.split())

    if not text_above: return ""
    tokens = text_above.split()
    if len(tokens) > n_tokens: limited_text = " ".join(tokens[-n_tokens:])
    else: limited_text = " ".join(tokens)
    return limited_text

def get_limited_text_from_bottom_of_page(
    page: pymupdf.Page,
    all_table_bboxes_on_page: List[pymupdf.Rect], 
    n_tokens: int,
    search_height_ratio: float = 0.25,
    min_search_height_from_bottom: int = 50,
    main_content_max_x_ratio: float = 0.65
) -> str:
    """Get limited text from bottom of page, searching below last table or in fixed bottom area if no tables"""
    if n_tokens <= 0: return ""
    page_height = page.rect.height
    page_width = page.rect.width

    effective_search_rect_y1: float

    if all_table_bboxes_on_page:
        max_y1_of_tables = 0.0
        for bbox in all_table_bboxes_on_page:
            if bbox.y1 > max_y1_of_tables:
                max_y1_of_tables = bbox.y1
        effective_search_rect_y1 = max_y1_of_tables + 1.0 # +1 to avoid table overlap
    else:
        search_area_actual_height = max(page_height * search_height_ratio, min_search_height_from_bottom)
        effective_search_rect_y1 = max(0, page_height - search_area_actual_height)

    search_rect_y0_for_bottom_area = page_height

    if effective_search_rect_y1 >= search_rect_y0_for_bottom_area:
        return ""

    search_rect = pymupdf.Rect(0, effective_search_rect_y1, page_width, search_rect_y0_for_bottom_area)
    if search_rect.is_empty or search_rect.height <= 0:
        return ""

    blocks_in_search_area = page.get_text("blocks", clip=search_rect, sort=True)
    filtered_text_lines = []

    for block_data in blocks_in_search_area:
        if block_data[6] != 0: continue # Only text blocks
        block_rect = pymupdf.Rect(block_data[0], block_data[1], block_data[2], block_data[3])
        block_text_content = block_data[4]
        
        is_part_of_a_table = False
        for table_bbox_on_this_page in all_table_bboxes_on_page:
            test_table_rect = pymupdf.Rect(table_bbox_on_this_page) if isinstance(table_bbox_on_this_page, tuple) else table_bbox_on_this_page
            if not test_table_rect.is_empty:
                if not (block_rect.irect & test_table_rect.irect).is_empty:
                    is_part_of_a_table = True
                    break
        if is_part_of_a_table: continue

        if block_rect.x0 < page_width * main_content_max_x_ratio:
            cleaned_block_text = block_text_content.replace("\n", " ").strip()
            if cleaned_block_text:
                filtered_text_lines.append(cleaned_block_text)
            
    text_from_bottom = " ".join(filtered_text_lines)
    text_from_bottom = " ".join(text_from_bottom.split())

    if not text_from_bottom: return ""
    
    tokens = text_from_bottom.split()
    if len(tokens) > n_tokens:
        limited_text = " ".join(tokens[-n_tokens:])
    else:
        limited_text = " ".join(tokens)
        
    return limited_text


def extract_tables_and_contexts(
    doc: pymupdf.Document,
    n_tokens_context: int = 10,
    page_numbers_to_process: Union[int, List[int], str] = "all",
    search_height_multiplier: float = 2.5,
    min_search_height: int = 40,
    look_to_prev_page_threshold_y0: float = 50.0,
) -> Tuple[ExtractedDataType, List[str]]:
    """Extract tables and their contexts from PDF document"""
    all_dataframes: List[pd.DataFrame] = []
    all_contexts: List[str] = []
    all_df_page_numbers: List[int] = []
    default_extracted_data: ExtractedDataType = {"dataframes": [], "page_numbers": []}

    if not isinstance(doc, pymupdf.Document):
        logging.error("Error: 'doc' must be a pymupdf.Document object.")
        return default_extracted_data, []

    if isinstance(page_numbers_to_process, int):
        pages_to_process_indices = [page_numbers_to_process - 1]
    elif isinstance(page_numbers_to_process, list) and all(
        isinstance(pn, int) for pn in page_numbers_to_process
    ):
        pages_to_process_indices = sorted(
            list(set(p - 1 for p in page_numbers_to_process))
        )
    elif page_numbers_to_process == "all":
        pages_to_process_indices = list(range(len(doc)))
    else:
        logging.error(
            "Error: page_numbers_to_process must be int or list of ints."
        )
        return default_extracted_data, []

    for page_index in pages_to_process_indices:
        if not (0 <= page_index < len(doc)):
            logging.warning(
                f"Warning: Page number {page_index + 1} is invalid. Skipping."
            )
            continue

        page = doc[page_index]
        current_page_human_readable = page.number + 1
        page_table_finder = page.find_tables(strategy="lines_strict")
        all_found_table_bboxes_on_current_page: List[pymupdf.Rect] = []
        if page_table_finder.tables:
            all_found_table_bboxes_on_current_page = [
                pymupdf.Rect(tbl.bbox) for tbl in page_table_finder.tables if tbl.bbox
            ]

        if page_table_finder.tables:
            for table_obj in page_table_finder.tables:
                if not table_obj.bbox:
                    continue

                df_original = table_obj.to_pandas()
                df_original = preprocess_df(df_original)

                if not df_original.empty:
                    current_table_actual_bbox = pymupdf.Rect(table_obj.bbox)
                    context_text_current_page = get_limited_text_before_table(
                        page,
                        current_table_actual_bbox,
                        all_found_table_bboxes_on_current_page,
                        n_tokens_context,
                        search_height_multiplier=search_height_multiplier,
                        min_search_height=min_search_height,
                    ).strip()
                    combined_context = context_text_current_page
                    if (
                        not context_text_current_page
                        and current_table_actual_bbox.y0
                        < look_to_prev_page_threshold_y0
                        and page_index > 0
                    ):
                        prev_page_index = page_index - 1
                        prev_page = doc[prev_page_index]
                        prev_page_table_finder = prev_page.find_tables()
                        all_found_table_bboxes_on_prev_page: List[pymupdf.Rect] = []
                        if prev_page_table_finder.tables:
                            all_found_table_bboxes_on_prev_page = [
                                pymupdf.Rect(tbl.bbox)
                                for tbl in prev_page_table_finder.tables
                                if tbl.bbox
                            ]
                        context_from_prev_page_bottom = (
                            get_limited_text_from_bottom_of_page(
                                prev_page,
                                all_found_table_bboxes_on_prev_page,
                                n_tokens_context,
                            ).strip()
                        )
                        if context_from_prev_page_bottom:
                            combined_context = context_from_prev_page_bottom
                    all_contexts.append(combined_context)
                    all_dataframes.append(df_original)
                    all_df_page_numbers.append(current_page_human_readable)
    extracted_data_output: ExtractedDataType = {
        "dataframes": all_dataframes,
        "page_numbers": all_df_page_numbers,
    }
    if not (len(all_dataframes) == len(all_contexts) == len(all_df_page_numbers)):
        logging.error(
            f"Critical logic error: List lengths don't match! DFs: {len(all_dataframes)}, Contexts: {len(all_contexts)}, PageNums: {len(all_df_page_numbers)}"
        )
        return default_extracted_data, []
    return extracted_data_output, all_contexts


def get_column_types(df: pd.DataFrame) -> List[str]:
    """Get data types for each column in DataFrame"""
    types: List[str] = []
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
    """Get sample of DataFrame in markdown format"""
    if df.empty:
        return ""
    df_copy = df.copy()
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
    num_original_rows = len(df_copy)
    sampled_df_parts: List[pd.DataFrame] = []
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
    markdown_table = final_df_to_sample.to_markdown(index=False, tablefmt="pipe")
    return markdown_table


def build_llm_prompt(extracted_data: ExtractedDataType, contexts: List[str]) -> str:
    """Build prompt for LLM with table data and contexts"""
    full_prompt = ""
    dataframes = extracted_data["dataframes"]
    for i, df in enumerate(dataframes):
        if df.empty:
            continue
        df_string = get_input_df(df, n_rows=7)
        column_types = get_column_types(df=df)
        n_columns = len(df.columns)
        prompt_part = f"Table {i}:\n"
        if (
            i < len(contexts) and contexts[i]
        ):
            prompt_part += f"Context before table:\n{contexts[i]}\n\n"
        prompt_part += f"{df_string}\nNumber of columns: {n_columns}\n\n"
        full_prompt += prompt_part
    return full_prompt


class ProcessedTableEntry(TypedDict):
    """Structure for processed table data"""
    dataframe: pd.DataFrame
    page_numbers: List[int]
    source: str
    associated_contexts: List[str]


def solve_non_header_table(df: pd.DataFrame, target_headers: List[str]) -> pd.DataFrame:
    """Fix table without proper headers by adding target headers"""
    if not isinstance(target_headers, list):
        logging.warning(
            "Warning: target_headers is not a list. Returning original DataFrame."
        )
        return df.copy()
    df_copy = df.copy()
    first_row_data_values: List[Any] = []
    for col_original_name in df_copy.columns:
        if isinstance(col_original_name, str) and col_original_name.startswith(
            "Unnamed:"
        ):
            first_row_data_values.append(np.nan)
        else:
            first_row_data_values.append(str(col_original_name))
    if len(first_row_data_values) != len(target_headers):
        if len(first_row_data_values) > len(target_headers):
            first_row_data_values = first_row_data_values[: len(target_headers)]
        else:
            first_row_data_values.extend(
                [np.nan] * (len(target_headers) - len(first_row_data_values))
            )
    new_first_row_df = pd.DataFrame([first_row_data_values], columns=target_headers)
    num_target_cols = len(target_headers)
    current_data_cols = df_copy.shape[1]
    if num_target_cols == 0:
        if current_data_cols > 0:
            df_copy = pd.DataFrame(index=df_copy.index)
    else:
        if current_data_cols < num_target_cols:
            for i in range(num_target_cols - current_data_cols):
                df_copy[f"__temp_added_col_{i}"] = np.nan
        elif current_data_cols > num_target_cols:
            df_copy = df_copy.iloc[:, :num_target_cols]
    df_copy.columns = target_headers
    result_df = pd.concat([new_first_row_df, df_copy], ignore_index=True)
    return result_df.reset_index(drop=True)


def main_concatenation_logic(
    extracted_data: ExtractedDataType,
    contexts: List[str],
    concat_json: Dict[str, Any],
    source_name: str,
) -> List[ProcessedTableEntry]:
    """Main logic for concatenating tables based on LLM output"""
    all_original_dataframes = extracted_data["dataframes"]
    all_original_page_numbers = extracted_data["page_numbers"]

    if len(all_original_dataframes) != len(all_original_page_numbers):
        if len(all_original_dataframes) != len(contexts):
            logging.error(
                "DataFrame count, page_numbers count and/or contexts count in extracted_data don't match."
            )
            raise ValueError(
                "DataFrame count, page_numbers count and/or contexts count don't match."
            )

    processed_tables_result: List[ProcessedTableEntry] = []

