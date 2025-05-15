import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict, Union, cast

import numpy as np
import pandas as pd
from check_using_api import generate
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import TableItem, TextItem
from docling_core.types.doc.labels import DocItemLabel


class ExtractedDataType(TypedDict):
    dataframes: List[pd.DataFrame]
    page_numbers: List[int]


class ProcessedTableEntry(TypedDict):
    dataframe: pd.DataFrame
    page_numbers: List[int]
    source: str


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
        format_options: Dict[InputFormat, PdfFormatOption] = {
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
        # Assuming self.converter.convert(source).document returns a Document object
        conversions: Dict[str, Any] = {
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
                except (
                    AttributeError
                ):  # Handle cases where elements might not be strings
                    df_copy.loc[:, col_name] = current_series.apply(
                        lambda x: " ".join(str(x).split()[:max_tokens])
                    )
                # pass was here, removed as it's not needed

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

    return final_df_to_sample.to_csv(index=False, header=False, sep=sep)


def _get_item_from_cref(
    docling_document: Any, cref: str
) -> Optional[Union[TextItem, TableItem]]:
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
        if (
            item_type == "texts"
            and hasattr(docling_document, "texts")
            and docling_document.texts
            and item_index < len(docling_document.texts)
        ):
            return docling_document.texts[item_index]
        elif (
            item_type == "tables"
            and hasattr(docling_document, "tables")
            and docling_document.tables
            and item_index < len(docling_document.tables)
        ):
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
) -> Tuple[ExtractedDataType, str]:
    """
    Extract raw tables from a Docling Document object.

    Args:
        docling_document: A Docling Document object.
        n_tokens_previous: The number of tokens to use for the preceding context.

    Returns:
        A tuple containing a dictionary with extracted data and a string representation of the prompt.
    """
    extracted_data: ExtractedDataType = {"dataframes": [], "page_numbers": []}
    full_prompt = ""
    table_index = 0

    doc_items = (
        docling_document.body.children
        if hasattr(docling_document, "body")
        and hasattr(docling_document.body, "children")
        and docling_document.body.children is not None  # Ensure children is not None
        else []
    )

    for i, item_ref in enumerate(doc_items):
        # Kiểm tra xem item hiện tại có phải là một bảng không
        if (
            hasattr(item_ref, "cref")
            and isinstance(item_ref.cref, str)
            and item_ref.cref.startswith("#/tables/")
        ):
            table_item = _get_item_from_cref(docling_document, item_ref.cref)

            # Chỉ xử lý nếu là TableItem hợp lệ và có label là TABLE
            if isinstance(table_item, TableItem) and table_item.label in [
                DocItemLabel.TABLE
            ]:
                # --- Lấy ra page_number ---
                page_number = -1  # default value of page
                # Di chuyển kiểm tra hasattr và table_item.prov lên trước
                if (
                    hasattr(table_item, "prov")
                    and isinstance(table_item.prov, list)
                    and table_item.prov
                ):
                    first_provenance_item = table_item.prov[0]
                    if hasattr(first_provenance_item, "page_no") and isinstance(
                        first_provenance_item.page_no, int
                    ):
                        page_number = (
                            first_provenance_item.page_no
                        )  # Giả sử page_no đã là 1-indexed
                    else:
                        logging.warning(
                            f"Attribute 'page_no' is missing, not an int, or None in the first ProvenanceItem for table {item_ref.cref}."
                        )
                else:
                    logging.warning(
                        f"Attribute 'prov' is missing, not a list, or empty for table {item_ref.cref}. Cannot determine page number."
                    )

                # --- Lấy ngữ cảnh văn bản phía trước ---
                previous_text_snippet = ""
                if i > 0:  # Đảm bảo không phải là item đầu tiên
                    prev_item_ref = doc_items[i - 1]
                    # Kiểm tra xem item trước đó có phải là text không
                    if (
                        hasattr(prev_item_ref, "cref")
                        and isinstance(prev_item_ref.cref, str)
                        and prev_item_ref.cref.startswith("#/texts/")
                    ):
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
                extracted_data["dataframes"].append(df)
                extracted_data["page_numbers"].append(page_number)
                table_index += 1

    logging.info(
        f"{len(extracted_data['dataframes'])} tables extracted and processed with preceding context."
    )
    return extracted_data, full_prompt


def _build_header_details_map(concat_json: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Xây dựng một map từ table_index đến thông tin header của nó từ concat_json.
    """
    header_details_map: Dict[int, Dict[str, Any]] = {}
    if not isinstance(concat_json, dict):
        logging.warning(
            "Cảnh báo: concat_json không phải là dictionary. Không thể xây dựng header_details_map."
        )
        return header_details_map

    headers_info_list = concat_json.get("headers_info", [])
    if not isinstance(headers_info_list, list):
        if headers_info_list is not None:
            logging.warning(
                "Cảnh báo: 'headers_info' trong concat_json không phải list."
            )
        return header_details_map

    for info_entry in headers_info_list:
        if not (
            isinstance(info_entry, dict)
            and "table_index" in info_entry
            and "headers" in info_entry
        ):
            logging.warning(
                f"Cảnh báo: Entry headers_info không hợp lệ: {info_entry}. Bỏ qua."
            )
            continue

        table_indices = info_entry["table_index"]
        headers = info_entry["headers"]
        is_guessed = info_entry.get("is_header_guessed", True)

        if not isinstance(table_indices, list):
            if isinstance(table_indices, int):
                table_indices = [table_indices]
            else:
                logging.warning(
                    f"Cảnh báo: 'table_index' trong headers_info không phải list hoặc int: {info_entry}. Bỏ qua."
                )
                continue

        if not isinstance(headers, list):
            logging.warning(
                f"Cảnh báo: 'headers' trong headers_info không phải list: {info_entry}. Bỏ qua."
            )
            continue

        for idx in table_indices:
            if not isinstance(idx, int):
                logging.warning(
                    f"Cảnh báo: Chỉ mục không phải int '{idx}' trong headers_info.table_index. Bỏ qua."
                )
                continue
            if idx not in header_details_map:
                header_details_map[idx] = {
                    "headers": headers, "is_guessed": is_guessed}

    if not header_details_map:
        logging.info(
            "Thông tin: 'header_details_map' rỗng sau khi xử lý 'headers_info'."
        )
    return header_details_map


def _apply_headers_to_table(
    df_original: pd.DataFrame,  # DataFrame gốc (chưa copy)
    target_headers: List[str],
    is_guessed: bool,
) -> Optional[pd.DataFrame]:  # Trả về DataFrame đã xử lý hoặc None nếu lỗi nghiêm trọng
    """
    Áp dụng header cho DataFrame. Trả về DataFrame đã xử lý,
    hoặc None nếu có lỗi không thể xử lý (ví dụ: số cột không khớp).
    """
    df_to_process = df_original.copy()  # Làm việc trên bản sao
    if len(df_to_process.columns) != len(target_headers):
        logging.warning(
            f"Cảnh báo: Số cột không khớp khi áp dụng header. Bảng có {len(df_to_process.columns)} cột, "
            f"header đích có {len(target_headers)} cột. Không thể xử lý header."
        )
        return None

    if is_guessed:
        return solve_non_header_table(df_to_process, target_headers)
    else:
        try:
            df_to_process.columns = target_headers
            return df_to_process
        except ValueError as e:
            logging.error(
                f"Lỗi nghiêm trọng khi gán target_headers trực tiếp: {e}. "
                f"(DataFrame columns: {len(df_to_process.columns)}, target_headers length: {len(target_headers)})"
            )
            return None


def _process_concatenation_group(
    group_indices: List[int],
    all_dataframes: List[pd.DataFrame],
    all_page_numbers: List[int],
    header_details_map: Dict[int, Dict[str, Any]],
    source_name: str,  # Thêm source_name
) -> List[ProcessedTableEntry]:
    """
    Xử lý một nhóm các bảng được chỉ định để ghép nối.
    Trả về một danh sách các ProcessedTableEntry.
    """
    group_results: List[ProcessedTableEntry] = []

    group_level_header_details: Optional[Dict[str, Any]] = None
    for idx_in_group in group_indices:
        if idx_in_group in header_details_map:
            group_level_header_details = header_details_map[idx_in_group]
            break

    if group_level_header_details is None:
        logging.warning(
            f"Cảnh báo quan trọng: Không tìm thấy 'headers_info' cho group {group_indices}. Thêm bảng gốc riêng lẻ."
        )
        for original_df_idx in group_indices:
            page_num = (
                all_page_numbers[original_df_idx]
                if 0 <= original_df_idx < len(all_page_numbers)
                else -1
            )
            group_results.append(
                {
                    "dataframe": all_dataframes[original_df_idx].copy(),
                    "page_numbers": [page_num] if page_num != -1 else [-1],
                    "source": source_name,
                }
            )
        return group_results

    group_target_headers: List[str] = group_level_header_details["headers"]
    is_group_header_guessed: bool = group_level_header_details["is_guessed"]

    # Lưu trữ (DataFrame đã xử lý, danh sách trang gốc của chính DF đó)
    processed_dfs_with_their_pages: List[Tuple[pd.DataFrame, List[int]]] = []

    for original_df_idx in group_indices:
        current_df_original = all_dataframes[original_df_idx]
        page_num_list_for_df: List[int] = [-1]
        if 0 <= original_df_idx < len(all_page_numbers):
            page_val = all_page_numbers[original_df_idx]
            page_num_list_for_df = [page_val] if page_val != -1 else [-1]

        processed_df = _apply_headers_to_table(
            current_df_original,  # _apply_headers_to_table sẽ tự copy
            group_target_headers,
            is_group_header_guessed,
        )

        if processed_df is None:
            logging.warning(
                f"Bảng tại index {original_df_idx} trong group {group_indices} không thể xử lý header. Thêm bảng gốc."
            )
            group_results.append(
                {  # Thêm trực tiếp vào group_results nếu không xử lý được cho nhóm
                    "dataframe": current_df_original.copy(),
                    "page_numbers": page_num_list_for_df,
                    "source": source_name,
                }
            )
        else:
            processed_dfs_with_their_pages.append(
                (processed_df, page_num_list_for_df))

    if not processed_dfs_with_their_pages:
        # group_results có thể đã chứa các bảng gốc nếu tất cả lỗi
        if not group_results and group_indices:
            logging.warning(
                f"Cảnh báo: Không có DataFrame nào trong group {group_indices} được chuẩn bị để ghép (sau khi xử lý header)."
            )
        return group_results

    if len(processed_dfs_with_their_pages) == 1:
        df_single, pages_single = processed_dfs_with_their_pages[0]
        group_results.append(
            {
                "dataframe": df_single,
                "page_numbers": pages_single,
                "source": source_name,
            }
        )
    else:
        dataframes_to_pd_concat = [item[0]
                                   for item in processed_dfs_with_their_pages]
        combined_pages_for_group: List[int] = sorted(
            list(
                set(
                    p
                    for item in processed_dfs_with_their_pages
                    for p in item[1]
                    if p != -1
                )
            )
        )
        if not combined_pages_for_group and any(
            p == -1
            for item in processed_dfs_with_their_pages
            for p_list in item[1]
            for p in p_list
        ):
            combined_pages_for_group = [-1]

        try:
            final_df_group = pd.concat(
                dataframes_to_pd_concat, ignore_index=True)
            group_results.append(
                {
                    "dataframe": final_df_group,
                    "page_numbers": combined_pages_for_group,
                    "source": source_name,
                }
            )
        except Exception as e:
            logging.error(
                f"Lỗi khi ghép DataFrames trong group {group_indices}: {e}. Thêm các DF đã xử lý header riêng lẻ."
            )
            for df_item, pages_item in processed_dfs_with_their_pages:
                group_results.append(
                    {
                        "dataframe": df_item,
                        "page_numbers": pages_item,
                        "source": source_name,
                    }
                )

    return group_results


def main_concatenation_logic(
    extracted_data: ExtractedDataType,
    concat_json: Dict[str, Any],
    source_name: str,  # Tham số mới
) -> List[ProcessedTableEntry]:
    """
    Ghép nối DataFrames dựa trên nhóm trong concat_json.
    Header được áp dụng dựa trên 'headers_info'.
    Trả về danh sách các dictionary, mỗi dictionary chứa 'dataframe',
    'page_numbers', và 'source'.
    """
    final_results: List[ProcessedTableEntry] = []
    processed_indices_from_json: set[int] = set()

    all_dataframes: List[pd.DataFrame] = extracted_data["dataframes"]
    all_page_numbers: List[int] = extracted_data["page_numbers"]

    if not all(isinstance(t, pd.DataFrame) for t in all_dataframes) or len(
        all_dataframes
    ) != len(all_page_numbers):
        logging.error(
            "Lỗi: 'extracted_data' không hợp lệ hoặc 'dataframes' và 'page_numbers' không khớp."
        )
        return []

    header_details_map = _build_header_details_map(concat_json)

    instructions_list: List[Dict[str, Any]] = []
    if isinstance(concat_json, dict):
        instructions_list = concat_json.get("concatable_tables", [])
        if not isinstance(instructions_list, list):
            logging.warning(
                "Cảnh báo: 'concatable_tables' trong concat_json không phải list."
            )
            instructions_list = []
    elif concat_json is not None:
        logging.warning("Cảnh báo: 'concat_json' không phải dictionary.")

    for group_instruction in instructions_list:
        if not (
            isinstance(group_instruction,
                       dict) and "table_index" in group_instruction
        ):
            logging.warning(
                f"Cảnh báo: Group instruction không hợp lệ: {group_instruction}. Bỏ qua."
            )
            continue

        table_indices_from_json_group = group_instruction.get(
            "table_index", [])
        if not (
            isinstance(table_indices_from_json_group, list)
            and table_indices_from_json_group
        ):
            logging.warning(
                f"Cảnh báo: 'table_index' trong group không hợp lệ hoặc rỗng: {group_instruction}. Bỏ qua."
            )
            continue

        valid_indices_in_this_json_group: List[int] = []
        for index in table_indices_from_json_group:
            if not isinstance(index, int):
                logging.warning(
                    f"Cảnh báo: Chỉ mục không phải int '{index}' trong group {table_indices_from_json_group}. Bỏ qua."
                )
                continue
            if 0 <= index < len(all_dataframes):
                valid_indices_in_this_json_group.append(index)
            else:
                logging.warning(
                    f"Cảnh báo: Chỉ mục {index} ngoài phạm vi. Bỏ qua trong group {table_indices_from_json_group}."
                )

        if not valid_indices_in_this_json_group:
            logging.warning(
                f"Cảnh báo: Không có DataFrame hợp lệ để xử lý cho group {table_indices_from_json_group}."
            )
            continue

        for idx in valid_indices_in_this_json_group:
            processed_indices_from_json.add(idx)

        group_processed_entries = _process_concatenation_group(
            valid_indices_in_this_json_group,
            all_dataframes,
            all_page_numbers,
            header_details_map,
            source_name,  # Truyền source_name vào hàm xử lý nhóm
        )
        final_results.extend(group_processed_entries)

    # Xử lý các bảng độc lập
    all_available_indices_set = set(range(len(all_dataframes)))
    independent_table_indices = sorted(
        list(all_available_indices_set - processed_indices_from_json)
    )

    for idx in independent_table_indices:
        original_df = all_dataframes[idx]

        page_num = all_page_numbers[idx] if 0 <= idx < len(
            all_page_numbers) else -1
        current_pages_list: List[int] = [page_num] if page_num != -1 else [-1]

        header_details = header_details_map.get(idx)
        final_df_for_independent_table: pd.DataFrame

        if header_details is None:
            logging.info(
                f"Thông tin: Không tìm thấy 'headers_info' cho bảng độc lập tại index {idx}. Sử dụng bảng gốc."
            )
            final_df_for_independent_table = original_df.copy()
        else:
            target_headers: List[str] = header_details["headers"]
            is_guessed: bool = header_details["is_guessed"]

            processed_df = _apply_headers_to_table(
                original_df, target_headers, is_guessed
            )  # original_df sẽ được copy bên trong

            if (
                processed_df is None
            ):  # Lỗi khi áp dụng header, _apply_headers_to_table trả về None
                logging.warning(
                    f"Không thể xử lý header cho bảng độc lập {idx}. Sử dụng bảng gốc."
                )
                final_df_for_independent_table = original_df.copy()
            else:
                final_df_for_independent_table = processed_df

        final_results.append(
            {
                "dataframe": final_df_for_independent_table,
                "page_numbers": current_pages_list,
                "source": source_name,
            }
        )

    return final_results


def extract_tables_from_sources(
    sources: List[str],
) -> List[ProcessedTableEntry]:  # Cập nhật type hint trả về
    """
    Trích xuất các bảng đã ghép nối từ danh sách các file PDF.
    """
    converter = PDFConverter()
    conversions = converter.convert(
        sources
    )  # conversions là Dict[str, Any] (Any là docling_document)

    # Khởi tạo để tích lũy kết quả
    all_final_results: List[ProcessedTableEntry] = []

    for source_path, docling_document in conversions.items():
        # Trích xuất tên file rút gọn từ source_path
        # Ví dụ: "/path/to/filename.pdf" -> "filename"
        base_name_with_ext = os.path.basename(source_path)
        source_name_short = os.path.splitext(base_name_with_ext)[0]

        logging.info(
            f"Đang xử lý nguồn: {source_name_short} (từ: {source_path})")

        extracted_data, full_prompt = extract_raw_tables_from_docling(
            docling_document)

        # Giả sử hàm generate của bạn trả về Dict[str, Any]
        # Bạn cần đảm bảo hàm generate(full_prompt) được định nghĩa và hoạt động
        try:
            concat_json: Dict[str, Any] = generate(full_prompt)  # Gọi API LLM
        except Exception as e:
            logging.error(
                f"Lỗi khi gọi API LLM cho source {source_name_short}: {e}")
            # Quyết định cách xử lý: bỏ qua source này, hoặc thử lại, hoặc thêm các bảng chưa xử lý
            # Ví dụ: thêm tất cả các bảng từ extracted_data mà không ghép nối
            for i, df in enumerate(extracted_data["dataframes"]):
                page_num = extracted_data["page_numbers"][i]
                all_final_results.append(
                    {
                        "dataframe": df.copy(),
                        "page_numbers": [page_num] if page_num != -1 else [-1],
                        "source": source_name_short,
                    }
                )
            continue  # Chuyển sang source tiếp theo

        results_for_current_source = main_concatenation_logic(
            extracted_data,
            concat_json,
            source_name_short,  # Truyền tên nguồn đã rút gọn
        )
        all_final_results.extend(
            results_for_current_source)  # Tích lũy kết quả

    return all_final_results


def extract_tables_from_docling(
    docling_document: Any,
) -> List[ProcessedTableEntry]:  # Cập nhật type hint
    """
    Trích xuất bảng từ một đối tượng Docling Document.
    """
    source_name_short = "unknown_source"  # Giá trị mặc định
    if (
        hasattr(docling_document, "origin")
        and docling_document.origin
        and hasattr(docling_document.origin, "filename")
        and docling_document.origin.filename
    ):
        base_name_with_ext = os.path.basename(docling_document.origin.filename)
        source_name_short = os.path.splitext(base_name_with_ext)[0]

    logging.info(f"Đang xử lý tài liệu Docling: {source_name_short}")

    extracted_data, full_prompt = extract_raw_tables_from_docling(
        docling_document)

    try:
        concat_json: Dict[str, Any] = generate(full_prompt)  # Gọi API LLM
    except Exception as e:
        logging.error(
            f"Lỗi khi gọi API LLM cho docling_document {source_name_short}: {e}"
        )
        # Xử lý tương tự như trong extract_tables_from_sources nếu generate lỗi
        fallback_results: List[ProcessedTableEntry] = []
        for i, df in enumerate(extracted_data["dataframes"]):
            page_num = extracted_data["page_numbers"][i]
            fallback_results.append(
                {
                    "dataframe": df.copy(),
                    "page_numbers": [page_num] if page_num != -1 else [-1],
                    "source": source_name_short,
                }
            )
        return fallback_results

    concatenated_results = main_concatenation_logic(
        extracted_data,
        concat_json,
        source_name_short,  # Truyền tên nguồn đã rút gọn
    )
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


def table_to_markdown(table: pd.DataFrame, name: str, page_numbers: List[int]) -> str:
    markdown_str = f"# {name}\n"
    if len(page_numbers) > 1:
        # cross-page table
        markdown_str += f"This is a cross-page table. It spans multiple pages. Page numbers: {page_numbers}\n"
    else:
        # single-page table
        markdown_str += (
            f"This is a single-page table. Page number: {page_numbers[0]}\n\n"
        )
    markdown_str += table.to_markdown(index=False)

    table_shape = table.shape
    markdown_str += f"\n\nShape: {table_shape}\n"

    return markdown_str


def get_table_content(
    processed_tables_with_pages: List[ProcessedTableEntry],
) -> Dict[str, List[Dict[str, Any]]]:
    results: Dict[str, List[Dict[str, Any]]] = {}

    sources_ = set(table["source"] for table in processed_tables_with_pages)

    for source in sources_:
        pdf_name = os.path.basename(source)
        file_name_part, file_extension_part = os.path.splitext(pdf_name)

        # lấy ra các processed_tables_with_pages_i mà có key 'source' = với file_name_part
        processed_tables_source = [
            table
            for table in processed_tables_with_pages
            if table["source"] == file_name_part
        ]

        # logging
        print(
            f"source {file_name_part} has {len(processed_tables_source)} tables")

        for idx, table in enumerate(processed_tables_source):
            table_df = table["dataframe"]
            page_numbers = table["page_numbers"]
            table_name = f"{file_name_part}_table_{idx}"
            markdown_str = table_to_markdown(
                table_df, table_name, page_numbers)

            result = {
                "table_content": markdown_str,
                "page_numbers": page_numbers,
                "source": file_name_part,
                "table_idx": idx,
            }
            if file_name_part not in results:
                results[
                    file_name_part
                ] = []
            results[file_name_part].append(result)
    return results
