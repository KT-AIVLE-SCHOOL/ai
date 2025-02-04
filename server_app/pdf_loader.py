import os
import re
import json
import pymupdf as fitz
# import fitz

# PDF 데이터 전처리
def clean_text(text):
    text = re.sub(r'[.]{2,}', "", text)
    text = re.sub(r'[-=]+', '', text)
    text = re.sub(r'[hH]{3,}', '', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r"(\b\w+\b)(\s+\1)+", r"\1", text)
    text = re.sub(r'[^가-힣\s0-9.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\.\.\.', '', text)
    text = re.sub(r'\.\.\d+\.\n', '', text)
    return text

# ocr 데이터 전처리
def clean_ocr_text(text):
    text = re.sub(r'(\b\S*[요다])(\s|$)', r'\1.', text)
    text = re.sub(r'[!]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^가-힣\s0-9.,!?]', '', text)
    return text

# PDF 파일에서 텍스트 추출
def extract_text_from_pdf(pdf_path):
    text = ""

    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            page_text = page.get_text()
            if page_text.strip():
                text += clean_text(page_text) + "\n"

    return text.strip()

# OCR 텍스트 파일 처리
def process_ocr_data(folder_path):
    if not os.path.exists(folder_path):
        print(f"OCR 폴더 경로가 존재하지 않습니다: {folder_path}")
        return {}

    ocr_data = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):  # OCR 파일이 .txt 형식이라고 가정
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.readlines()
            processed_lines = [clean_ocr_text(line) for line in raw_text if line.strip()]
            ocr_data[file_name] = "\n".join(processed_lines)
            print(f"OCR 파일 처리 완료: {file_name}, 라인 수: {len(processed_lines)}")

    return ocr_data


# PDF 폴더 내 모든 파일 처리
def process_pdf_folder(pdf_folder):
    pdf_data = {}

    if not os.path.exists(pdf_folder):
        print(f"PDF 폴더 경로가 존재하지 않습니다: {pdf_folder}")
        return pdf_data

    for file_name in os.listdir(pdf_folder):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, file_name)
            print(f"Processing file: {file_name}...")

            # PDF 텍스트 추출
            pdf_text = extract_text_from_pdf(pdf_path)
            pdf_data[file_name] = pdf_text
            print(f"Completed: {file_name}")

    return pdf_data

# JSON 파일 생성
def create_json_from_data(pdf_data, ocr_data, output_path):
    combined_data = {"pdf_data": pdf_data, "ocr_data": ocr_data}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)
    print(f"JSON 파일 저장 완료: {output_path}")