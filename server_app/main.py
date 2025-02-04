from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from typing import Annotated
from pydantic import BaseModel
import requests
import os
import json
from pdf_loader import process_pdf_folder, process_ocr_data, create_json_from_data
from chunk_embedding import FAISSRetrieverWithCohere, initialize_retriever
from rag import create_rag_chain, get_rag_response
from langchain_upstage import UpstageEmbeddings
import faiss
from dotenv import load_dotenv

PDF_FOLDER = "./data/pdf_files"
OCR_FOLDER = "./data/ocr_files"
OUTPUT_JSON = "./data/pdf_load_data.json"
FAISS_INDEX = "./data/faiss_index.bin"
UPSTAGE_API_KEY = "up_G8VMTIIVsK8F1sNmcdacfrQacgvgl"
OPENAI_API_KEY = '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'
COHERE_API_KEY = "vMK0W1WSLCNDo1zZVR2bOmXYzKqQJ0YhTd4MGFsE"

app = FastAPI()
path = os.getcwd()

def check_faiss_index(index_path):
    """FAISS 인덱스 파일의 존재 여부를 확인하고 로드"""
    print(f"FAISS 인덱스 파일 경로: {index_path}")
    if os.path.exists(index_path):
        print("FAISS 인덱스 파일이 존재합니다.")
        index = faiss.read_index(index_path)
        print(f"FAISS 인덱스 로드 완료. 벡터 개수: {index.ntotal}")
        return index
    else:
        print("Error: FAISS 인덱스 파일이 존재하지 않습니다.")
        return None

# PDF 및 OCR 데이터 처리
pdf_data = process_pdf_folder(PDF_FOLDER)
ocr_data = process_ocr_data(OCR_FOLDER)
create_json_from_data(pdf_data, ocr_data, OUTPUT_JSON)
print(f"JSON 파일 저장 완료: {OUTPUT_JSON}")

# JSON 데이터 로드 텍스트 청크 생성
with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
    combined_data = json.load(f)

# 텍스트 청크 생성
chunks = []
# for content in combined_data["pdf_data"].values():
#     chunks.extend(content.split(". "))  

# print(f"전체 청크 수: {len(chunks)}")
for content in combined_data["pdf_data"].values():
    chunks.extend([chunk.strip() for chunk in content.split(".") if chunk.strip()])

print(f"전체 청크 수: {len(chunks)}")


# FAISS 인덱스 로드
# FAISS 인덱스 확인 및 로드
index = check_faiss_index(FAISS_INDEX)

# 임베딩 초기화
embeddings = UpstageEmbeddings(upstage_api_key=UPSTAGE_API_KEY, model="solar-embedding-1-large-query")

# FAISSRetrieverWithCohere 초기화
retriever = initialize_retriever(FAISS_INDEX, chunks, UPSTAGE_API_KEY)
print("Retriever 초기화 완료")

# RAG 체인 
rag_chain = create_rag_chain(chunks, retriever, OPENAI_API_KEY)
print("RAG 체인 생성 완료")

class ChatRequest(BaseModel):
    question: str

@app.get('/api/getAnswer')
async def getAnswer(req: Annotated[ChatRequest, Query()]):
   response = {'answer': get_rag_response(rag_chain, req.question)}
   return JSONResponse(content=response)