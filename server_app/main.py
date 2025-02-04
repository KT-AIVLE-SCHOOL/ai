from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from typing import Annotated
from pydantic import BaseModel
import requests
import os
import json
import os
import pickle
from chunk_embedding import initialize_retriever
from rag import create_rag_chain, get_rag_response
from dotenv import load_dotenv
import faiss

CHUNKS = "./data/all_chunks.pkl"
FAISS_INDEX = "./data/faiss_index.bin"
UPSTAGE_API_KEY = os.getenv('UPSTAGE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')

app = FastAPI()
path = os.getcwd()

def check_faiss_index(index_path):
    """FAISS 인덱스 파일의 존재 여부를 확인하고 로드"""
    # print(f"FAISS 인덱스 파일 경로: {index_path}")
    if os.path.exists(index_path):
        # print("FAISS 인덱스 파일이 존재합니다.")
        index = faiss.read_index(index_path)
        # print(f"FAISS 인덱스 로드 완료. 벡터 개수: {index.ntotal}")
        return index
    else:
        print("Error: FAISS 인덱스 파일이 존재하지 않습니다.")
        return None

with open(CHUNKS, "rb") as f:
    chunks = pickle.load(f)
# print(f"전체 청크 개수: {len(chunks)}")

# FAISS 인덱스 로드
# FAISS 인덱스 확인 및 로드
index = check_faiss_index(FAISS_INDEX)

# FAISSRetrieverWithCohere 초기화
retriever = initialize_retriever(FAISS_INDEX, chunks, UPSTAGE_API_KEY)
# print("Retriever 초기화 완료")

# RAG 체인 
rag_chain = create_rag_chain(retriever, OPENAI_API_KEY)
# print("RAG 체인 생성 완료")

class ChatRequest(BaseModel):
    question: str

@app.get('/api/getAnswer')
async def getAnswer(req: Annotated[ChatRequest, Query()]):
   response = {'answer': get_rag_response(rag_chain, req.question)}
   return JSONResponse(content=response)