import os
import faiss
import numpy as np
import cohere
from langchain_upstage import UpstageEmbeddings
from langchain.schema import Document
from typing import List
from pydantic import Field
from langchain.schema import BaseRetriever
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv("chat_api_key.env")

# Cohere API Key 설정
cohere_api_key = os.getenv("cohere_api_key")
co = cohere.Client(cohere_api_key)

# FAISS 기반 + Cohere Reranking 적용한 Retriever
class FAISSRetrieverWithCohere(BaseRetriever):
    index: faiss.IndexIVFFlat = Field(...)
    embeddings: UpstageEmbeddings = Field(...)
    documents: List[str] = Field(...)

    def _get_relevant_documents(self, query: str, k=5) -> List[Document]:
        query_emb = np.array([self.embeddings.embed_query(query)]).astype("float32")
        distances, indices = self.index.search(query_emb, k * 2)  # 더 많은 후보 검색

        retrieved_docs = [Document(page_content=self.documents[idx]) for idx in indices[0] if idx != -1]

        # Cohere API를 사용하여 Reranking 수행
        reranked_docs = self.rerank_with_cohere(query, retrieved_docs)
       
        return reranked_docs[:k]  # Top-K 문서 반환

    def rerank_with_cohere(self, query: str, documents: List[Document]) -> List[Document]:
        response = co.rerank(
            model="rerank-multilingual-v3.0",
            query=query,
            documents=[doc.page_content for doc in documents],
            top_n=len(documents)
        )

        sorted_indices = [result.index for result in response.results]

        # 검색 결과 정렬
        reranked_docs = [documents[i] for i in sorted_indices]
        return reranked_docs

    async def _aget_relevant_documents(self, query: str, k=5) -> List[Document]:
        return self._get_relevant_documents(query, k)

# FAISS Index 불러오기 함수
def load_faiss_index(index_path: str):
    return faiss.read_index(index_path)

# FAISSRetrieverWithCohere 초기화 함수
def initialize_retriever(index_path: str, chunks: List[str], upstage_api_key: str):
    index = load_faiss_index(index_path)
    embeddings = UpstageEmbeddings(upstage_api_key=upstage_api_key, model="solar-embedding-1-large-query")
    return FAISSRetrieverWithCohere(
        index=index,
        embeddings=embeddings,
        documents=chunks
    )
