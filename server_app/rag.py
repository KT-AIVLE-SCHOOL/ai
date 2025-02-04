import os
from chunk_embedding import FAISSRetrieverWithCohere
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# RAG
def create_rag_chain(chunks, retriever, openai_api_key, model="gpt-4-1106-preview"):
    # LLM 초기화
    llm = ChatOpenAI(
        model=model,
        temperature=0.3,
        openai_api_key=openai_api_key,
        streaming=True
    )

    # 시스템 프롬프트 정의
    system_prompt = (
        "당신은 육아와 아동 발달 전문가인 영유아 육아 지원 챗봇입니다. 따라서서, 육아에 대한 사회적 제도나 응급처치 등에 관한 전문적 지식도 갖고 있습니다."
        "당신은 신혼부부의 아이에게 필요한 정보를 제공해야 됩니다."
        "사용자의 질문에서 요구하는게 무엇인지 꼭 기억하고 알맞은 답변을 제공하세요."
        "사용자가 추가적인 질문을 하지 않도록 자세한 내용을 알려주세요."
        "마크다운을 사용하지 말고, 문맥을 매끄럽게 만들어서 답변해주세요."
        "다음 검색된 Context 조각을 사용하여 질문에 답을 해야합니다."
        "사용자가 제시한 정보가 부족해 추가적인 정보가 필요하다면 되물어보세요"
        "맥락이 없거나 답을 모르는 경우에는 다음 문맥을 사용하여 질문에 답하세요."
        "정확한 답변을 생성하기 어렵습니다. 보건복지부의 아이사랑 홈페이지를 참고하세요." 
        "\n\n{context}"
    )

    # 프롬프트 템플릿 생성
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # 출력 파서
    output_parser = StrOutputParser()

    # QA 체인 및 RAG 체인
    qa_chain = create_stuff_documents_chain(llm, prompt_template, output_parser=output_parser)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    return rag_chain

# 사용자 질문에 대한 답변 생성 
def get_rag_response(rag_chain, query):
    print(f"get_rag_response 함수 시작. 쿼리: {query}")
    try:
        print("rag_chain.invoke 호출 시작")
        response = rag_chain.invoke({"input": query})
        print(f"rag_chain.invoke 호출 완료. 응답: {response}")
        if "answer" in response:
            return response["answer"]
        else:
            print("Warning: 'answer' key not found in response")
            return "죄송합니다. 답변을 생성하는 데 문제가 발생했습니다."
    except Exception as e:
        print(f"Error in get_rag_response: {str(e)}")
        return "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다. 다시 시도해 주세요."


# # 비동기 방식
# import asyncio
# from concurrent.futures import TimeoutError

# async def get_rag_response_async(rag_chain, query):
#     print(f"get_rag_response 함수 시작. 쿼리: {query}")
#     try:
#         print("rag_chain.invoke 호출 시작")
#         response = await asyncio.wait_for(rag_chain.ainvoke({"input": query}), timeout=30.0)
#         print(f"rag_chain.invoke 호출 완료. 응답: {response}")
#         if "answer" in response:
#             return response["answer"]
#         else:
#             print("Warning: 'answer' key not found in response")
#             return "죄송합니다. 답변을 생성하는 데 문제가 발생했습니다."
#     except TimeoutError:
#         print("Error: rag_chain.invoke 호출 시간 초과")
#         return "죄송합니다. 응답 생성 시간이 초과되었습니다. 다시 시도해 주세요."
#     except Exception as e:
#         print(f"Error in get_rag_response: {str(e)}")
#         return "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다. 다시 시도해 주세요."

# def get_rag_response(rag_chain, query):
#     return asyncio.run(get_rag_response_async(rag_chain, query))
