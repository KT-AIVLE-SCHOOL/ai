from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score

# 평가 데이터셋 생성 함수
def create_evaluation_dataset(queries, hybrid_search, index, bm25, embeddings, documents, rag_chain, k=5):
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    references = []

    for query, ground_truth in queries:
        # 하이브리드 검색
        relevant_chunks = hybrid_search(query, index, bm25, embeddings, documents, k=k)
        context = [chunk.page_content for chunk in relevant_chunks]

        # RAG 체인 사용해 답변 생성
        response = rag_chain.invoke({"input": query})
        answer = response["answer"]

        questions.append(query)
        answers.append(answer)
        contexts.append(context)
        ground_truths.append([ground_truth])
        references.append(ground_truth)

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": ground_truths,
        "reference": references,
    }

    return Dataset.from_dict(data)

# 평가 함수
def evaluate_responses(evaluation_dataset):
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
    )

    df = result.to_pandas()
    return df

# 텍스트 생성 평가 함수
def evaluate_generation(ground_truth, generated_answer):
    # ROUGE
    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge_scores = rouge.score(ground_truth, generated_answer)

    # BLEU
    smoothing = SmoothingFunction().method4
    bleu_score = sentence_bleu([ground_truth.split()], generated_answer.split(), smoothing_function=smoothing)

    # BERTScore
    P, R, F1 = bert_score([generated_answer], [ground_truth], lang="ko")

    return {
        "rouge1": rouge_scores["rouge1"].fmeasure,
        "rougeL": rouge_scores["rougeL"].fmeasure,
        "bleu": bleu_score,
        "bert_f1": F1.mean().item()
    }

# 평가 결과 출력 함수
def print_evaluation_results(df):
    for i in range(len(df['user_input'])):
        print(f'input_text{i} = {df["user_input"][i]}')
        print(f'response{i} = {df["response"][i]}')
        print(f'answer_relevancy = {df["answer_relevancy"][i]}')
        print()

if __name__ == "__main__":
    evaluation_queries = [
    ("육아휴직 제도에 대해 설명해주세요.", "육아휴직은 만 8세 이하 또는 초등학교 2학년 이하의 자녀를 양육하기 위해 근로자가 신청할 수 있는 제도입니다. 최대 1년까지 사용 가능하며, 육아휴직 기간 동안 급여를 지원받을 수 있습니다."),
    ("육아휴직 제도에 대해 자세히 설명해줘", "육아휴직은 만 8세 이하 또는 초등학교 2학년 이하의 자녀를 양육하기 위해 근로자가 신청할 수 있는 제도입니다. 최대 1년까지 사용 가능하며, 육아휴직 기간 동안 급여를 지원받을 수 있습니다."),
    ("산후 우울증의 증상은 무엇인가요?", "산후 우울증의 주요 증상으로는 지속적인 슬픔, 불안, 무기력감, 수면 장애, 식욕 변화, 집중력 저하, 아기에 대한 무관심 또는 과도한 걱정 등이 있습니다."),
    ("모유를 언제까지 먹여야 될까?", "모유수유의 경우 WHO에서는 2세까지 권장합니다."),
    ("아기를 트름시키려면 어떻게 해야 돼?", "아기를 엄마 무릎에 앉혀서 엄마의 한쪽 손은 윗가슴과 아래턱을 받치고, 다른 손은 손바닥으로 아기의 등을아래에서 위로 쓸어 올리거나 토닥거립니다.아기를 엄마의 어깨 위까지 올려서 아기의 상체가 엄마의 어깨에 걸치게 합니다. 그리고 한 손은 아기의 엉덩이를 받치고, 다른 손은 아가의 등을 쓰다듬거나 토닥거려 줍니다. 아기를 엄마 무릎위에 눕혀서 손바닥으로 쓸어주거나 가볍게 토닥거려 줍니다."),
    ("육아휴직 지원금에 대해 알려줘.", "육아휴직급여는 육아기 근로자의 고용안정과 일·가정 양립 지원을 위해 육아휴직을 사용한 근로자를 지원하는 제도입니다. 육아휴직급여를 1년간 통상임금의 80%(상한액 150만 원, 하한액 70만 원)로 지원합니다."),
    ("엄마는 이미 육아휴직을 1년 다 쓴 상황에서 아빠가 3개월을 쓰면 엄마의 육아휴직도 6개월이 연장되나요?", "3개월의 육아휴직을 언제 사용했는지와 관계없이 부모 모두 3개월 이상 사용했다면 6개월 연장됩니다. 다만, 연장된 6개월은 ’25.2.23. 이후 사용할 수 있습니다."),
    ("미숙아를 출산한 경우 출산전후휴가 기간이 확대되는데, 이때 ‘미숙아’의 범위는 어떻게 되나요?", "미숙아는 임신 37주 미만의 출생아로 출생 후 24시간 이내에 신생아 중환자실에 입원한 신생아 또는 출생 시 체중이 2.5킬로그램 미만인 영유아로 출생 후 24시간 이내에 신생아 중환자실에 입원한 신생아를 뜻합니다."),
    ("아기가 화상을 입었어요 어떻게 해야 될까요?", "가장 중요한 일은 화상 부위를 식히는 것입니다. 흐르는 차가운 물로 충분히 식혀 주어야 하며, 이때 화상 부위가 너무 넓다면 아이가 저체온이 되지 않도록 주의해야 합니다."),
    ("6개월 미만의 아기 발달 체크리스트를 알려줘.", "아기는 말을 걸거나 안아주면 차분해지며, 사람의 얼굴을 바라보고 미소 짓는 것에 반응합니다. 친숙한 사람을 알아보기 시작하고, 거울 속 자신의 모습에 흥미를 보입니다. 울음 외에도 다양한 소리를 내며, 우우, 아아 같은 옹알이를 합니다. 사람의 목소리에 반응하여 소리 나는 쪽으로 고개를 돌리고, 상대방과 번갈아가며 소리를 내기도 합니다. 인지 발달 면에서는 배고픔을 표현하고, 자신의 손을 관심 있게 바라봅니다. 물건을 입으로 탐색하고, 원하는 장난감을 향해 손을 뻗습니다. 신체 발달로는 머리를 잘 가누고, 장난감을 쥐거나 흔들 수 있습니다. 6개월 즈음에는 혼자 앉기 시작하고, 한 손에서 다른 손으로 물건을 옮길 수 있습니다. 이러한 발달 지표들은 일반적인 것이지만, 개별 아기마다 발달 속도의 차이가 있을 수 있습니다.")
    ]

    # 평가 데이터셋 생성
    evaluation_dataset = create_evaluation_dataset(evaluation_queries, hybrid_search, index, bm25, embeddings, documents, rag_chain)

    # 응답 평가 실행
    df = evaluate_responses(evaluation_dataset)
    print_evaluation_results(df)

    # 생성 텍스트 평가
    ground_truth = "육아휴직급여는 육아기 근로자의 고용안정과 일·가정 양립 지원을 위해 육아휴직을 사용한 근로자를 지원하는 제도입니다. 육아휴직급여를 1년간 통상임금의 80%(상한액 150만 원, 하한액 70만 원)로 지원합니다."
    generated_answer = "육아휴직 지원금은 육아휴직을 사용한 근로자를 지원하기 위한 제도로, 육아기 근로자의 고용안정과 일가정 양립을 도와주는 역할을 합니다. 육아휴직을 사용한 근로자는 최대 1년 동안 통상임금의 일정 비율에 따라 지원금을 받을 수 있습니다. 일반적으로 육아휴직급여는 월 상한액 150만 원, 하한액 70만 원으로 설정되어 있습니다. 한부모 근로자의 경우 첫 3개월 동안은 통상임금의 상한액이 250만 원으로 높아지며, 이후 12개월까지는 다시 150만 원으로 조정됩니다. 부모가 동시에 또는 순차적으로 육아휴직을 사용할 경우, 각각 사용한 기간 중 공통으로 사용한 기간을 기준으로 지원금이 지급됩니다. 육아휴직을 30일 이상 부여받은 근로자여야 하며, 육아휴직 시작일 이전에 피보험 단위기간이 총 180일 이상이어야 합니다. 신청은 관할 고용센터를 통해 직접 하거나 우편 제출, 또는 고용보험 홈페이지에서 가능하며, 육아휴직 종료 후 1개월 이내에 신청해야 합니다. 이와 같은 육아휴직 제도는 일하는 부모가 자녀를 돌보는 데 필요한 경제적 지원을 제공하여, 보다 나은 육아 환경을 조성하는 데 기여하고 있습니다."

    generation_metrics = evaluate_generation(ground_truth, generated_answer)
    print("Generation Metrics:", generation_metrics)