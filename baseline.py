import time

import torch

from helpers import (
    evaluate_predictions,
    get_embeddings_from_contexts,
    load_json,
    load_question_answering_model,
    load_semantic_search_model,
    load_text_generation_model,
)


def get_context(model, query, contexts, contexts_emb):
    # Encode query and contexts with the encode function
    query_emb = model.encode(query)

    # Compute dot score between query and all contexts embeddings
    scores = (
        torch.mm(
            torch.from_numpy(query_emb.reshape(1, -1)),
            torch.from_numpy(contexts_emb).transpose(0, 1),
        )[0]
        .cpu()
        .tolist()
    )

    # Combine contexts & scores
    contexts_score_pairs = list(zip(contexts, scores))

    return max(contexts_score_pairs, key=lambda x: x[1])[0]


def get_answer(model, query, context):
    formatted_query = f"{query}\n{context}"
    res = model(formatted_query)
    return res[0]["generated_text"]


def evaluate_t2t_and_semantic_model(
    model, questions, contexts, contexts_emb, generator, test_answers
):
    predictions = []
    for question in questions:
        context = get_context(model, question, contexts, contexts_emb)
        answer = get_answer(generator, question, context)

        predictions.append(answer)

    evaluate_predictions(predictions, test_answers, "Text2Text + Semantic Search")


def evaluate_qa_and_semantic_model(
    semantic_model, queries, qa_model, contexts, contexts_emb
):
    predictions = []
    for query in queries:
        context = get_context(semantic_model, query, contexts, contexts_emb)
        formatted_question = {"question": query, "context": context}
        generated_answers = qa_model(formatted_question)["answer"]
        predictions.append(generated_answers)

    evaluate_predictions(
        predictions, test_answers, "Question Answering + Semantic Search"
    )


def evaluate_semantic_model(model, questions, contexts, contexts_emb, test_answers):
    predictions = [
        get_context(model, question, contexts, contexts_emb) for question in questions
    ]

    evaluate_predictions(predictions, test_answers, "Semantic Search")


if __name__ == "__main__":

    qa_model = load_question_answering_model(
        "Palak/google_electra-base-discriminator_squad"
    )
    t2t_model = load_text_generation_model("allenai/unifiedqa-t5-large")
    semantic_search_model = load_semantic_search_model("all-mpnet-base-v2")

    contexts = open("./data/cleaned_contexts.txt", "r", encoding="utf-8").readlines()
    contexts_emb = get_embeddings_from_contexts(semantic_search_model, contexts)
    test_qa_pairs = load_json("./data/test_qac.json")
    test_questions = [pair["question"] for pair in test_qa_pairs]
    test_answers = [pair["answer"] for pair in test_qa_pairs]

    start = time.time()
    evaluate_semantic_model(
        semantic_search_model, test_questions, contexts, contexts_emb, test_answers
    )
    evaluate_t2t_and_semantic_model(
        semantic_search_model,
        test_questions,
        contexts,
        contexts_emb,
        t2t_model,
        test_answers,
    )
    evaluate_qa_and_semantic_model(
        semantic_search_model, test_questions, qa_model, contexts, contexts_emb
    )

    end = time.time()
    print(f"Time taken: {end - start}")