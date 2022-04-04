#!/usr/bin/env python
# coding: utf-8
###


import time

import numpy as np
import pandas as pd
import torch

import faiss
from utils import (
    evaluate_predictions,
    get_embeddings_from_contexts,
    load_json,
    load_question_answering_model,
    load_semantic_search_model,
    load_text_generation_model,
)


def convert_embeddings_to_faiss_index(embeddings, context_ids):
    embeddings = np.array([embedding for embedding in embeddings]).astype(
        "float32"
    )  # Step 1: Change data type

    index = faiss.IndexFlatL2(embeddings.shape[1])  # Step 2: Instantiate the index
    index = faiss.IndexIDMap(index)  # Step 3: Pass the index to IndexIDMap

    index.add_with_ids(embeddings, context_ids)  # Step 4: Add vectors and their IDs
    return index


def vector_search(query, model, index, num_results=10):
    """Tranforms query to vector using a pretrained, sentence-level
    model and finds similar vectors using FAISS.
    """
    vector = model.encode(list(query))
    D, I = index.search(np.array(vector).astype("float32"), k=num_results)
    return D, I


def id2details(df, I, column):
    """Returns the paper titles based on the paper index."""
    return [list(df[df.index.values == idx][column]) for idx in I[0]]


def combine(user_query, model, index, df, column, num_results=1):
    D, I = vector_search([user_query], model, index, num_results=num_results)
    return id2details(df, I, column)[0][0]


def get_answer(model, query, context):
    formatted_query = f"{query}\n{context}"
    res = model(formatted_query)
    return res[0]["generated_text"]


def evaluate_t2t_and_semantic_model(
    model, questions, contexts, index, generator, test_answers
):
    predictions = []
    for question in questions:
        context = combine(question, model, index, contexts, "contexts")
        answer = get_answer(generator, question, context)

        predictions.append(answer)

    evaluate_predictions(predictions, test_answers, "Text2Text + Semantic Search")


def evaluate_qa_and_semantic_model(
    semantic_model, questions, qa_model, contexts, index
):
    predictions = []
    for question in questions:
        context = combine(question, semantic_model, index, contexts, "contexts")
        formatted_question = {"question": question, "context": context}
        generated_answers = qa_model(formatted_question)["answer"]
        predictions.append(generated_answers)

    evaluate_predictions(
        predictions, test_answers, "Question Answering + Semantic Search"
    )


def evaluate_semantic_model(model, questions, contexts, index, test_answers):
    predictions = [
        combine(question, model, index, contexts, "contexts") for question in questions
    ]

    evaluate_predictions(predictions, test_answers, "Semantic Search")


if __name__ == "__main__":

    #best QA Models from fine tuning
    qa_models = ['albert_batch_4_lr_3e-05','bert_batch_4_lr_5e-05','electra_batch_4_lr_3e-05']

    t2t_model = load_text_generation_model("allenai/unifiedqa-t5-large")
    semantic_search_model = load_semantic_search_model("all-mpnet-base-v2")

    contexts = open("./data/cleaned_contexts.txt", "r", encoding="utf-8").readlines()
    contexts = pd.DataFrame(contexts, columns=["contexts"])
    contexts_emb = get_embeddings_from_contexts(
        semantic_search_model, contexts.contexts.values
    )

    index = convert_embeddings_to_faiss_index(contexts_emb, contexts.index.values)

    test_qac = load_json("./data/test_qac.json")
    test_questions = [pair["question"] for pair in test_qac]
    test_answers = [pair["answer"] for pair in test_qac]

    start = time.time()
    evaluate_semantic_model(
        semantic_search_model, test_questions, contexts, index, test_answers
    )
    evaluate_t2t_and_semantic_model(
        semantic_search_model,
        test_questions,
        contexts,
        index,
        t2t_model,
        test_answers,
    )
    for model_name in qa_models:
        qa_model = load_question_answering_model(
            f"./models/custom_{model_name}"
        )
        evaluate_qa_and_semantic_model(
            semantic_search_model, test_questions, qa_model, contexts, index
        )

    end = time.time()
    print(f"Time taken: {end - start}")
