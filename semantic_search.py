import argparse
import time

import numpy as np
import pandas as pd
import torch

import faiss
from sentence_transformers import util
from utils import (evaluate_predictions, get_embeddings_from_contexts,
                   load_json, load_question_answering_model,
                   load_semantic_search_model, save_answers)


def create_arg_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--faiss",
        action="store_true",
        help="Whether to use faiss or not (default: False)",
    )
    args = parser.parse_args()
    return args


def convert_embeddings_to_faiss_index(embeddings, context_ids):
    embeddings = np.array(embeddings).astype(
        "float32"
    )  # Step 1: Change data type

    index = faiss.IndexFlatIP(embeddings.shape[1])  # Step 2: Instantiate the index
    index = faiss.IndexIDMap(index)  # Step 3: Pass the index to IndexIDMap

    index.add_with_ids(embeddings, context_ids)  # Step 4: Add vectors and their IDs

    res = faiss.StandardGpuResources()  # Step 5: Instantiate the resources
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  # Step 6: Move the index to the GPU
    return gpu_index


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
    
def get_context(model, query, contexts, contexts_emb):
    # Encode query and contexts with the encode function
    query_emb = model.encode(query)
    query_emb = torch.from_numpy(query_emb.reshape(1, -1))
    contexts_emb = torch.from_numpy(contexts_emb)
    # Compute similiarity score between query and all contexts embeddings
    scores = util.cos_sim(query_emb,contexts_emb)[0].cpu().tolist()
    # Combine contexts & scores
    contexts_score_pairs = list(zip(contexts, scores))

    return max(contexts_score_pairs, key=lambda x: x[1])[0]


def get_answer(model, query, context):
    formatted_query = f"{query}\n{context}"
    res = model(formatted_query)
    return res[0]["generated_text"]

def evaluate_qa_and_semantic_model(
    semantic_model, queries, qa_model, contexts, contexts_emb, model_name, index=None
):
    predictions = []
    best_contexts = list()
    for query in queries:
        if index:
            context = combine(query, semantic_model, index, contexts, "contexts")
        else:
            context = get_context(semantic_model, query, contexts, contexts_emb)
        formatted_question = {"question": query, "context": context}
        generated_answers = qa_model(formatted_question)["answer"]
        best_contexts.append(context)
        predictions.append(generated_answers)

    evaluate_predictions(
        predictions, test_answers, f"Question Answering + Semantic Search + {model_name}"
    )
    save_answers(queries, best_contexts, test_answers, predictions, f"QA_Semantic_Search_{model_name}")

def evaluate_semantic_model(model, questions, contexts, contexts_emb, test_answers, index=None):
    predictions = [
        combine(question, model, index, contexts, "contexts") if index else get_context(model, question, contexts, contexts_emb) for question in questions
    ]

    evaluate_predictions(predictions, test_answers, "Semantic Search")
    save_answers(questions, predictions, test_answers, predictions, "Semantic_Search")


if __name__ == "__main__":

    args = create_arg_parser()

    #best QA Models from fine tuning
    qa_models = ['custom_electra_batch_16_lr_1e-05']

    semantic_search_model = load_semantic_search_model("all-mpnet-base-v2")

    contexts = open("/data/s4992113/nlp_data/cleaned_contexts.txt", "r", encoding="utf-8").readlines()

    if args.faiss:
        contexts = pd.DataFrame(contexts, columns=["contexts"])
        contexts_emb = get_embeddings_from_contexts(
            semantic_search_model, contexts.contexts.values
        )

        index = convert_embeddings_to_faiss_index(contexts_emb, contexts.index.values)
    else:
        contexts_emb = get_embeddings_from_contexts(semantic_search_model, contexts)
        index = None

    test_qa_pairs = load_json("/data/s4992113/nlp_data/test_qac.json")
    test_questions = [pair["question"] for pair in test_qa_pairs]
    test_answers = [pair["answer"] for pair in test_qa_pairs]
    test_contexts = [pair["context"] for pair in test_qa_pairs]

    start = time.time()
    evaluate_semantic_model(
        semantic_search_model, test_questions, contexts, contexts_emb, test_answers, index
    )
        
    for model_name in qa_models:
        qa_model = load_question_answering_model(
            f"./models/{model_name}"
        )
        if qa_model is not None:
            evaluate_qa_and_semantic_model(
                semantic_search_model, test_questions, qa_model, contexts, contexts_emb, model_name, index
            )

    end = time.time()
    print(f"Time taken: {end - start}")
