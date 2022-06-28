import json
import os
import re
import string

import numpy as np
import pandas as pd
import torch


class DomainDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def common_elements(list1, list2):
    return [element for element in list1 if element in list2]


def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_em(predictions, truths):
    f1_scores = []
    for pred, truth in zip(predictions, truths):
        f1_scores.append(int(normalize_text(pred) == normalize_text(truth)))

    return np.mean(f1_scores)


def compute_f1(predictions, truths):
    f1_scores = []
    for pred, truth in zip(predictions, truths):
        pred_tokens = normalize_text(pred).split()
        truth_tokens = normalize_text(truth).split()

        # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            f1_scores.append(int(pred_tokens == truth_tokens))
            continue

        common_tokens = common_elements(pred_tokens, truth_tokens)
        # if there are no common tokens then f1 = 0
        if len(common_tokens) == 0:
            f1_scores.append(0)
            continue

        prec = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
        rec = len(common_tokens) / len(truth_tokens) if truth_tokens else 0

        f1_scores.append(2 * (prec * rec) / (prec + rec) if (prec + rec) else 0)

    return np.mean(f1_scores)


def load_json(file_path):
    """load json file"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_semantic_search_model(model_name):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def load_question_answering_model(model_name):
    from transformers import pipeline
    try:
       return pipeline("question-answering", model=model_name)
    except:
        print("Model Not Found: Please download the models in to 'models' direcotry")
    


def get_embeddings_from_contexts(model, contexts):
    return model.encode(contexts)


def evaluate_predictions(predictions, test_answers, model_name):
    f1 = compute_f1(predictions, test_answers)
    em = compute_em(predictions, test_answers)

    print(f"Results for {model_name} model:")
    print(f"F1: {f1}, EM:{em}")


def write_to_file(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(line)
            f.write("\n")


def write_to_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def check_dir_exists(dir):
    path = str(dir)
    if not os.path.exists(path):
        os.makedirs(path)

def save_answers(questions, contexts, true_anwers, predicted_anwers, experiment):

    #save predicted answers in json format
    df = pd.DataFrame.from_dict(
        {
            "questions": questions,
            "contexts": contexts,
            "answers": true_anwers,
            "predictions": predicted_anwers,
        }
    )

    check_dir_exists("./output")
    df.to_json(f"./output/output_{experiment}.json", orient="records")


def model_name_to_class_dict():
    from transformers import (
        AlbertForQuestionAnswering,
        AlbertTokenizerFast,
        DistilBertForQuestionAnswering,
        DistilBertTokenizerFast,
        ElectraForQuestionAnswering,
        ElectraTokenizerFast,
        RobertaForQuestionAnswering,
        RobertaTokenizerFast,
    )

    return {
            "albert": {
                "model": AlbertForQuestionAnswering,
                "tokenizer": AlbertTokenizerFast,
                "fine_model_name": "albert-base-v2",
                "squad_model_name": "twmkn9/albert-base-v2-squad2",
            },
            "bert": {
                "model": DistilBertForQuestionAnswering,
                "tokenizer": DistilBertTokenizerFast,
                "fine_model_name": "distilbert-base-uncased",
                "squad_model_name": "distilbert-base-uncased-distilled-squad",
            },
            "electra": {
                "model": ElectraForQuestionAnswering,
                "tokenizer": ElectraTokenizerFast,
                "fine_model_name": "google/electra-base-discriminator",
                "squad_model_name": "Palak/google_electra-base-discriminator_squad",
            },
            "roberta": {
            "model": RobertaForQuestionAnswering,
            "tokenizer": RobertaTokenizerFast,
            "fine_model_name": "roberta-base",
            "squad_model_name": "deepset/roberta-base-squad2",
        },
        }

def saved_models_dict():
    from transformers import (
        AlbertForQuestionAnswering,
        AlbertTokenizerFast,
        DistilBertForQuestionAnswering,
        DistilBertTokenizerFast,
        ElectraForQuestionAnswering,
        ElectraTokenizerFast,
        RobertaForQuestionAnswering,
        RobertaTokenizerFast,
    )
    return {
        "albert": {
            "model": AlbertForQuestionAnswering,
            "tokenizer": AlbertTokenizerFast,
            "fine_model_name": "rowan1224/albert-slp",
            "squad_model_name": "rowan1224/albert-squad-slp",
        },
        "bert": {
            "model": DistilBertForQuestionAnswering,
            "tokenizer": DistilBertTokenizerFast,
            "fine_model_name": "rowan1224/distilbert-slp",
            "squad_model_name": "rowan1224/distilbert-squad-slp",
        },
        "electra": {
            "model": ElectraForQuestionAnswering,
            "tokenizer": ElectraTokenizerFast,
            "fine_model_name": "rowan1224/electra-slp",
            "squad_model_name": "rowan1224/electra-squad-slp",
        },

    }
