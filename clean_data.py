import csv
import string

import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from english_words import english_words_set
from helpers import write_to_file, write_to_json


def read_data(file_path):
    data = pd.read_csv(file_path)
    return data["text"]


def clean_data(data):
    cleaned_text = []
    removed = []

    progress = tqdm(total=len(data))

    for row in data:
        length_of_valid_text = 0

        for w in word_tokenize(row):
            if w in english_words_set and w not in string.ascii_letters:
                length_of_valid_text += 1

        if length_of_valid_text <= 8:
            removed.append(row)
        else:
            cleaned_text.append(row)

        progress.update(1)

    progress.close()

    return cleaned_text, removed


def read_test_qac_triplets():
    test_file = open("./data/slp_questions.csv", "r", encoding="utf-8")
    test_qac = csv.DictReader(test_file)
    qa_pairs = [
        {
            "question": sample["question"],
            "answer": sample["answer"],
            "context": sample["paragraph"],
        }
        for sample in test_qac
    ]

    return qa_pairs


if __name__ == "__main__":
    data = read_data("./data/slp3ed.csv")

    cleaned_contexts, removed_contexts = clean_data(data)

    write_to_file("./data/cleaned_contexts.txt", cleaned_contexts)
    write_to_file("./data/removed_contexts.txt", removed_contexts)

    test_qac = read_test_qac_triplets()
    write_to_json("./data/test_qac.json", test_qac)

    print(f"Number of Original Contexts: {data.shape[0]}")
    print(f"Number of Cleaned Contexts: {len(cleaned_contexts)}")
    print(f"Number of Removed Contexts: {len(removed_contexts)}")
    print(f"Number of Test QAC Triplets: {len(test_qac)}")
