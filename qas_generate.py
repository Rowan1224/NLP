import re
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import json
from pipelines import pipeline

def write_to_json(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def prepare_input_for_QAgenerator(contexts):
    cleaned_contexts = []

    replacements = [
        (re.compile(r"(?:^|\b)fig\.(?:\b|$)"), "figure"),
        (re.compile(r"(?:^|\b)Fig\.(?:\b|$)"), "Figure"),
        (re.compile(r"(?<!\$)\b(\d+\.\d+)\b"), r"[ \1 ]"),
        (re.compile(r"^(\d\.)$"), r"[ \1 ]"),
        (re.compile(r"(s/[^/]+/[^/]+/)"), r"[ \1 ]"),
        (re.compile(r"(/[^/]+/)"), r"[ \1 ]"),
        (re.compile(r"•"), r"[ • ]"),
    ]

    for context in contexts:
        cleaned_sentences = []

        for sent in sent_tokenize(context):
            for pattern, replacement in replacements:
                sent = pattern.sub(replacement, sent)

            cleaned_sentences.append(sent)

        cleaned_contexts.append(" ".join(cleaned_sentences))

    return cleaned_contexts


def main():
    contexts = open("./data/cleaned_contexts.txt", "r", encoding="utf-8").readlines()
    cleaned_contexts = prepare_input_for_QAgenerator(contexts)

    data = []

    nlp = pipeline("multitask-qa-qg", model="valhalla/t5-base-qa-qg-hl")

    progress = tqdm(total=len(cleaned_contexts))
    for context in cleaned_contexts:

        data.append(
            {
                "context": context,
                "questions_and_answers": [
                    {"answer_start": context.find(q_and_a["answer"]), **q_and_a}
                    for q_and_a in nlp(context)
                ],
            }
        )

        progress.update(1)

    progress.close()

    write_to_json("./data/synthetic_qa_pairs.json", data)



if __name__ == "__main__":
    main()
