import pandas as pd
import json
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from pipelines import pipeline
import torch, gc


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

    test_context = [contexts[274]]

    for context in contexts:
        cleaned_sentences = []
        for sent in sent_tokenize(context):
            for pat, rep in replacements:
                #print(f"A {sent!r}")
                sent = pat.sub(rep, sent)
                #print(f"B {sent!r}")
            cleaned_sentences.append(sent)
        cleaned_contexts.append(" ".join(cleaned_sentences))

    # print(cleaned_contexts)
    print(len(contexts))
    print(len(cleaned_contexts))
    # print(cleaned_contexts[274])
    # print(f"{cleaned_contexts[0]!r}")

    return cleaned_contexts


def main():


    # df = pd.read_csv('./Data/slp3ed.csv')
    # contexts = df["text"].tolist()

    contexts = []
    with open('./Data/clean-contexts.txt', 'r') as f:
        for context in f:
            contexts.append(context)

    train_data = []
    failed_contexts = []

    cleaned_contexts = prepare_input_for_QAgenerator(contexts)
    nlp = pipeline("multitask-qa-qg", model="valhalla/t5-base-qa-qg-hl")

    for idx, context in enumerate(cleaned_contexts):

        if idx%100==0:
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            print(idx)

        try:
            context = {
                "context": context,
                "questions_and_answers": [
                    {"answer_start": context.find(q_and_a["answer"]), **q_and_a}
                    for q_and_a in nlp(context)
                ],
            }
        except ValueError:
            failed_contexts.append(context)
            continue
        train_data.append(context)


    with open('./Data/train_data.json', 'w') as f:
        json.dump(train_data, f)

    with open('./Data/failed_contexts.json', 'w') as f:
        json.dump(failed_contexts, f)

    print(f"train_data: {len(train_data)}")
    print(f"failed contexts: {len(failed_contexts)}")

    # with open('../Data/failed_contexts.json') as file:
    #     bad = json.load(file)

    # for text in bad:

    #     nlp = pipeline("multitask-qa-qg", model="valhalla/t5-base-qa-qg-hl")
    #     # text = "The regular expression [ /[1234567890]/ ] specifies any single digit. While such classes of characters as digits or letters are important building blocks in expressions, they can get awkward (e.g., it\u2019s inconvenient to specify [ /[ABCDEFGHIJKLMNOPQRSTUVWXYZ]/ ] to mean \u201cany capital letter\u201d). In cases where there is a well-defined sequence associated with a set of characters, the brackets can be used with the dash (-) to specify any one character in a range. The pattern [ /[2-5]/ ] specifies any one of the characters 2, 3, 4, or 5. The pattern [ /[b-g]/ ] specifies one of the characters b, c, d, e, f, or g. Some other examples are shown in Figure 2.3."
    #     print(nlp(text))


if __name__ == '__main__':

    main()
