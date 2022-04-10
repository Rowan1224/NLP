import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import (
    DomainDataset,
    compute_em,
    compute_f1,
    load_json,
    save_answers,
    saved_models_dict,
)


def create_arg_parser():

    """Returns a map with commandline parameters taken from the user"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        default="albert",
        type=str,
        choices=["bert", "albert", "electra"],
        help="Select the model for evaluation",
    )

    parser.add_argument(
        "-t",
        "--type",
        default="squad",
        type=str,
        choices=["squad", "fine"],
        help="Select the model type for fine-tuning (base or fine-tuned/domain adapted)",
    )

    parser.add_argument(
        "-path",
        "--model_path",
        default=None,
        type=str,
        help="Provide the path of the saved model",
    )

    args = parser.parse_args()
    return args


def get_predictions(model, tokenizer, test_dataset):

    """Return predictions from model"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    # set model to evaluation mode
    model.eval()
    # setup loop (we use tqdm for the progress bar)
    predictions = []
    loop = tqdm(test_dataset, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        # pull all the tensor batches required for training
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)

        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end])
        )

        predictions.append(answer)

    return predictions


def main():

    args = create_arg_parser()

    saved_models = saved_models_dict()
    model, tokenizer, fine_model_name, squad_model_name = saved_models[
        args.model
    ].values()
    model_name = fine_model_name if args.type == "fine" else squad_model_name
    model_key = f"{args.model}-{args.type}"
    path_to_model = args.model_path

    if path_to_model is None:
        path_to_model = model_name

    try:
        tokenizer = tokenizer.from_pretrained(path_to_model)
        model = model.from_pretrained(path_to_model)
    except OSError:
        print("Please download the models in to 'models' direcotry")

    test_qac = load_json("./data/test_qac.json")
    test_questions = [pair["question"] for pair in test_qac]
    test_answers = [pair["answer"] for pair in test_qac]
    test_contexts = [pair["context"] for pair in test_qac]

    test_encodings = tokenizer(
        test_contexts, test_questions, truncation=True, padding=True
    )
    test_dataset = DomainDataset(test_encodings)
    test_dataset = DataLoader(test_dataset, batch_size=1)

    predictions = get_predictions(model, tokenizer, test_dataset)

    f1 = compute_f1(test_answers, predictions)
    em = compute_em(test_answers, predictions)

    print(f"F1: {f1}")
    print(f"EM: {em}")

    save_answers(test_questions, test_contexts, test_answers, predictions, model_key)


if __name__ == "__main__":
    main()
