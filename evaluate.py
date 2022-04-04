import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DomainDataset, compute_em, compute_f1, load_json, save_answers, model_name_to_class



def create_arg_parser():

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
        "-b", "--batch", default=4, type=int, help="Provide the number of batch"
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=5e-5,
        type=float,
        help="Provide the learning rate",
    )
    parser.add_argument(
        "-t",
        "--type",
        default="base",
        type=str,
        choices=["base", "fine"],
        help="Select the model type for fine-tuning (base or fine-tuned)",
    )


    args = parser.parse_args()
    return args


def get_predictions(model, tokenizer, test_dataset):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    # set model to evaluation mode
    model.eval()
    # setup loop (we use tqdm for the progress bar)
    predictions = list()
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
    batch = args.batch
    learning_rate = args.learning_rate

    model_args = f"batch_{batch}_lr_{learning_rate}"
    model_key = f"{args.model}-{args.type}"

    model, tokenizer, _ = model_name_to_class[model_key].values()

    model, tokenizer, _, _ = model_name_to_class[args.model].values()

    path_to_model = f"./models/custom_{args.model}_{args.type}_{model_args}"

    tokenizer = tokenizer.from_pretrained(path_to_model)
    model = model.from_pretrained(path_to_model)

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

    save_answers(test_questions,test_contexts,test_answers,predictions,model_key)

if __name__ == "__main__":
    main()
