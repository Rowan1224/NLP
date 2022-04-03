import logging
import argparse
import json
import numpy as np
from sklearn.utils import shuffle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from evaluate import get_predictions
import pandas as pd
from helpers import check_dir_exists, compute_em, compute_f1, model_name_to_class, DomainDataset
from transformers import AdamW


log = logging.getLogger('transformers')
def create_arg_parser():

    parser = argparse.ArgumentParser()

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
        "-m",
        "--model",
        default="albert",
        type=str,
        choices=["bert", "albert", "electra"],
        help="Select the model for fine-tuning",
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

def set_log(log, filename):

    
    log.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # create file handler which logs info
    check_dir_exists("./output")
    fh = logging.FileHandler(f"{filename}.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    return log


def read_train_data(data):
    """Read the training data and return a context list,
    question list, and answer list"""

    contexts = []
    questions = []
    answers = []
    start = []
    end = []

    for ctx_and_qas in data:
        for qas in ctx_and_qas["questions_and_answers"]:
            contexts.append(ctx_and_qas["context"])
            questions.append(qas["question"])
            answers.append(qas["answer"])
            start.append(qas["answer_start"])
            end.append(qas["answer_start"] + len(qas["answer"]))

    df = pd.DataFrame.from_dict(
        {
            "questions": questions,
            "contexts": contexts,
            "answers": answers,
            "answer_start": start,
            "answer_end": end,
        }
        
    )
    df = shuffle(df)
    split = int(df.shape[0]*0.80)
    train = df.iloc[:split,:]
    dev = df.iloc[split:,:]

    return train, dev


def add_token_positions(encodings, dataset, tokenizer):
    """Add answer start-end token position to encoding object."""
    start_positions = []
    end_positions = []
    dataset = dataset.to_dict('records')

    for idx in range(len(dataset)):
        start_positions.append(
            encodings.char_to_token(idx, dataset[idx]["answer_start"])
        )
        if dataset[idx]["answer_end"] is None:
            end_positions.append(
                encodings.char_to_token(idx, dataset[idx]["answer_end"])
            )
        else:
            end_positions.append(
                encodings.char_to_token(idx, dataset[idx]["answer_end"] - 1)
            )

        # if None, the answer passage has been truncated due to words > 512 so setting last position as 511
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length - 1
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length - 1

    encodings.update(
        {"start_positions": start_positions, "end_positions": end_positions}
    )


def checkpoint_model(
    model, dev_dataset, dev_answers, tokenizer, model_path, current_score, epoch, history, patience, mode="max"
):
    
    # get predicted answers for dev set
    predictions = get_predictions(model, tokenizer, dev_dataset)

    # get F1 score and Exact match for dev set
    f1 = compute_f1(dev_answers, predictions)
    em = compute_em(dev_answers, predictions)

    history.append({'loss': current_score, 'F1':f1, 'EM': em})
    losses = [score['loss'] for score in history]

    #log current results
    log.info(f"After {epoch}th epoch: loss: {current_score}, F1: {f1}, EM: {em}")

    do_break = False

    if (np.max(losses) == current_score and mode == "max") or (
        np.min(losses) == current_score and mode == "min"
    ):

        # Save a trained model and the associated configuration
        model.save_pretrained(model_path)
        return do_break

    else:

        if epoch < patience:
            return do_break
        do_break = True
        for score in losses[epoch - patience : -1]:
            if (mode == "max" and current_score - score > 0) or (
                mode == "min" and score - current_score > 0
            ):
                do_break = False

        return do_break


def train_and_save_model(
    train_dataset,
    dev_dataset,
    dev_answers,
    tokenizer,
    model_class,
    model_path,
    model_name,
    learning_rate,
    batch_size,
    epochs=10,
    patience=3,
):
    """train and save model"""
    model = model_class.from_pretrained(model_name)
    # setup GPU/CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # move model over to detected device
    model.to(device)
    # activate training mode of model
    model.train()
    # initialize adam optimizer with weight decay (reduces chance of overfitting)
    optim = AdamW(model.parameters(), lr=learning_rate)

    # initialize data loader for training data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    dev_dataset = DataLoader(dev_dataset, batch_size=1)
    history = list()
    for epoch in range(epochs):
        # set model to train mode
        model.train()
        # setup loop (we use tqdm for the progress bar)
        loop = tqdm(train_loader, leave=True)
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all the tensor batches required for training
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)
            # train model on batch and return outputs (incl. loss)
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )
            # extract loss
            loss = outputs[0]
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description(f"Epoch {epoch}")
            loop.set_postfix(loss=loss.item())

        
        if checkpoint_model(
            model, dev_dataset, dev_answers, tokenizer, model_path, loss.item(), epoch, history, patience, mode="min"
        ):
            return history
        
    return history    


def load_json(file_path):
    """load json file"""
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def main():

    #get model arguments
    args = create_arg_parser()
    batch = args.batch
    learning_rate = args.learning_rate
    model_args = f"batch_{batch}_lr_{learning_rate}"
    model_key = f"{args.model}-{args.type}"

    #set log
    set_log(log, f"./output/output_{model_key}_{model_args}")

    #set model and tokenizer     
    model, tokenizer, model_name = model_name_to_class[model_key].values()
    tokenizer = tokenizer.from_pretrained(model_name)



    # open JSON file and load into dataframe
    data = load_json("./data/synthetic_qa_pairs.json")
    
    train, dev = read_train_data(data)

    log.info(f"Train Size: {train.shape[0]}")
    log.info(f"Dev Size: {dev.shape[0]}")


   

    # tokenize train and dev set
    train_encodings = tokenizer(
        train['contexts'].ravel().tolist(), train['questions'].ravel().tolist(), truncation=True, padding=True
    )
    add_token_positions(train_encodings, train, tokenizer)


    dev_encodings = tokenizer(
         dev['contexts'].ravel().tolist(), dev['questions'].ravel().tolist(), truncation=True, padding=True
    )

    dev_answers = dev['answers'].ravel().tolist()
  
    # build datasets for our training data
    train_dataset = DomainDataset(train_encodings)
    dev_dataset = DomainDataset(dev_encodings)

   
    # train and save model
    model_path = f"./models/custom_{args.model}_{model_args}"
    check_dir_exists("./models")    
    history = train_and_save_model(
        train_dataset,
        dev_dataset,
        dev_answers,
        tokenizer,
        model,
        model_path,
        model_name,
        learning_rate=learning_rate,
        batch_size=batch,
    )

    # save tokenizer
    tokenizer.save_pretrained(model_path)

    #log training summary 
    history = pd.DataFrame(history)
    print('Training Complete')
    result = f"Avg loss: {history['loss'].mean()} \n Avg F1 dev: {history['F1'].mean()} \n Avg EM dev: {history['EM'].mean()}"
    print(result)
    logging.info(result)




if __name__ == "__main__":
    main()
