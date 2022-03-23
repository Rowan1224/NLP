import json
import torch
import os
from transformers import AlbertForQuestionAnswering, AdamW, AlbertTokenizerFast
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse

def create_arg_parser():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-b", "--batch", default=4, type=int, help='Provide the number of batch')
    parser.add_argument("-lr", "--learning_rate", default=5e-5, type=float, help='Provide the learning rate')               
    
    args = parser.parse_args()
    return args



def read_train_data(data):
    '''Read the training data and return a context list,
       question list, and answer list'''

    contexts = []
    questions = []
    answers = []

    for ctx_and_qas in data:
        for qas in ctx_and_qas["questions_and_answers"]:
            contexts.append(ctx_and_qas["context"])
            questions.append(qas["question"])
            answers.append({
                "text": qas["answer"],
                "answer_start": qas["answer_start"],
                "answer_end": qas["answer_start"] + len(qas["answer"]),
            })

    return contexts, questions, answers


def add_token_positions(encodings, answers, tokenizer):
    '''Add answer start-end token position to encoding object.'''
    start_positions = []
    end_positions = []

    for idx in range(len(answers)):
        start_positions.append(encodings.char_to_token(idx, answers[idx]['answer_start']))
        if answers[idx]['answer_end'] is None:
          end_positions.append(encodings.char_to_token(idx, answers[idx]['answer_end']))
        else:
          end_positions.append(encodings.char_to_token(idx, answers[idx]['answer_end'] - 1))

        #if None, the answer passage has been truncated due to words > 512 so setting last position as 511
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length-1
        if end_positions[-1] is None:
            end_positions[-1] = tokenizer.model_max_length-1

    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})


class DomainDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def checkpoint_model(model, model_path, current_score, epoch, history, patience, mode='max'):

    if mode=='min':
        print('Current best', min(history), 'after {}th epoch'.format(epoch))

    if mode=='max':
        print('Current best', max(history), 'after {}th epoch'.format(epoch))
    do_break = False

    if ((np.max(history) == current_score and mode == 'max') or
            (np.min(history) == current_score and mode == 'min')):

        # Save a trained model and the associated configuration
        model.save_pretrained(model_path)
        return do_break

    else:
        
        if epoch<patience: return do_break
        do_break = True
        for score in history[epoch-patience:-1]:
            if (mode=='max' and current_score-score > 0) or ( mode=='min' and score-current_score > 0):
                do_break = False
    
        return do_break



def train_and_save_model(train_dataset, model_path, model_name, learning_rate, batch_size, epochs=3, patience = 2):
    '''train and save model'''
    model = AlbertForQuestionAnswering.from_pretrained(model_name)
    # setup GPU/CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # move model over to detected device
    model.to(device)
    # activate training mode of model
    model.train()
    # initialize adam optimizer with weight decay (reduces chance of overfitting)
    optim = AdamW(model.parameters(), lr=learning_rate)

    # initialize data loader for training data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            # train model on batch and return outputs (incl. loss)
            outputs = model(input_ids, attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions)
            # extract loss
            loss = outputs[0]
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

        history.append(loss.item())
        if checkpoint_model(model, model_path, loss.item(), epoch, history, patience, mode='min'):
            break


def main():

    args = create_arg_parser()

    
    batch = args.batch
    learning_rate = args.learning_rate
    

    model_args = f"batch_{batch}_lr_{learning_rate}" 
    # open JSON file and load intro dictionary
    with open('./Data/train_data.json', 'r') as f:
        data = json.load(f)

    train_contexts, train_questions, train_answers = read_train_data(data)

    model_name = 'albert-base-v2'

    # tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    tokenizer = AlbertTokenizerFast.from_pretrained(model_name)

    # tokenize
    train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)

    add_token_positions(train_encodings, train_answers, tokenizer)

    # print(train_encodings.keys())

    # build datasets for our training data
    train_dataset = DomainDataset(train_encodings)

    # train and save model


    model_path = f'./Models/albert-custom/{model_args}'

    # if not os.path.exists(model_path):
    #         os.makedirs(model_path)

    train_and_save_model(train_dataset, model_path, model_name, learning_rate= learning_rate, batch_size=batch)

    # save tokenizer
    tokenizer.save_pretrained(model_path)


if __name__ == '__main__':
    main()
