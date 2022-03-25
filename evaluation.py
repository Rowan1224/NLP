
import argparse
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from nltk import word_tokenize
import string
from datasets import load_dataset
import os
from transformers import (

    AlbertForQuestionAnswering,
    AlbertTokenizerFast,
    DistilBertForQuestionAnswering,
    DistilBertTokenizerFast

)

def create_arg_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default='albert', type=str,
                       choices=['bert', 'albert', 'electra'],
                    help='Select the model for evaluation')
    parser.add_argument("-b", "--batch", default=4, type=int, help='Provide the number of batch')
    parser.add_argument("-lr", "--learning_rate", default=5e-5, type=float, help='Provide the learning rate') 

    args = parser.parse_args()
    return args


def verify(word):
    if word in string.punctuation:
        return False
    return True

def evaluateF1(predict,true):

    predict = [w for w in word_tokenize(predict.lower()) if verify(w)]
    true = [w for w in word_tokenize(true.lower()) if verify(w)]
    common = [w for w in true if w in predict]
    
    if len(common) == 0:
        return 0
    prec = len(common)/len(predict)
    recall = len(common)/len(true)
    
    return 2 * (prec*recall)/(prec+recall)

    
def evaluateEM(predict,true):

    predict = [w for w in word_tokenize(predict.lower()) if verify(w)]
    true = [w for w in word_tokenize(true.lower()) if verify(w)]
    
    
    if " ".join(predict) == " ".join(true):
        return 1
    else:
        return 0


class DomainDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)



def to_list(tensor):
    return tensor.detach().cpu().tolist()

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
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)

        answer_start = torch.argmax(outputs.start_logits) 
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))
        
        predictions.append(answer)

    return predictions

def main():
    
    args = create_arg_parser()
    batch = args.batch
    learning_rate = args.learning_rate
    

    model_args = f"batch_{batch}_lr_{learning_rate}" 

    if args.model =='albert':
        path_to_model = f'./Models/albert-custom/{model_args}'
        tokenizer = AlbertTokenizerFast.from_pretrained(path_to_model)
        model = AlbertForQuestionAnswering.from_pretrained(path_to_model)

    elif args.model == 'bert':
        path_to_model = f'./Models/distilBert-custom'
        tokenizer = DistilBertTokenizerFast.from_pretrained(path_to_model)
        model = DistilBertForQuestionAnswering.from_pretrained(path_to_model)




    test = load_dataset("GroNLP/ik-nlp-22_slp", "questions")['test']
    questions = list(test['question'])
    contexts = list(test['paragraph'])
    answers = list(test['answer'])

    test_encodings = tokenizer(contexts, questions, truncation=True, padding=True)

    test_dataset = DomainDataset(test_encodings)


    test_dataset = DataLoader(test_dataset, batch_size=1)

    predictions = get_predictions(model,tokenizer,test_dataset)


    resultsF1 = list()
    resultsEM = list()
    output = []
    for question, context, predict, true in zip(questions, contexts, predictions,answers):

        output.append([question, context, true, predict])
        f1 = evaluateF1(predict,true)
        em = evaluateEM(predict,true)
        resultsF1.append(f1)
        resultsEM.append(em)


    baselineF1 = sum(resultsF1)/len(resultsF1)
    baselineEM = sum(resultsEM)/len(resultsEM)

    print(f"F1: {baselineF1}, EM:{baselineEM}")

    df = pd.DataFrame(output,columns=['questions','contexts','answers','predictions'])
    if os.path.exists(path=path_to_model+"/output/"):
        df.to_json(path_to_model+"/output/output.json", orient='records')
    else:
        os.makedirs(path_to_model+"/output/")
        df.to_json(path_to_model+"/output/output.json", orient='records')

if __name__ == '__main__':
    main()





