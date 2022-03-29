#!/usr/bin/env python
# coding: utf-8
###


from re import L
from nltk.tokenize import word_tokenize
import string
from datasets import load_dataset
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers import pipeline
import pandas as pd
import os
from tqdm import tqdm
from transformers.data.metrics import squad_metrics
import argparse

def create_arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--with_context", action='store_true',
                    help='Select evaluation type')

    args = parser.parse_args()
    return args


def dot_score(a: Tensor, b: Tensor):
    """
    Computes the dot-product dot_prod(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = dot_prod(a[i], b[j])
    Taken from the SentenceTransformer library
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    print(a.shape, b.shape)
    # Compute the dot-product
    return torch.mm(a, b.transpose(0, 1))


#Mean Pooling - Average all the embeddings produced by the model
def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output.last_hidden_state
    # Expand the mask to the same size as the token embeddings to avoid indexing errors
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # Compute the mean of the token embeddings
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


#Encode text
def encode(model, tokenizer, texts):

    # Tokenize sentences
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt") # Your code here

    encoded_input = encoded_input.to(device)
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input) # Your code here
    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings.cpu()



def get_context(model, tokenizer, query, contexts_emb):
    #Encode query and contexts with the encode function
    query_emb = encode(model, tokenizer, query) # Your code here
    

    #Compute dot score between query and all contexts embeddings
    scores = torch.mm(query_emb, contexts_emb.transpose(0, 1))[0].cpu().tolist()

    #Combine contexts & scores
    contexts_score_pairs = list(zip(contexts, scores))

    return max(contexts_score_pairs, key=lambda x: x[1])[0]



def get_answer(model, query, context):
    res = model(query+"\n"+context)
    return res[0]['generated_text']

def encode_contexts(contexts):

    contexts_enc = list()
    loop = tqdm(contexts, leave=True)
    for context in loop:
        encoding = encode(model, tokenizer, context)
        contexts_enc.append(encoding) # Your code here

    contexts_emb = torch.Tensor(len(contexts),768)
    return torch.cat(contexts_enc,out=contexts_emb)


generator = pipeline("text2text-generation", model="allenai/unifiedqa-t5-large")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# move model over to detected device


# Load the model and tokenizer from HuggingFace Hub 
model_name = 'sentence-transformers/all-mpnet-base-v2' #sentence-transformers/multi-qa-MiniLM-L6-cos-v1 sentence-transformers/all-mpnet-base-v2
tokenizer = AutoTokenizer.from_pretrained(model_name) # Your code here
model = AutoModel.from_pretrained(model_name) # Your code here
model.to(device)


paragraphs = load_dataset("GroNLP/ik-nlp-22_slp", "paragraphs")['train']
questions = load_dataset("GroNLP/ik-nlp-22_slp", "questions")['test']


contexts = list()
with open("Data/clean-contexts.txt",'r') as file:
    for line in file:
        contexts.append(line.replace('\n',''))

args = create_arg_parser()

if not args.with_context: 
    contexts_emb = encode_contexts(contexts)

output = []
blank_answers = []

resultsF1 = list()
resultsEM = list()

loop = tqdm(questions, leave=True)

for question in loop:
    query = question['question']
    true = question['answer']

    if args.with_context:
        best_context = question['paragraph']
    else:
        best_context = get_context(model, tokenizer, query, contexts_emb)

    predict = get_answer(generator, query, best_context)

    output.append([query, best_context, true, predict])

    if predict=="":
        blank_answers.append([query, best_context, true, predict])

    resultsF1.append(squad_metrics.compute_f1(true,predict))
    resultsEM.append(squad_metrics.compute_exact(true,predict))


baselineF1 = sum(resultsF1)/len(resultsF1)
baselineEM = sum(resultsEM)/len(resultsEM)

df = pd.DataFrame(output,columns=['questions','contexts','answers','predictions'])
df2 = pd.DataFrame(blank_answers,columns=['questions','contexts','answers','predictions'])


if os.path.exists(path="Models/baseline/output/"):
    df.to_json("Models/baseline/output/output.json", orient='records')
    df2.to_json("Models/baseline/output/blank.json", orient='records')
else:
    os.makedirs("Models/baseline/output/")
    df.to_json("Models/baseline/output/output.json", orient='records')
    df2.to_json("Models/baseline/output/blank.json", orient='records')


print(f"F1: {baselineF1}, EM:{baselineEM}")





