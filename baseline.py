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
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input) # Your code here
    # Perform pooling
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    return embeddings



def get_context(model, tokenizer, query, contexts):
    #Encode query and contexts with the encode function
    query_emb = encode(model, tokenizer, query) # Your code here
    contexts_emb = encode(model, tokenizer, contexts) # Your code here

    #Compute dot score between query and all contexts embeddings
    scores = torch.mm(query_emb, contexts_emb.transpose(0, 1))[0].cpu().tolist()

    #Combine contexts & scores
    contexts_score_pairs = list(zip(contexts, scores))

    return max(contexts_score_pairs, key=lambda x: x[1])[0]




def verify(word):
    if word in string.punctuation:
        return False
    return True


def evaluateF1(predict,true):

    predict = [w for w in word_tokenize(predict) if verify(w)]
    true = [w for w in word_tokenize(true) if verify(w)]
    common = [w for w in true if w in predict]
    
    if len(common) == 0:
        return 0
    prec = len(common)/len(predict)
    recall = len(common)/len(true)
    
    return 2 * (prec*recall)/(prec+recall)

    
def evaluateEM(predict,true):

    predict = [w for w in word_tokenize(predict) if verify(w)]
    true = [w for w in word_tokenize(true) if verify(w)]
    
    
    if " ".join(predict) == " ".join(true):
        return 1
    else:
        return 0

def get_answer(model, query, context):
    res = model(query+"\n"+context)
    return res[0]['generated_text']




generator = pipeline("text2text-generation", model="allenai/unifiedqa-t5-large")

# Load the model and tokenizer from HuggingFace Hub 
model_name = 'sentence-transformers/all-mpnet-base-v2' #sentence-transformers/multi-qa-MiniLM-L6-cos-v1 sentence-transformers/all-mpnet-base-v2
tokenizer = AutoTokenizer.from_pretrained(model_name) # Your code here
model = AutoModel.from_pretrained(model_name) # Your code here

paragraphs = load_dataset("GroNLP/ik-nlp-22_slp", "paragraphs")['train']
questions = load_dataset("GroNLP/ik-nlp-22_slp", "questions")['test']

contexts = list(paragraphs['text']) #[:500]

resultsF1 = list()
resultsEM = list()
for question in questions:
    query = question['question']
    true = question['answer']
    best_context = get_context(model, tokenizer, query, contexts)
    predict = get_answer(generator, query, best_context)
    ### evaluate answers 
    f1 = evaluateF1(predict,true)
    em = evaluateEM(predict,true)

    resultsF1.append(f1)
    resultsEM.append(em)


baselineF1 = sum(resultsF1)/len(resultsF1)
baselineEM = sum(resultsEM)/len(resultsEM)


print(f"F1: {baselineF1}, EM:{baselineEM}")





