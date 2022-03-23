from lib2to3.pgen2 import token
import pandas as pd
from nltk.tokenize import word_tokenize
from english_words import english_words_set
import string

data = pd.read_csv('Data/slp3ed.csv')

data = data['text']


english_words_set.remove('log')
english_words_set.remove('lim')

cleaned_text = list()
removed = list()
for row in data:
    
    flag = False
    words = list()
    for w in word_tokenize(row):
        if w in english_words_set and w not in string.ascii_letters:
            flag = True
            words.append(w)

    if flag==False or len(words)<=8:
        removed.append(row)
    else:
        cleaned_text.append(row)
    


print(f"original: {(data.shape)[0]}")
print(f"cleaned: {len(cleaned_text)}")
print(f"removed: {len(removed)}")

with open('Data/clean-contexts.txt','w') as file:
    for context in cleaned_text:
        file.write(context+'\n')

with open('Data/bad-contexts.txt','w') as file:
    for context in removed:
        file.write(context+'\n')

