
### Table of Contents

1. [Project Motivation](#motivation)
2. [File Descriptions](#files)

## Installation <a name="installation"></a>

The libraries required for the successful execution of this code are mentioned in requirements.txt. In order to install all the libraries:
`pip install -r requirements.txt`

## File Descriptions <a name="files"></a>

To start with pre-processing and removing the bad contexts, we need to run the `clean_data.py` file which will generate the `./data/cleaned_contexts.txt` and also store the test data:
```python clean_data.py```


Then we can generate the synthetic question answer pairs for fine-tuning purposes. To do this, run `python qas_generate.py` which generates the question answer pairs along with the contexts and store it as `./data/synthetic_qa_pairs.json`

To run the baseline models, execute:
- `python baseline.py`



Finally, we can fine-tune the model using three different models:
- Distilbert: `python fine_tune.py -m bert -t fine -ts`
- Albert: `python fine_tune.py -m albert -t squad -ts`
- Electra `python fine_tune.py -m electra -t fine -ts`

We can also customize the batch size and learning rate here using parameters `-b` and `-lr`. The argument `-t` refers to the type of the model e.g `fine` is the base pre-trained model and `squad` is the squad model. Finally, `-ts` should be given if the model is to be trained on the full training set.

This will store the fine-tuned models inside the `./models/` directory. Finally, we can use `evaluate.py` in the similar way:
- Distilbert: `python evaluate.py -m bert -t fine -path [saved model path]`
- Albert: `python evaluate.py -m albert -t squad -path [saved model path]`
- Electra `python evaluate.py -m electra -t fine -path [saved model path]`

`-path` argument is optional, by defualt this evaluate the saved models uploaded in hugginface hub

To run the semantic search models, execute:
- `python semantic_search.py` (cosine similarity)
- `python semantic_search.py -f` (faiss)

The code returns the results of following three models:
- Just a semantic search model which returns the best context
- Best Question answering models with the semantic search approach

