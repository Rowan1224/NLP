
### Table of Contents

1. [Project Motivation](#motivation)
2. [File Descriptions](#files)

## Installation <a name="installation"></a>

The libraries required for the successful execution of this code are mentioned in requirements.txt. In order to install all the libraries:
`pip install -r requirements.txt`

## File Descriptions <a name="files"></a>

To start with pre-processing and removing the bad contexts, we need to run the `clean_data.py` file which will generate the `./data/cleaned_contexts.txt` and also store the test data:
```python clean_data.py```

Then we can already run the baselines as well which include the following three models:
- Just a semantic search model which returns the best context
- A semantic search model in addition to the text2text generation model
- A question answering model with the semantic search model

Two baselines are calculated with one using `faiss`. To run the baseline, run either `python baseline.py` or `python baseline_faiss.py`

Then we can generate the synthetic question answer pairs for fine-tuning purposes. To do this, run `python qas_generate.py` which generates the question answer pairs along with the contexts and store it as `./data/synthetic_qa_pairs.json`

Finally, we can fine-tune the model using three different models:
- Distilbert: `python fine_tune.py -m bert`
- Albert: `python fine_tune.py -m albert`
- Electra `python fine_tune.py -m electra`

We can also customize the batch size and learning rate here using parameters `-b` and `-lr`.

This will store the fine-tuned models inside the `./models/` directory. Finally, we can use `evaluate.py` in the similar way:
- Distilbert: `python evaluate.py -m bert`
- Albert: `python evaluate.py -m albert`
- Electra `python evaluate.py -m electra`
