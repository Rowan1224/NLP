
### Table of Contents

1. [Installation](#motivation)
2. [File Descriptions](#files)

## Installation <a name="installation"></a>

The libraries required for the successful execution of this code are mentioned in requirements.txt. In order to install all the libraries:
`pip install -r requirements.txt`

## File Descriptions <a name="files"></a>

To start with pre-processing and removing the bad contexts, please run `clean_data.py` file to generate a cleaned contexts file `./data/cleaned_contexts.txt` and to store the test data in json file `./data/test_qac.json`:

- ```python clean_data.py```


Next, we generate the synthetic question answer pairs for fine-tuning/domain adaptation purposes. To do this, run `python qas_generate.py` which generates the question answer pairs along with the contexts and store it as `./data/synthetic_qa_pairs.json`

To run the baseline models, execute:
- `python baseline.py`

To fine-tune/domain-adapt the models, use the sample commandos as bellow:
- Distilbert: `python fine_tune.py -m bert -t fine -ts`
- Albert: `python fine_tune.py -m albert -t squad -ts`
- Electra `python fine_tune.py -m electra -t fine -ts`

To customize the batch size and learning rate, use parameters `-b` and `-lr`. The argument `-t` refers to the type of the model e.g `fine` is fine-tuning the base pre-trained model and `squad` is domain-adapting for the squad model. The argument `-ts` should be given if the model is to be trained on the full training set.

The fine-tuned/domain-adapted models will be stored in the `./models/` directory. 

To evalute the QA models, use `evaluate.py`, use the sample commandos as bellow:
- Distilbert: `python evaluate.py -m bert -t fine -path [saved model path]`
- Albert: `python evaluate.py -m albert -t squad -path [saved model path]`
- Electra `python evaluate.py -m electra -t fine -path [saved model path]`

Notice: `-path` argument is optional, by defualt this script evaluate the saved models uploaded in hugginface hub (https://huggingface.co/rowan1224).

To run the semantic search models, execute:
- `python semantic_search.py` (cosine similarity)
- `python semantic_search.py -f` (faiss)

The script returns the results of following three models:
- Just a semantic search model which returns the best context
- Best Question answering models with the semantic search approach

