from torch.utils.data import DataLoader
from helpers import DomainDataset, compute_f1, load_json, compute_em, model_name_to_class, save_answers
import evaluate



for k,v in model_name_to_class.items():


    model, tokenizer, model_name = v.values()

    test_qac = load_json("./data/test_qac.json")
    test_questions = [pair["question"] for pair in test_qac]
    test_answers = [pair["answer"] for pair in test_qac]
    test_contexts = [pair["context"] for pair in test_qac]

    tokenizer = tokenizer.from_pretrained(model_name)
    model = model.from_pretrained(model_name)
    
    test_encodings = tokenizer(
        test_contexts, test_questions, truncation=True, padding=True
    )
    test_dataset = DomainDataset(test_encodings)
    test_dataset = DataLoader(test_dataset, batch_size=1)

    predictions = evaluate.get_predictions(model, tokenizer, test_dataset)

    f1 = compute_f1(test_answers, predictions)
    em = compute_em(test_answers, predictions)

    print(f"{k}")
    print(f"F1: {f1}")
    print(f"EM: {em}")

    save_answers(test_questions,test_contexts,test_answers,predictions,f"{k}_baseline")

