import json
import pandas as pd
import glob
import os
from mechan_ent_rel_ext.part_metrics import F1_Score
from transformers import AutoTokenizer
import numpy as np
import nltk
import pickle
from mechan_ent_rel_ext.seq2seq_ner_dataset import S2SNERDataset



tokenizer = AutoTokenizer.from_pretrained("saved_dir/t5_ner/only_scierc/scierc_2022-10-24 t5/model")
model_type = "t5"
query2label = {
    "Which are the general terms or pronouns?": "Generic",
    "Which are the applications, problems to be solved, systems to construct?": "Task",
    "Which are the methods , models, systems to be used, or tools, components of a system, frameworks?": "Method",
    "Which are the metrics, measures, or entities that can express quality of a system/method?": "Metric",
    "Which are the data, datasets, resources, corpus, knowledge base": "Material",
    "which are the phrases that are a scientific terms but not above classes?": "OtherScientificTerm",
    "What is the measurable and comparable metric entity ?": "Effect"
}


def get_gold_test(test_dataset_path):
    test_dataset = S2SNERDataset(
        json_path=test_dataset_path,
        tokenizer=tokenizer,
        max_length=1000,
        desc="test",
        model_type=model_type
    )
    test_data_v2 = []

    for i, ex in enumerate(test_dataset):
        test_data_v2.append(
            {
                "idx": i,
                "context": ex["context"],
                "query": ex["query"],
                "label": ex["labels_text"],
                "ent_type": query2label.get(ex["query"], "Operation"),
                "rel_type": ex["rel_type"],
            }
        )
    return test_data_v2


def get_raw_test_dataset_path(checkpoint_path):
    if "from_scratch" in checkpoint_path or "from_scierc" in checkpoint_path:
        return "./mechan_ent_rel_ext/data/test.json"
    elif "only_scierc" in checkpoint_path:
        return "./mechan_ent_rel_ext/scierc/test.json"
        


def get_test_pred(checkpoint_path):
    result_path = checkpoint_path + "/test_result/test_result.pkl"

    text_result = pickle.load(open(result_path, "rb"))
    predictions, labels, rouge_metrics = text_result
    decoded_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True
        )
    # Replace -100 in the labels as we can't decode them.

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                     for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip()))
                      for label in decoded_labels]

    ref_data = get_gold_test(get_raw_test_dataset_path(checkpoint_path))
    if model_type == "bart":
        sep_word = " <> "
    elif  model_type == "t5":
        sep_word = " >> "

    for i, (ex, pred) in enumerate(zip(ref_data, decoded_preds)):

        if ex["label"] == set(decoded_labels[i].split(sep_word)):
            ex["pred"] = set(pred.split(sep_word))
        else:
            print(ex["label"], pred)

    return ref_data, rouge_metrics


def compute_checkpoint_f1_score(checkpoint_path,include_non=True):
    print(checkpoint_path)
    test_data, rouge_metrics = get_test_pred(checkpoint_path)
    ent_metric = F1_Score()
    for ex in test_data:

        if "non entity" in ex["label"] and not include_non:
            continue

        ent_metric.add_ent(ex["pred"], ex["label"], ex["ent_type"])
        if "Effect" != ex["ent_type"]:
            ent_metric.add_rel(ex["pred"], ex["label"], ex["rel_type"])

    if save_test_result:
        for ex in test_data:
            ex["pred"] = list(ex["pred"])
            ex["label"] = list(ex["label"])
        json.dump(
            test_data, open(checkpoint_path +
                            "/test_prediction.json", "w", encoding="utf-8"),
            indent=2, ensure_ascii=False
        )

    f1_score = ent_metric.get_f1_score()

    f1_score.update(rouge_metrics)

    rel_f1_score = ent_metric.get_rel_f1_score()
    f1_score.update(rel_f1_score)

    return f1_score


def get_all_models_dir_path():
    train_type = [
        "saved_dir/bart_ner/from_scierc/*",
        "saved_dir/t5_ner/from_scierc/mechan_*",
        "saved_dir/t5_ner/from_scratch/mechan_*"
    ]
    result = []

    for models_dir in train_type:
        for  model_dir in glob.glob(models_dir):
            result.append(
                [model_dir + "/model", float(model_dir.split("_")[-2])]
            )

    for model_dir in glob.glob("saved_dir/bart_ner/from_scratch/*"):
        result.append(
            [model_dir + "/model", float(model_dir.split("_")[-1].split("-")[0])]
            )
    return result


def get_model_checkpoint(model_path):
    for checkpoint in os.listdir(model_path):
        if checkpoint.startswith("checkpoint"):
            yield model_path + "/" + checkpoint


def compute_f1(include_non = False):
    _train_result = []

    for model, neg_value in get_all_models_dir_path():
        for checkpoint in get_model_checkpoint(model):
            res = {}
            res["neg_value"] = neg_value
            res["checkpoint"] = checkpoint
            res["with_scierc"] = "Y" if "from_scierc" in checkpoint else  "N"
            res["backbone"] = "bart" if "bart_ner" in checkpoint else  "t5"


            res.update(compute_checkpoint_f1_score(checkpoint,include_non))
            _train_result.append(res)

    return _train_result




if __name__ == "__main__":
    save_test_result = True

    with pd.ExcelWriter('./result_ana/ana-output-1025.xlsx') as writer:
        post_fix_1 = "relax" 
        post_fix_2 =  "without_non"
    
        _train_result = compute_f1(include_non=False)
        
        df_1 = pd.DataFrame(_train_result)
        df_1.to_excel(writer, sheet_name='full_{}_{}'.format(
            post_fix_1, post_fix_2))

        