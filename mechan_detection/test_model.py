import json
from transformers import AutoTokenizer, TextClassificationPipeline,BertForSequenceClassification
from sklearn.metrics import classification_report
import os
from tqdm import tqdm
import spacy
from transformers import AutoTokenizer


nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe('sentencizer')


def split_para_to_sents(text):
    doc = nlp(text)
    sents = []
    for sent_idx, sent in enumerate(doc.sents):
        sents.append(sent.text)
    return sents


def get_all_checkpoint(models_dir="./saved_dir/mechanism_detect"):
    for model_dir in os.listdir(models_dir):
        train_data_type = model_dir[20:]
        model_dir = os.path.join(models_dir, model_dir+"/model")
        for checkpoint_name in os.listdir(model_dir):
            if checkpoint_name.startswith("check"):
                yield os.path.join(model_dir, checkpoint_name), train_data_type


def prediction(model, tokenizer, data, text_field_name="text", sent_level=False, device=0):
    def format_convert(item_res):
        logits = [0, 0]
        for temp in item_res:
            logits[int(temp["label"][-1])] = temp["score"]

        return {
            "label": int(item_res[0]["label"][-1]),
            "logits": logits
        }

    pred_pipe = TextClassificationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=device

    )
    result = {}
    i = 0
    for paper in tqdm(data):
        abstract_text = paper[text_field_name]
        if sent_level:
            res = pred_pipe(split_para_to_sents(abstract_text),
                            function_to_apply="softmax", top_k=2)
            result[i] = [format_convert(sent_res) for sent_res in res]

        else:
            res = pred_pipe(
                abstract_text, function_to_apply="softmax", top_k=2)
            result[i] = format_convert(res)

        i += 1
    return result


def test_checkpoint(checkpoint_path="./", sent_level=False, text_field_name="text"):

    model = BertForSequenceClassification.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    test_data = json.load(
        open("./mechan_detection/data/test.json", "r")
    )

    result = prediction(model, tokenizer, test_data,
                        text_field_name=text_field_name, sent_level=sent_level, device=0)

    reals = []
    preds = []

    for ex_id, pred_res in tqdm(result.items(), desc=checkpoint_path.split("/")[-1]):
        real_label = test_data[ex_id]["label"]
        reals.append(real_label)
        if not sent_level:
            preds.append(pred_res["label"])
        else:
            _temp = [t["label"] for t in pred_res]
            if sum(_temp) > 0:
                preds.append(1)
            else:
                preds.append(0)
        test_data[ex_id]["pred"] = pred_res
    clf_rep = classification_report(reals, preds, digits=4)
    with open(checkpoint_path+"/classification_report.txt", "w") as f:
        f.write(clf_rep)
        print(clf_rep)
    json.dump(
        test_data,
        open(checkpoint_path+"/test_pred_res.json", "w"),
        indent=2,
    )
    metrics = classification_report(reals, preds, digits=4, output_dict=True)
    flat_metrics = {}
    for k, v in metrics.items():
        if type(v) == dict:
            for sub_k, sub_v in v.items():
                flat_metrics["{}-{}".format(k, sub_k)] = sub_v * 100
        flat_metrics["{}".format(k)] = sub_v * 100
    flat_metrics["checkpoint_path"] = checkpoint_path
    return flat_metrics


if __name__ == "__main__":

    data = []
    for checkpoint, data_type in tqdm(get_all_checkpoint()):
        if "balance" in data_type:
            temp = test_checkpoint(
                checkpoint
            )
            temp["train_data_type"] = data_type
            data.append(
                temp
            )
