import nltk
import numpy as np
from mechan_ent_rel_ext.config import cfg
from transformers import AutoTokenizer
from datasets import load_metric
from sklearn.metrics import classification_report
from itertools import product

tokenizer = AutoTokenizer.from_pretrained(
    cfg.pretrain_model_path, use_fast=True)

rouge_metric = load_metric("mechan_ent_rel_ext/rouge.py")

query2label = {
    "Which are the general terms or pronouns?": "Generic",
    "Which are the applications, problems to be solved, systems to construct?": "Task",
    "Which are the methods , models, systems to be used, or tools, components of a system, frameworks?": "Method",
    "Which are the metrics, measures, or entities that can express quality of a system/method?": "Metric",
    "Which are the data, datasets, resources, corpus, knowledge base": "Material",
    "which are the phrases that are a scientific terms but not above classes?": "OtherScientificTerm",
    "What is the measurable and comparable metric entity ?": "Effect"
}

def relax_match(p_str, r_str):
    if r_str == "non entity":
        if r_str == p_str:
            return 1
        else:
            return 0
    if r_str == p_str:
        return 1
    else:
        overlap_1 = len(set(r_str.split(" ")) & set(
            p_str.split(" "))) / len(set(r_str.split(" ")))
        overlap_2 = len(set(r_str.split(" ")) & set(
            p_str.split(" "))) / len(set(p_str.split(" ")))

        overlap = (overlap_2 + overlap_1)/2
        if overlap > 0.9:
            return 1
        else:
            return 0


class F1_Score:
    def __init__(self):
        self.eff_p = []
        self.eff_r = []
        self.op_p = []
        self.op_r = []
        self.rel_p = {
            "Affect": [],
            "Neg_Affect": [],
            "Pos_Affect": [],
            "Total": []
        }
        self.rel_r = {
            "Affect": [],
            "Neg_Affect": [],
            "Pos_Affect": [],
            "Total": []
        }

    def relax_match(self, pred, real):
        temp = set()
        overlap_num = 0

        for p_str, r_str in product(pred, real):
            res = relax_match(p_str, r_str)

            if res == 1 and p_str not in temp:
                overlap_num += 1
                temp.add(p_str)
        return overlap_num

    def match(self, pred, real):
        try:
            temp = len(pred & real)
        except Exception as e:
            print(pred, real)
        return temp

    def add_ent(self, pred, real, ent_type):
        _p = self.relax_match(pred, real)/len(pred)
        _r = self.relax_match(pred, real)/len(real)

        if ent_type == "Effect":
            self.eff_p.append(_p)
            self.eff_r.append(_r)
        elif ent_type == "Operation":
            self.op_p.append(_p)
            self.op_r.append(_r)

    def add_rel(self, pred, real, rel_type):
        _p = self.relax_match(pred, real)/len(pred)
        _r = self.relax_match(pred, real)/len(real)
        
        self.rel_p[rel_type].append(_p)
        self.rel_r[rel_type].append(_r)

        self.rel_p["Total"].append(_p)
        self.rel_r["Total"].append(_r)

    def get_f1_score(self):
        eff_p = sum(self.eff_p)/len(self.eff_p)
        eff_r = sum(self.eff_r)/len(self.eff_r)

        op_p = sum(self.op_p)/len(self.op_p)
        op_r = sum(self.op_r)/len(self.op_r)

        p = sum(self.eff_p+self.op_p)/len(self.eff_p+self.op_p)
        r = sum(self.eff_r+self.op_r)/len(self.eff_r+self.op_r)

        temp_1 = eff_r+eff_p
        if temp_1 == 0:
            eff_f1 = 0
        else:
            eff_f1 = 2*(eff_p*eff_r)/temp_1

        temp_2 = op_r+op_p
        if temp_2 == 0:
            op_f1 = 0
        else:
            op_f1 = 2*(op_r*op_p)/temp_2

        temp_3 = p+r
        if temp_3 == 0:
            f1 = 0
        else:
            f1 = 2*p*r/temp_3

        return {
            "eff_p": eff_p,
            "eff_r": eff_r,
            "eff_f1": eff_f1,
            "op_p": op_p,
            "op_r": op_r,
            "op_f1": op_f1,
            "p": p,
            "r": r,
            "f1": f1
        }

    def get_rel_f1_score(self,):
        result = {}

        for k, temp_list in self.rel_p.items():
            result[k+"_p"] = sum(temp_list)/len(temp_list)
        for k, temp_list in self.rel_r.items():
            result[k+"_r"] = sum(temp_list)/len(temp_list)
        for k in self.rel_p.keys():
            div = result[k+"_r"]+result[k+"_p"]
            if div > 0:
                result[k+"_f1"] = 2*(result[k+"_r"]*result[k+"_p"]) / \
                    (result[k+"_r"]+result[k+"_p"])
            else:
                result[k+"_f1"] = 0
        return result


def compute_metrics(eval_pred):
    metric_dict = {}
    predictions, labels, inputs = eval_pred
    decoded_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                     for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip()))
                      for label in decoded_labels]

    lookup = [0, 2, 3, 4, 5]
    for idx in lookup:
        texts = tokenizer.decode(
            [t for t in inputs[idx] if t > 1], clean_up_tokenization_spaces=True)
        print("-*"*15)
        if cfg.model_type == "t5" or cfg.model_type == "bart":
            query, text = texts.split("[SEP]")
        elif cfg.model_type == "t5-v2":
            query, text = texts.split("context:")

        print("{} Text :{}".format(idx, text))
        label = "Operation"
        for k, v in query2label.items():
            if query in k:
                label = v
        print("Query  : {}".format(query))
        print("{} Real:{}".format(label, decoded_labels[idx]))
        print("{} Pred:{}".format(label, decoded_preds[idx]))

    rel_r = []
    rel_p = []
    _real_ents_text_list = []
    _pred_ents_text_list = []

    for idx, (p, r) in enumerate(zip(decoded_preds, decoded_labels)):
        texts = tokenizer.decode(
            [t for t in inputs[idx] if t > 1], clean_up_tokenization_spaces=True)
        if cfg.model_type == "t5" or cfg.model_type == "bart":
            query, text = texts.split("[SEP]")

        if not "measurable" in query[:40]:
            if r == "non entity" and p == "non entity":
                continue
            if r == "non entity":
                rel_r.append(0)
            else:
                rel_r.append(1)
                _real_ents_text_list.append(r)
                _pred_ents_text_list.append(p)

            if "non entity" in p:
                rel_p.append(0)
            else:
                rel_p.append(1)

    rel_metrics = classification_report(
        rel_r, rel_p, digits=4, output_dict=True)
    metric_dict["rel_macro_f1"] = rel_metrics["macro avg"]["f1-score"]*100
    metric_dict["rel_pos_f1"] = rel_metrics["1"]["f1-score"]*100
    metric_dict["rel_neg_f1"] = rel_metrics["0"]["f1-score"]*100
    metric_dict["rel_accuracy"] = rel_metrics["accuracy"]*100

    part_rouge_metrics = rouge_metric.compute(
        predictions=_pred_ents_text_list, references=_real_ents_text_list, use_stemmer=True)
    total_rouge_metrics = rouge_metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    metric_dict.update({"pure_"+key: value.mid.fmeasure * 100 for key,
                        value in part_rouge_metrics.items()})

    metric_dict.update({key: value.mid.fmeasure * 100 for key,
                        value in total_rouge_metrics.items()})

    metric_dict["total_score"] = (metric_dict["pure_rouge1"] +
                                  metric_dict["pure_rouge2"] + metric_dict["rel_macro_f1"])/3

    # Add mean generated length
    prediction_lens = [np.count_nonzero(
        pred != tokenizer.pad_token_id) for pred in predictions]
    metric_dict["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in metric_dict.items()}
