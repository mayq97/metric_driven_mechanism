# 指标函数
from curses import noecho

from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report
import json

from datasets import load_metric
# metric = load_metric("seqeval")
tag2id_path ="bert_ner/BIO_tag2id.json"
tag2id = json.load(open(tag2id_path,"r"))
id2tag ={}
for k,v in tag2id.items():
    id2tag[v] = k

def compute_metrics(eval_pred):
    predictions, references = eval_pred
    predictions= predictions.argmax(-1)
    batch_size = predictions.shape[0]
    pred_list = []
    real_list = []
    for idx in range(batch_size):
        pred = []
        real = []
        for p,r in zip(predictions[idx],references[idx]):
            if  r == -100:
                continue
            else:
                pred.append(id2tag[p])
                real.append(id2tag[r])

        pred_list.append(pred)
        real_list.append(real)
    cl_report = classification_report(real_list,pred_list,digits=4)
    print(real_list[0])
    print(pred_list[0])
    print(cl_report)
    result = classification_report(real_list,pred_list,output_dict=True,digits=4)
    flatten_result = {}
    for k,v in result.items():
        for sub_k ,sub_v in v.items():
            flatten_result["{}_{}".format(k.replace(" ","-"),sub_k)] = sub_v
    return flatten_result