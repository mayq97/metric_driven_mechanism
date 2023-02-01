import sys
from typing import List
sys.path.append(".")

from task_ext.model_for_task_clf import BaseMultiLabelClf
import json
from tqdm import tqdm
import pickle
from myLabTools.text_classification.predict_on_raw_data import PredictorForPairTextClf



def test_model(model_path):

    test_data = [json.loads(l) for l in open(
        "./pwc_tasks/raw_test_data_5000.json", "r")]
    id2label_path = model_path + "/id2label.json"
    id2label = {}
    for k, v in json.load(open(id2label_path, "r")).items():
        id2label[int(k)] = v

    predictor = PredictorForPairTextClf(
        predictor_config={"model_checkpoint": model_path,
                          "device": "cuda:2", "max_length": 512},
        id2label=id2label,
        ClfModelClass=BaseMultiLabelClf
    )

    preds = []

    for paper in tqdm(test_data):
        pred_result = predictor.predict(paper["title"], paper["abstract"])
        temp = []
        for label_name, prob in pred_result["logits"].items():
            if prob > 0.5:
                temp.append({
                    "label": label_name,
                    "prob": prob
                })
        paper["pred"] = temp
    pickle.dump(
        test_data,
        open(
            "./pwc_tasks/raw_test_data_5000.json.pred.pkl", "wb"
        )
    )


def compute_p_r_f1(test_result, type="all"):
    precision = []
    recall = []
    cnt = 0
    for paper in test_result:
        if type == "all" or \
            (type == "high" and "high" in paper["label_freq_class_list"]) or \
            (type == "mid" and "mid" in paper["label_freq_class_list"]) or \
            (type == "low" and (
                "low" in paper["label_freq_class_list"] or "exlow" in paper["label_freq_class_list"])):

            pred_tasks = set([p["label"] for p in paper["pred"]])
            real_tasks = set(paper["label_name_list"])
            overlap = pred_tasks & real_tasks
            precision.append(
                len(overlap) / len(pred_tasks) if len(pred_tasks) > 0 else 0
            )
            recall.append(
                len(overlap) / len(real_tasks)
            )
            cnt += 1

    p = round(sum(precision)/len(precision),3)*100
    r = round(sum(recall)/len(recall),3)*100
    f_1 = round(2*p*r/(p+r),1)
    print("*"*10, type, "*"*10)
    print("p:", p)
    print("r:", r)
    print("f1:", f_1)
    print("cnt:", cnt)
    print()
    print(p,r,f_1)

def compute_metrics():
    test_result = pickle.load(open(
        "./pwc_tasks/raw_test_data_5000.json.pred.pkl", "rb"

    ))
    compute_p_r_f1(test_result, type="all")
    compute_p_r_f1(test_result, type="high")
    compute_p_r_f1(test_result, type="mid")
    compute_p_r_f1(test_result, type="low")
    # compute_p_r_f1(test_result, type="exlow")


if __name__ == '__main__':
    # test_model("./saved_dir/task_extraction")
    compute_metrics()
