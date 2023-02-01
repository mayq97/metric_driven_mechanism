from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
from tqdm import tqdm
import json
from transformers.pipelines import SummarizationPipeline
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

model_checkpoint = "./saved_dir/bart_ner/from_scierc/mechan_2022-08-03 13_27_27_with_0.5_empty/model/checkpoint-9711"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
config = AutoConfig.from_pretrained(model_checkpoint, use_fast=True)


model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model.eval()
pipline = SummarizationPipeline(model=model, tokenizer=tokenizer, device=0)

# test_dataset = BARTNERDataset(
#         json_path="./data/标注结果/机制实体关系抽取/train_data/test.json",
#         tokenizer=tokenizer,
#         max_length=512,
#     )

data = json.load(
    open("./spider/parse_result/2020.json", "r")
)

rel_map = {
    "Affect": "Which entities have Affect on {} but direction unknown ?",
    "Pos_Affect": "Which entities have Positive Affect on {} ?",
    "Neg_Affect": "Which entities have Negative Affect on {} ?"
}
eff_query = "What is the measurable and comparable metric entity ?"

error_data = []
for paper in tqdm(data):

    input_text = eff_query + "[SEP]" + paper["text"]

    pred_text = pipline(
        input_text, min_length=5, max_length=40,
    )[0]

    paper["pred_effect"] = pred_text["summary_text"]
    if not "non entity" in paper["pred_effect"]:
        mechanism = []
        for eff in paper["pred_effect"].split(" <> "):

            for rel, query in rel_map.items():

                input_text = query.format(eff) + "[SEP]" + paper["text"]
                operation_text = pipline(
                    input_text, min_length=5, max_length=40,
                )[0]["summary_text"]

                if not "non entity" in operation_text:
                    for op in operation_text.split(" <> "):
                        mechanism.append(
                            [eff, rel, op]
                        )
        paper["mechanism"] = mechanism
    else:
        paper["mechanism"] = []


json.dump(
    data,
    open(
        "./data/bart_output"+"/acl-2021-2022.json",
        "w", encoding="utf-8"
    ),
    indent=2
)
