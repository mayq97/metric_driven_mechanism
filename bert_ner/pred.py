import json
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer, TokenClassificationPipeline, AutoConfig


tag2id_path = "./bert_ner/tag2id.json"
tag2id = json.load(open(tag2id_path, "r"))
model_path = "./saved_dir/bertner/2022-07-31 10_46_41/checkpoint-1287"
tokenizer = AutoTokenizer.from_pretrained(model_path)
config = AutoConfig.from_pretrained(
    "./saved_dir/bertner/2022-07-31 10_46_41/checkpoint-1287/config.json")
model = AutoModelForTokenClassification.from_config(config)

pipe = TokenClassificationPipeline(
    model=model, tokenizer=tokenizer, device=4, grouped_entities=True, aggregation_strategy="simple")


test_data = [json.loads(line) for line in open(
    "./mechan_ent_rel_ext/data/test.json", "r")]
for ex in test_data:
    res = pipe(ex["text"], grouped_entities=True)
    pipe.aggregate_words(res, "simple")
    print()
