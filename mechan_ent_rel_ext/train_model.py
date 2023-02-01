import copy
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments

from utils.S2SNerTrainer import Seq2SeqNERTrainer
from mechan_ent_rel_ext.part_metrics import compute_metrics
from mechan_ent_rel_ext.seq2seq_ner_dataset import S2SNERDataset, SCIERCRDataset
from mechan_ent_rel_ext.config import cfg, S2SNERConfig

from utils.util import convert_config_to_dict
from datetime import datetime
from tensorboardX import SummaryWriter
import json
import pickle
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def load_dataset(train_config, data_type, neg_sampleing_prob):
    tokenizer = AutoTokenizer.from_pretrained(train_config.pretrain_model_path)

    if data_type == "scierc":
        train_config.data_dir = "./mechan_ent_rel_ext/scierc"
        train_dataset = SCIERCRDataset(
            json_path=train_config.data_dir + "/train.json",
            tokenizer=tokenizer,
            max_length=train_config.max_length,
            desc="train",
            model_type=train_config.model_type
        )

        dev_dataset = SCIERCRDataset(
            json_path=train_config.data_dir + "/dev.json",
            tokenizer=tokenizer,
            max_length=train_config.max_length,
            desc="dev",
            model_type=train_config.model_type
        )

        test_dataset = SCIERCRDataset(
            json_path=train_config.data_dir + "/test.json",
            tokenizer=tokenizer,
            max_length=train_config.max_length,
            desc="test",
            model_type=train_config.model_type
        )
    elif data_type == "mechan":
        train_dataset = S2SNERDataset(
            json_path=train_config.data_dir + "/train.json",
            tokenizer=tokenizer,
            max_length=train_config.max_length,
            train_sampleing_prob=neg_sampleing_prob,
            desc="train",
            model_type=train_config.model_type
        )

        dev_dataset = S2SNERDataset(
            json_path=train_config.data_dir + "/dev.json",
            tokenizer=tokenizer,
            max_length=train_config.max_length,
            desc="dev",
            model_type=train_config.model_type

        )

        test_dataset = S2SNERDataset(
            json_path=train_config.data_dir + "/test.json",
            tokenizer=tokenizer,
            max_length=train_config.max_length,
            desc="test",
            model_type=train_config.model_type

        )
    return train_dataset, dev_dataset, test_dataset


def main(neg_sampleing_prob=0.1, train_config: S2SNERConfig = None, data_type="scierc", ):
    TIMESTAMP = "{0:%Y-%m-%d %H_%M_%S}".format(datetime.now())

    train_config.saved_path = train_config.saved_path + "/" + data_type + \
        "_" + TIMESTAMP+"_with_{}_empty".format(neg_sampleing_prob)


    train_dataset, dev_dataset, test_dataset = load_dataset(
        train_config, data_type, neg_sampleing_prob)

    config_data = convert_config_to_dict(train_config)
    os.mkdir(train_config.saved_path)
    json.dump(config_data, open("{}/config.json".format(train_config.saved_path),"w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # 模型加载
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.pretrain_model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        train_config.pretrain_model_path, use_fast=True)

    # 参数设置
    training_args = Seq2SeqTrainingArguments(
        train_config.saved_path + "/model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        logging_steps=10,
        logging_dir=train_config.saved_path + "/logs",
        warmup_ratio=0.2,
        per_device_train_batch_size=train_config.train_batch_size,
        per_device_eval_batch_size=train_config.dev_batch_size,
        num_train_epochs=train_config.num_train_epochs,

        save_strategy="epoch",
        report_to="tensorboard",
        generation_max_length=train_config.generation_max_length,
        weight_decay=0.01,
        save_total_limit=train_config.max_keep_ckpt,
        gradient_accumulation_steps=1,
        predict_with_generate=True,
        include_inputs_for_metrics=True,
        metric_for_best_model="eval_rouge2",
    )
    training_args.__setattr__("min_length", train_config.generation_min_length)
    # training_args.__setattr__("debug","underflow_overflow")
    # training_args.__setattr__("use_legacy_prediction_loop",True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqNERTrainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.evaluate()
    trainer.save_model()

    test_result = trainer.predict(test_dataset)
    with open(train_config.saved_path + "/model/test_result-neg_sampleing_prob-{}.txt".format(neg_sampleing_prob), "w") as f:
        f.write(
            json.dumps(test_result.metrics, indent=2)
        )
    pickle.dump(
        test_result, open(
            train_config.saved_path + "/model/test_result.pkl", "wb"
        )
    )


if __name__ == "__main__":

    train_config = copy.deepcopy(cfg)
    main(0, train_config, data_type="mechan")
