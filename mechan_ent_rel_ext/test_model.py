import copy
from mechan_ent_rel_ext.config import cfg
from mechan_ent_rel_ext.seq2seq_ner_dataset import S2SNERDataset, SCIERCRDataset

from utils.S2SNerTrainer import Seq2SeqNERTrainer
from mechan_ent_rel_ext.part_metrics import compute_metrics
from glob import glob
import json
import pickle
import os

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def test_model(test_config,):
    # config缓存
    test_config.saved_path = test_config.saved_path + "/test_result"

    if not os.path.exists(test_config.saved_path):
        os.mkdir(test_config.saved_path)

    tokenizer = AutoTokenizer.from_pretrained(test_config.pretrain_model_path)

    if "only_scierc" in test_config.pretrain_model_path:
        test_dataset = SCIERCRDataset(
            json_path="mechan_ent_rel_ext/scierc/test.json",
            tokenizer=tokenizer,
            max_length=test_config.max_length,
            desc="test"
        )
    else:
        test_dataset = S2SNERDataset(
            json_path=test_config.data_dir + "/test.json",
            tokenizer=tokenizer,
            max_length=test_config.max_length,
            desc="test"
        )

    # 模型加载
    # base line
    model = AutoModelForSeq2SeqLM.from_pretrained(
        test_config.pretrain_model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        test_config.pretrain_model_path, use_fast=True)

    # 参数设置
    training_args = Seq2SeqTrainingArguments(
        test_config.saved_path + "/model",
        do_predict=True,
        resume_from_checkpoint=True,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        logging_steps=10,
        logging_dir=test_config.saved_path + "/logs",
        warmup_ratio=0.2,
        per_device_train_batch_size=test_config.train_batch_size,
        per_device_eval_batch_size=test_config.dev_batch_size,
        num_train_epochs=test_config.num_train_epochs,

        save_strategy="epoch",

        report_to="tensorboard",
        generation_max_length=test_config.generation_max_length,
        weight_decay=0.01,
        save_total_limit=test_config.max_keep_ckpt,
        gradient_accumulation_steps=1,
        predict_with_generate=True,
        include_inputs_for_metrics=True,
        metric_for_best_model="eval_rouge2",
    )
    training_args.__setattr__("min_length", test_config.generation_min_length)
    # training_args.__setattr__("debug","underflow_overflow")
    # training_args.__setattr__("use_legacy_prediction_loop",True)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqNERTrainer(
        model,
        training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    test_result = trainer.predict(test_dataset)
    with open(test_config.saved_path + "/test_result.txt", "w") as f:
        f.write(
            json.dumps(test_result.metrics, indent=2)
        )
    pickle.dump(
        test_result, open(
            test_config.saved_path + "/test_result.pkl", "wb"
        )
    )


def test_all_checkpoints(model_dir):
    for checkpoint in os.listdir(model_dir):
        if checkpoint.startswith("checkpoint"):
            checkpoint_dir = model_dir + "/" + checkpoint

            test_config = copy.deepcopy(cfg)
            test_config.saved_path = checkpoint_dir
            test_config.pretrain_model_path = checkpoint_dir
            test_model(test_config)


if __name__ == "__main__":
    # 测试
    # saved_dir = "./saved_dir/t5_ner"
    # for model_dir in glob(saved_dir+"/*/mechan*/model"):
    #     test_all_checkpoints(model_dir)

    test_all_checkpoints("./saved_dir/bart_ner/mechan_2022-08-03/model")
