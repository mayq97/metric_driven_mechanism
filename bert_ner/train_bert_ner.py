from bert_ner.bert_ner_dataset import load_encode_dataset
from bert_ner.ner_metrics import compute_metrics
from datetime import datetime
import pickle
import json
from transformers import DataCollatorForTokenClassification, AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import sys
import os

sys.path.append(".")
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def train_model(saved_dir, label_type="BIO", pretrain_model_path="/home/myq/bert/sci_bert_allenai"):
    TIMESTAMP = "{0:%Y-%m-%d %H_%M_%S}".format(datetime.now())
    saved_dir = saved_dir + "/" + TIMESTAMP
    if label_type == "BIO":
        tag2id_path = "bert_ner/BIO_tag2id.json"
    tag2id = json.load(open(tag2id_path, "r"))

    tokenizer = AutoTokenizer.from_pretrained(pretrain_model_path)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    file_paths = {
        "train": "bert_ner/data/train.json",
        "dev": "bert_ner/data/dev.json",
        "test": "bert_ner/data/test.json"
    }

    dataset = load_encode_dataset(file_paths, label_type=label_type)

    if label_type == "BIO":
        model = AutoModelForTokenClassification.from_pretrained(
            pretrain_model_path, num_labels=5)
    elif label_type == "BILUO":
        model = AutoModelForTokenClassification.from_pretrained(
            pretrain_model_path, num_labels=8)

    training_args = TrainingArguments(
        output_dir=saved_dir + "/model",
        logging_dir=saved_dir + "/logs",

        run_name="bert_ner",
        report_to="tensorboard",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        warmup_ratio=0.05,
        per_device_train_batch_size=16,
        eval_steps=100,
        logging_steps=100,
        per_device_eval_batch_size=16,
        num_train_epochs=40,
        weight_decay=0.01,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro-avg_f1-score"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()
    test_outputs = trainer.predict(dataset["test"])
    print(test_outputs.metrics)
    with open(saved_dir + "/test_result.txt", "w") as f:
        f.write(
            json.dumps(test_outputs.metrics, indent=2)
        )
    pickle.dump(
        test_outputs,
        open(saved_dir + "/test_result.pkl", "wb")
    )


def test_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    file_paths = {"test": "bert_ner/data/test.json"}
    dataset = load_encode_dataset(file_paths)

    model = AutoModelForTokenClassification.from_pretrained(model_path)

    training_args = TrainingArguments(
        logging_dir=model_path + "/logs",

        output_dir=model_path + "/test_result",
        run_name="bert_ner",
        report_to="tensorboard",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        warmup_ratio=0.05,
        per_device_train_batch_size=16,
        eval_steps=100,
        logging_steps=100,
        per_device_eval_batch_size=16,
        num_train_epochs=40,
        weight_decay=0.01,
        save_total_limit=3,
        load_best_model_at_end=True,
        resume_from_checkpoint=True,

        metric_for_best_model="eval_macro-avg_f1-score",

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    test_outputs = trainer.predict(dataset["test"])
    print(test_outputs.metrics)
    with open(model_path + "/test_result/test_result.txt", "w") as f:
        f.write(
            json.dumps(test_outputs.metrics, indent=2)
        )
    pickle.dump(
        test_outputs,
        open(model_path + "/test_result/test_result.pkl", "wb")
    )


if __name__ == "__main__":
    test_model(
        "./saved_dir/bertner/2022-08-03 23_32_06/model/checkpoint-825")

    train_model("./saved_dir/bertner")
