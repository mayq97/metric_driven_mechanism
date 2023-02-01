from transformers import BertTokenizer, DataCollatorWithPadding
import os
import json
from utils.text_clf_trainer import TextClfTrainer
import numpy as np
from mechan_detection.dataset_util import MechanismDetectionDataset

from transformers.training_args import TrainingArguments
from datetime import datetime
from utils.util import convert_config_to_dict
from mechan_detection.config import MechanismDetectionConfig
from tensorboardX import SummaryWriter
from utils.base import BertForSequenceClassification, BertConfig
import torch
import random

os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


cfg = MechanismDetectionConfig()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# 设置随机数种子
setup_seed(42)


def load_dataset(train_data_type, data_dir, tokenizer, max_length=512):
    train_dataset = MechanismDetectionDataset(
        json_path=data_dir + "/{}".format(data_type2data_file[train_data_type]), desc="train", tokenizer=tokenizer, max_length=max_length,
    )

    dev_dataset = MechanismDetectionDataset(
        json_path=data_dir + "/dev.json", desc="dev", tokenizer=tokenizer, max_length=max_length,
    )

    test_dataset = MechanismDetectionDataset(
        json_path=data_dir + "/test.json", desc="test", tokenizer=tokenizer, max_length=max_length,
    )
    return train_dataset, dev_dataset, test_dataset


def train_model(train_data_type="raw", train_config: MechanismDetectionConfig = None):

    train_config.__setattr__(
        "train_data_file", data_type2data_file[train_data_type])

    # config缓存
    TIMESTAMP = "{0:%Y-%m-%d %H_%M_%S}".format(datetime.now())
    train_config.saved_path = "{}/{}_{}".format(
        train_config.saved_path, TIMESTAMP, train_data_type)

    config_data = convert_config_to_dict(train_config)
    os.mkdir(train_config.saved_path)
    json.dump(config_data, open("{}/config.json".format(train_config.saved_path),
              "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # 数据集加载
    tokenizer = BertTokenizer.from_pretrained(train_config.pretrain_model_path)
    train_dataset, dev_dataset, test_dataset = load_dataset(
        train_data_type, train_config.data_dir, tokenizer, train_config.max_length)

    # 模型加载
    model_config = BertConfig.from_pretrained(train_config.pretrain_model_path)
    model_config.__setattr__("num_labels", 2)
    model_config.__setattr__("problem_type", "single_label_classification")
    model_config.__setattr__("pos_weight", 1)
    model_config.__setattr__(
        "num_hidden_layers", train_config.num_hidden_layers)
    model = BertForSequenceClassification(model_config)

    # 训练参数
    train_args = TrainingArguments(
        output_dir=train_config.saved_path + "/model",
        logging_dir=train_config.saved_path + "/logs",

        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        report_to="tensorboard",

        logging_steps=100,
        metric_for_best_model="eval_f1",
        per_device_train_batch_size=train_config.train_batch_size,
        per_device_eval_batch_size=train_config.dev_batch_size,

        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        save_total_limit=train_config.max_keep_ckpt,

        num_train_epochs=train_config.num_train_epochs,
        use_legacy_prediction_loop=True,
        load_best_model_at_end=True,
        resume_from_checkpoint=True
    )
    data_collator = DataCollatorWithPadding(tokenizer)
    trainer = TextClfTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,

    )

    trainer.train()
    trainer.evaluate()
    trainer.save_model()

    metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")

    with open(train_config.saved_path + "/model/test_result.txt", "w") as f:
        f.write(
            json.dumps(metrics, indent=2)
        )


if __name__ == "__main__":
    import copy
    data_type2data_file = {
        "raw": "train.json",
        "with_pos_neg_sents": "train_with_sent_level_data.json",
        "aug_train_v1": "aug_train.json",
    }
    for data_type in [
        "raw",
        "with_pos_neg_sents",
        "aug_train_v1"
    ]:
        train_config = copy.deepcopy(cfg)
        train_model(train_data_type=data_type, train_config=train_config)
