import json
import torch
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset
import torch
from typing import Dict, List
import os
from tqdm import tqdm
import random

random.seed(42)


def mechanism_data_format_convert(data, with_empty=False, sampleing_prob=0.1):
    """
    数据格式转换，
    context 为摘要，
    query为 prompt（不同的实体类型的query 也不一样）
    ent_text 为 与query 对应的实体列表
    """

    rel_map = {
        "Affect": "Which entities have Affect on {} but direction unknown ?",
        "Pos_Affect": "Which entities have Positive Affect on {} ?",
        "Neg_Affect": "Which entities have Negative Affect on {} ?"
    }

    text_sum = []
    for sample_idx, sample in tqdm(enumerate(data), desc="dataformat_convert"):
        context = " ".join(sample["tokens"])
        tmp = {
            "paper_id": sample["id"],
            "context": context,
            "words": sample["tokens"],
            "qas_id": "{}.{}".format(sample_idx, 0),
            "entity_label": "Effect",
            "rel_type": None,
            "query": "What is the measurable and comparable metric entity ?",
            "ent_text": set([ent["text"] for ent in sample["entities"] if ent["type"] == "Effect"]),
        }
        text_sum.append(tmp)

        eff_to_op = {}
        for rel in sample["relations"]:
            eff = sample["entities"][rel["tail"]]["text"]
            direction = rel["type"]
            temp_dict_for_eff = eff_to_op.get(eff, {
                "Affect": set(),
                "Pos_Affect": set(),
                "Neg_Affect": set()
            })
            temp_dict_for_eff[direction].add(
                sample["entities"][rel["head"]]["text"]
            )
            eff_to_op[eff] = temp_dict_for_eff

        for eff, op_info in eff_to_op.items():
            for direction, op_ent_list in op_info.items():
                if len(op_ent_list) > 0:
                    text_sum.append(
                        {
                            "paper_id": sample["id"],
                            "context": context,
                            "words": sample["tokens"],
                            "qas_id": "{}.{}".format(sample_idx, 1),
                            "entity_label": "Operation",
                            "query": rel_map[direction].format(eff),
                            "rel_type": direction,
                            "ent_text": op_ent_list

                        }
                    )
                elif len(op_ent_list) == 0 and with_empty and random.random() < sampleing_prob:
                    text_sum.append(
                        {
                            "paper_id": sample["id"],
                            "context": context,
                            "words": sample["tokens"],
                            "qas_id": "{}.{}".format(sample_idx, 1),
                            "entity_label": "Operation",
                            "query": rel_map[direction].format(eff),
                            "rel_type": direction,
                            "ent_text": set(["non entity"])

                        }
                    )
    return text_sum


class S2SNERDataset(Dataset):

    def __init__(self,
                 json_path,
                 tokenizer: BertWordPieceTokenizer,
                 max_length: int = 512,
                 train_sampleing_prob=0.1,
                 desc="train",
                 model_type="t5"
                 ):
        raw_data = [json.loads(line)
                    for line in open(json_path, encoding="utf-8")]
        self.tokenizer = tokenizer
        self.max_length = max_length
        if desc == "train":
            self.all_data = mechanism_data_format_convert(
                raw_data, with_empty=True, sampleing_prob=train_sampleing_prob)
            random.shuffle(self.all_data)

        elif desc == "dev" or desc == "test":
            self.all_data = mechanism_data_format_convert(
                raw_data, with_empty=True, sampleing_prob=1.1)
            random.shuffle(self.all_data)

        self.model_type = model_type

    def __len__(self):
        return len(self.all_data)

    def preprocess_function(self, prefix, context, target_ent_list):
        if self.model_type == "bart":

            inputs = prefix + "[SEP]" + context

            model_inputs = self.tokenizer(
                inputs, max_length=512, truncation=True)

            with self.tokenizer.as_target_tokenizer():
                target = " <> ".join(target_ent_list)
                labels = self.tokenizer(target, max_length=64, truncation=True)

            model_inputs["labels"] = labels["input_ids"]

            return model_inputs

        elif self.model_type == "t5":
            inputs = prefix + "[SEP]" + context

            model_inputs = self.tokenizer(
                inputs, max_length=512, truncation=True)

            with self.tokenizer.as_target_tokenizer():
                target = " >> ".join(target_ent_list)
                labels = self.tokenizer(target, max_length=64, truncation=True)

            model_inputs["labels"] = labels["input_ids"]

            return model_inputs

    def __getitem__(self, item):

        data = self.all_data[item]
        qas_id = data.get("qas_id", "0.0")
        sample_idx, label_idx = qas_id.split(".")
        sample_idx = torch.LongTensor([int(sample_idx)])
        label_idx = torch.LongTensor([int(label_idx)])

        result = self.preprocess_function(
            prefix=data["query"], context=data["context"],
            target_ent_list=data["ent_text"])

        return {
            "input_ids": torch.LongTensor(result["input_ids"]),
            "attention_mask": torch.LongTensor(result["attention_mask"]),
            "labels": torch.LongTensor(result["labels"]),
            "sample_idx": sample_idx,
            "query": data["query"],
            "context": data["context"],
            "labels_text": data["ent_text"],
            "rel_type": data["rel_type"],
            "ent_idx": label_idx
        }

    def pad(self, lst, value=0, max_length=None):
        max_length = max_length or self.max_length
        while len(lst) < max_length:
            lst.append(value)
        return lst


scierc_mapping = {
    "Generic": "Which are the general terms or pronouns?",
    "Task": "Which are the applications, problems to be solved, systems to construct?",
    "Method": "Which are the methods , models, systems to be used, or tools, components of a system, frameworks?",
    "Metric": "Which are the metrics, measures, or entities that can express quality of a system/method?",
    "Material": "Which are the data, datasets, resources, corpus, knowledge base",
    "OtherScientificTerm": "which are the phrases that are a scientific terms but not above classes?",
}


def scierc_data_format_convert(data, with_empty=True, sampleing_prob=1.1):
    text_sum = []
    for sample_idx, sample in tqdm(enumerate(data), desc="dataformat_convert"):
        tokens = []
        for sent in sample["sentences"]:
            tokens.extend(sent)

        context = " ".join(tokens)
        ents = {
            "Generic": set(),
            "Task": set(),
            "Method": set(),
            "Metric": set(),
            "Material": set(),
            "OtherScientificTerm": set()
        }
        for sent in sample["ner"]:
            for start, end, ent_type in sent:
                if ent_type not in ents.keys():
                    continue
                ents[ent_type].add(" ".join(tokens[start:end+1]))
        i = 0
        for k, ent_words_list in ents.items():
            if len(ent_words_list) > 0:
                tmp = {
                    "paper_id": sample["doc_key"],
                    "context": context,
                    "words": tokens,
                    "qas_id": "{}.{}".format(sample_idx, i),
                    "entity_label": "",
                    "query": scierc_mapping[k],
                    "ent_text": ent_words_list,
                }
                text_sum.append(tmp)
            elif with_empty and random.random() <= sampleing_prob:
                tmp = {
                    "paper_id": sample["doc_key"],
                    "context": context,
                    "words": tokens,
                    "qas_id": "{}.{}".format(sample_idx, i),
                    "entity_label": "",
                    "query": scierc_mapping[k],
                    "ent_text": set(["non entity"])
                }
                text_sum.append(tmp)
            i += 1

    return text_sum


class SCIERCRDataset(Dataset):

    def __init__(self,
                 json_path,
                 tokenizer: BertWordPieceTokenizer,
                 max_length: int = 512,
                 desc="train",
                 model_type="t5"
                 ):
        raw_data = [json.loads(line)
                    for line in open(json_path, encoding="utf-8")]
        self.tokenizer = tokenizer
        self.max_length = max_length
        if desc == "train":
            self.all_data = scierc_data_format_convert(
                raw_data, with_empty=True, sampleing_prob=1.1)
            random.shuffle(self.all_data)

        elif desc == "dev" or desc == "test":
            self.all_data = scierc_data_format_convert(
                raw_data, with_empty=True, sampleing_prob=1.1)

        self.model_type = model_type

    def __len__(self):
        return len(self.all_data)

    def preprocess_function(self, prefix, context, target_ent_list):
        if self.model_type == "bart":
            inputs = prefix + "[SEP]" + context

            model_inputs = self.tokenizer(
                inputs, max_length=512, truncation=True)

            with self.tokenizer.as_target_tokenizer():
                target = " <> ".join(target_ent_list)
                labels = self.tokenizer(target, max_length=64, truncation=True)

            model_inputs["labels"] = labels["input_ids"]

            return model_inputs
        elif self.model_type == "t5":
            inputs = prefix + "[SEP]" + context

            model_inputs = self.tokenizer(
                inputs, max_length=512, truncation=True)

            with self.tokenizer.as_target_tokenizer():
                target = " >> ".join(target_ent_list)
                labels = self.tokenizer(target, max_length=64, truncation=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

    def __getitem__(self, item):

        data = self.all_data[item]
        qas_id = data.get("qas_id", "0.0")
        sample_idx, label_idx = qas_id.split(".")
        sample_idx = torch.LongTensor([int(sample_idx)])
        label_idx = torch.LongTensor([int(label_idx)])

        result = self.preprocess_function(
            prefix=data["query"], context=data["context"],
            target_ent_list=data["ent_text"])

        return {
            "input_ids": torch.LongTensor(result["input_ids"]),
            "attention_mask": torch.LongTensor(result["attention_mask"]),
            "labels": torch.LongTensor(result["labels"]),
            "sample_idx": sample_idx,
            "ent_idx": label_idx
        }

    def pad(self, lst, value=0, max_length=None):
        max_length = max_length or self.max_length
        while len(lst) < max_length:
            lst.append(value)
        return lst
