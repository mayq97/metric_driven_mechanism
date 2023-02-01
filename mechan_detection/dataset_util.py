import json
from torch.utils.data import Dataset
from tokenizers import BertWordPieceTokenizer
import torch
import random


class MechanismDetectionDataset(Dataset):
    """
    MRC NER Dataset
    Args:
        json_path: path to mrc-ner style json
        tokenizer: BertTokenizer
        max_length: int, max length of query+context
        possible_only: if True, only use possible samples that contain answer for the query/context
        is_chinese: is chinese dataset
    """

    def __init__(self, json_path, desc="train", tokenizer: BertWordPieceTokenizer = None, max_length: int = 512):
        raw_data = json.load(open(json_path, encoding="utf-8"))
        if desc == "train":
            random.shuffle(raw_data)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.all_data = raw_data

    def __len__(self):
        return len(self.all_data)

    def preprocess_function(self, text):
        result = self.tokenizer(text,)
        position_ids = list(range(len(result["input_ids"])))
        return {
            "input_ids": torch.LongTensor(result["input_ids"]),
            "token_type_ids": torch.LongTensor(result["token_type_ids"]),
            "attention_mask": torch.LongTensor(result["attention_mask"]),
            "position_ids": torch.LongTensor(position_ids)
        }

    def __getitem__(self, item):
        data = self.all_data[item]

        result = self.preprocess_function(
            text=data["text"])
        result["labels"] = data["label"]
        return result

    def pad(self, lst, value=0, max_length=None):
        max_length = max_length or self.max_length
        while len(lst) < max_length:
            lst.append(value)
        return lst
