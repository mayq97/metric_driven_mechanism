import json
import os
os.environ["TOKENIZERS_PARALLELISM"]  = "true"
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
import nlpaug.augmenter.word as naw
from nlpaug.util import Action
import json
from tqdm import tqdm
import random



bert_model_path = "/home/myq/bert/sci_bert_allenai"


class DataAugmenter:

    def __init__(self,
        augment_type_config = {}
    ) -> None:
        self.augmenter = {}
        self.augment_type_config = augment_type_config
        if "aug_by_insert" in augment_type_config.keys():
            self.augmenter["aug_by_insert"] = naw.ContextualWordEmbsAug(
            model_path=bert_model_path, action="insert",device="cuda")
        
        if "aug_by_substitute" in augment_type_config.keys():
            self.augmenter["aug_by_substitute"] = naw.ContextualWordEmbsAug(
            model_path=bert_model_path, action="substitute",device="cuda")
        
        if "aug_by_swap" in augment_type_config.keys():
            self.augmenter["aug_by_swap"] = naw.RandomWordAug(action="swap")
        
        if "back_translation_aug" in augment_type_config.keys():
            self.augmenter["back_translation_aug"] = naw.BackTranslationAug(
                from_model_name='../opus-mt-en-zh/', 
                to_model_name='../opus-mt-zh-en/',
                device="cuda"
            )
        
    def augment(self,text,paper_id,label):
        all_texts = []

        for k,n in self.augment_type_config.items():
            aug = self.augmenter[k]
            texts = aug.augment(text,n = n)
            for new_text in texts:
                all_texts.append(
                    {
                        "text":new_text,
                        "label":label,
                        # "sent_idx":sent_idx,
                        "id":paper_id,
                        "aug":k
                    }
                )
        return all_texts


if __name__ == "__main__":
    data_dir = "./mechan_detection/data/"
    data = json.load(open(data_dir + "train.json","r",encoding="utf-8"))
    new_data = []
    aug = DataAugmenter(
        {
            "aug_by_substitute":10,
            "aug_by_swap":4,
        }
    )
    for paper in tqdm(data):
        if paper["label"] == 1 or (random.random() < 0.4 and paper["label"] == 0):
            new_data.extend(
                aug.augment(paper["text"],paper_id=paper["id"],
                label = paper["label"]
                )
            )
    data_aug = data + new_data
    random.shuffle(data_aug)
    json.dump(
        data_aug,
        open(
            data_dir+"train.aug.json","w",encoding="utf-8"
        ),
        indent=2,
        ensure_ascii=False
    )

