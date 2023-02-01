import spacy
import json
from myLabTools.nlp.data_process import dict_list2jsonline_file
from spacy.training import offsets_to_biluo_tags, biluo_to_iob
import os
from tqdm import tqdm
nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe('sentencizer')


class NERFormatConv:
    """
    将以word offset 形式存储的书句转换为 biluo tags
    """

    def __init__(self, conv_tag2id=True):
        if conv_tag2id:
            self.biluo_tag2id = json.load(
                open("bert_ner/BILUO_tag2id.json", "r"))
            self.bio_tag2id = json.load(open("bert_ner/BIO_tag2id.json", "r"))

        self.conv_tag2id = conv_tag2id

    def process_dataset_from_jsonline(self, data_path, saved_dir="", saved_name="", type="biluo"):
        """命名实体识别书句处理主函数

        原始数据格式，有text，tokens，entities，relations，id等字段的dict

        {
        "text": "We improve the basic framework by Skip-chain CRFs and 2D CRFs to better accommodate the features of forums for better performance .", 
        "tokens": ["We", "improve", "the", "basic", "framework", "by", "Skip", "-", "chain", "CRFs", "and", "2D", "CRFs", "to", "better", "accommodate", "the", "features", "of", "forums", "for", "better", "performance", "."], 
        "entities": [{"type": "Operation", "start": 6, "end": 10, "text": "Skip-chain CRFs"}, {"type": "Effect", "start": 22, "end": 23, "text": "performance"}, {"type": "Operation", "start": 11, "end": 13, "text": "2D CRFs"}], 
        "relations": [{"type": "Pos_Affect", "head": 0, "tail": 1}, {"type": "Pos_Affect", "head": 2, "tail": 1}] }


        Args:
            data_path (_type_): 输入数据的路径
            saved_dir (_type_): 输入文件夹
            saved_name (_type_): 保存名字
            type (str, optional): 转换类型 dygie / biluo. Defaults to "biluo".
        """

        # "id": "4c4a95645bb719cb53d668cca3b104e529746377_3"}
        data = [json.loads(line) for line in open(data_path, "r")]
        new_data = []
        for item in data:
            if type == "biluo":
                new_data.append(self.conv2biluo(
                    item["tokens"], item["entities"], item["id"]))

        if not len(saved_dir) == 0:
            dict_list2jsonline_file(new_data, saved_dir, saved_name)
        return new_data

    def process_data(self, data, saved_dir="", saved_name="", type="dygie"):
        """命名实体识别书句处理主函数

        原始数据格式，有text，tokens，entities，relations，id等字段的dict

        {
        "text": "We improve the basic framework by Skip-chain CRFs and 2D CRFs to better accommodate the features of forums for better performance .", 
        "tokens": ["We", "improve", "the", "basic", "framework", "by", "Skip", "-", "chain", "CRFs", "and", "2D", "CRFs", "to", "better", "accommodate", "the", "features", "of", "forums", "for", "better", "performance", "."], 
        "entities": [{"type": "Operation", "start": 6, "end": 10, "text": "Skip-chain CRFs"}, {"type": "Effect", "start": 22, "end": 23, "text": "performance"}, {"type": "Operation", "start": 11, "end": 13, "text": "2D CRFs"}], 
        "relations": [{"type": "Pos_Affect", "head": 0, "tail": 1}, {"type": "Pos_Affect", "head": 2, "tail": 1}] ,
        "id": "4c4a95645bb719cb53d668cca3b104e529746377_3"}


        Args:
            data_path (_type_): 输入数据的路径
            saved_dir (_type_): 输入文件夹
            saved_name (_type_): 保存名字
            type (str, optional): 转换类型 dygie / biluo. Defaults to "dygie".
        """

        new_data = []
        for i, item in enumerate(data):
            if "id" in item.keys():
                item_id = item["id"]
            else:
                item_id = i

            if type == "biluo":
                new_data.append(self.conv2biluo(
                    item["tokens"], item["entities"], item_id))
            elif type == "dygie":
                new_data.append(self.conv2dygiepp(
                    item["tokens"], item["entities"], item["relations"], item_id))
            else:
                exit
        if not len(saved_dir) == 0:
            dict_list2jsonline_file(new_data, saved_dir, saved_name)
        return new_data

    def conv2biluo(self, tokens, entities, id=""):
        """Conoll 格式

        ['O', 'O', 'O', 'O', 'O', 'O', 'B-Operation', 'I-Operation', 'I-Operation', 'L-Operation', 'O', 'B-Operation', 'L-Operation', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'U-Effect', 'O']

        Args:
            tokens (_type_): _description_
            entities (_type_): _description_
            id (_type_): _description_

        Returns:
            _type_: _description_
        """
        text = " ".join(tokens)
        pos_map = {
            -1: {"start": -1, "end": -1}
        }
        for i, token in enumerate(tokens):
            pos_map[i] = {
                "start": pos_map[i-1]["end"]+1,
                "end": pos_map[i-1]["end"]+1+len(token)
            }
        ents = []
        new_entities = []
        for e in entities:
            ents.append({
                "start": pos_map[e["start"]]["start"],
                "end": pos_map[e["end"]-1]["end"],
                "label": e["type"]}
            )
            new_entities.append(
                (pos_map[e["start"]]["start"], pos_map[e["end"]-1]["end"], e["type"]))

        doc = nlp(text)
        ner_tags = offsets_to_biluo_tags(doc, new_entities)
        assert len(ner_tags) == len(tokens)
        # print("BILUO before adding new entity:", ner_tags)
        #  ['O', 'O', 'O', 'O', 'O', 'O', 'B-Operation', 'I-Operation', 'I-Operation', 'L-Operation', 'O', 'B-Operation', 'L-Operation', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'U-Effect', 'O']
        new_tokens = []
        for t in tokens:
            t = t.strip()
            if len(t) == 0:
                new_tokens.append("[UNK]")
            else:
                new_tokens.append(t)
        result = {
            "id": id,

            "tokens": new_tokens
        }
        if self.conv_tag2id:
            result["biluo_tags"] = [self.biluo_tag2id[t] for t in ner_tags]
            result["bio_tags"] = [self.bio_tag2id[t]
                                  for t in biluo_to_iob(ner_tags)]
        else:
            result["biluo_tags"] = ner_tags
            result["bio_tags"] = biluo_to_iob(ner_tags)
        return result


if __name__ == "__main__":
    demo = NERFormatConv()
    input_dir = "mechan_ent_rel_ext/data"

    output_dir = "bert_ner/data"
    for file in tqdm(["train.json", "dev.json", "test.json"]):
        demo.process_dataset_from_jsonline(
            os.path.join(input_dir, file), output_dir, file, type="biluo"
        )
