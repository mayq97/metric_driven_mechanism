摘要级别的数据

aug_train.json 对 train.json 中的正样本数据进行增强，增强的参数        {
            "aug_by_substitute":10,
            "aug_by_swap":4,
        }

train_with_sent_level_data.json 段落级的数据加句子级（pos-neg sents）的数据 text_clf_sent_level.ipynb 获取方法见 text_clf_sent_level.ipynb


aug_train_with_sent_level_data.json  对train_with_sent_level_data.json 的数据（paper["label"] == 1 ）进行增强，增强的参数        {
            "aug_by_substitute":10,
            "aug_by_swap":4,
        }

aug_train_with_sent_level_data_v2.json ： 相比于 aug_train_with_sent_level_data 添加了更多的 负样例 paper["label"] == 1 or (random.random() < 0.2 and paper["label"] == 0)，然后进行数据增强


aug_train_with_sent_level_data_v2.4.json 相比于 aug_train_with_sent_level_data 添加了更多的 负样例 paper["label"] == 1 or (random.random() < 0.4 and paper["label"] == 0)，然后进行数据增强