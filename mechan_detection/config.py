class MechanismDetectionConfig:
    # pretrain_model_path = "/media/mayq/文档/pretrained_model/bert-scibert"
    pretrain_model_path = "/home/myq/bert/sci_bert_allenai"
    data_dir = "./data/train_data/6_2_2_data_split_for_mechanism_find"
    saved_path = "./saved_dir/mechanism_detect"
    max_keep_ckpt = 3
    max_length = 400
    num_train_epochs = 40
    train_batch_size = 16
    dev_batch_size = 16
    num_workers = 4
    num_hidden_layers = 12
    # 模型参数