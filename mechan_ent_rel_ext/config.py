class S2SNERConfig:
    pretrain_model_path = "/home/myq/bert/t5/t5-base"
    data_dir = "/home/myq/code/mechanism_ext/data/标注结果/机制实体关系抽取/train_data"
    saved_path = "/home/myq/code/mechanism_ext/saved_dir/t5_ner/from_scierc"
    max_keep_ckpt = 3
    max_length = 400

    num_train_epochs = 40
    train_batch_size = 8
    dev_batch_size = 8
    generation_max_length = 20
    generation_min_length = 5
    generation_num_beams = 5
    num_workers = 1
    num_hidden_layers = 12
    word_rand_mask = 0.05
    model_type = "bart"

cfg = S2SNERConfig()
