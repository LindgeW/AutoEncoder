from config import config
from datautil.dataloader import load_dataset
from vocab.Vocab import create_vocab
import torch
import numpy as np
from modules.auto_encoder import AutoEncoder
from classifier import Classifier


if __name__ == '__main__':
    # 设置随机种子(固定随机值)
    np.random.seed(666)
    torch.manual_seed(6666)
    torch.cuda.manual_seed(1234)  # 为当前GPU设置种子
    # torch.cuda.manual_seed_all(4321)  # 为所有GPU设置种子(如果有多个GPU)

    print('GPU available: ', torch.cuda.is_available())
    print('CuDNN available: ', torch.backends.cudnn.enabled)
    print('GPU number: ', torch.cuda.device_count())

    # 加载数据(训练集-学习、开发集-调参、测试集-评估)
    data_opts = config.data_path_parse('./config/data_path.json')
    train_data = load_dataset(data_opts['data']['train_data'])
    dev_data = load_dataset(data_opts['data']['dev_data'])
    test_data = load_dataset(data_opts['data']['test_data'])
    print('train_size=%d  dev_size=%d  test_size=%d' % (len(train_data), len(dev_data), len(test_data)))

    # 设置参数(数据参数+模型参数)
    args = config.arg_parse()
    if args.enable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda', args.cuda)
    else:
        args.device = torch.device('cpu')
    print(args.device)

    # 创建词表
    vocab = create_vocab(data_opts['data']['train_data'])
    embedding_weights = vocab.get_embedding_weights(data_opts['data']['embedding_path'])
    # vocab.save_vocab(data_opts['model']['save_vocab_path'])

    # 构建分类模型
    args.pad = vocab.PAD
    # args.label_size = vocab.label_size
    # args.vocab_size = vocab.vocab_size
    auto_encoder = AutoEncoder(args, embedding_weights).to(args.device)

    # if torch.cuda.device_count() > 1:
    #     auto_encoder = torch.nn.DataParallel(auto_encoder, device_ids=[0, 1])

    classifier = Classifier(auto_encoder, args, vocab)
    classifier.summary()

    # 训练
    classifier.train(train_data, dev_data)

    # 评估
    classifier.evaluate(test_data)
