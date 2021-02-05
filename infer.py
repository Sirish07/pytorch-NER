import random
import torch
import numpy as np
import argparse
import os
from utils import WordVocabulary, LabelVocabulary, Alphabet, build_pretrain_embedding, my_collate_fn, lr_decay
import time
from dataset import MyDataset
from torch.utils.data import DataLoader
from model import NamedEntityRecog
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from train import train_model, evaluate, test_model
from helper import process, prediction

seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Named Entity Recognition Model')
    parser.add_argument('--word_embed_dim', type=int, default=100)
    parser.add_argument('--word_hidden_dim', type=int, default=100)
    parser.add_argument('--char_embedding_dim', type=int, default=30)
    parser.add_argument('--char_hidden_dim', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--pretrain_embed_path', default='/content/pytorch-NER/data/glove.6B/glove.6B.100d.txt')
    parser.add_argument('--savedir', default='data/model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--lr', type=float, default=0.015)
    parser.add_argument('--feature_extractor', choices=['lstm', 'cnn'], default='cnn')
    parser.add_argument('--use_char', type=bool, default=True)
    parser.add_argument('--train_path', default='/content/train.iob')
    parser.add_argument('--dev_path', default='/content/eval.iob')
    parser.add_argument('--test_path', default='/content/eval.iob')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--number_normalized', type=bool, default=True)
    parser.add_argument('--use_crf', type=bool, default=True)

    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()
    print('use_crf:', args.use_crf)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    test_path = '/content/test.iob'
    text = input ("Enter the sentence :")
    print(text)
    process(text, test_path)
    

    eval_path = "evaluation"
    eval_temp = os.path.join(eval_path, "temp")
    if not os.path.exists(eval_temp):
        os.makedirs(eval_temp)
    test_pred_file = eval_temp + '/test_pred.txt'

    # Loading the vocabulary
    model_name = args.savedir + '/' + args.feature_extractor + str(args.use_char) + str(args.use_crf)
    word_vocab = WordVocabulary(args.train_path, args.dev_path, args.test_path, args.number_normalized)
    label_vocab = LabelVocabulary(args.train_path)
    alphabet = Alphabet(args.train_path, args.dev_path, args.test_path) 


    emb_begin = time.time()
    pretrain_word_embedding = build_pretrain_embedding(args.pretrain_embed_path, word_vocab, args.word_embed_dim)
    emb_end = time.time()
    emb_min = (emb_end - emb_begin) % 3600 // 60
    print('build pretrain embed cost {}m'.format(emb_min))

    args.test_path = test_path
    test_dataset = MyDataset(args.test_path, word_vocab, label_vocab, alphabet, args.number_normalized)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)

    model = NamedEntityRecog(word_vocab.size(), args.word_embed_dim, args.word_hidden_dim, alphabet.size(),
                             args.char_embedding_dim, args.char_hidden_dim,
                             args.feature_extractor, label_vocab.size(), args.dropout,
                             pretrain_embed=pretrain_word_embedding, use_char=args.use_char, use_crf=args.use_crf,
                             use_gpu=use_gpu)
    if use_gpu:
        model = model.cuda()
    if use_gpu:
        model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    model.load_state_dict(torch.load(model_name))
    test_model(test_dataloader, model, word_vocab, label_vocab, test_pred_file, use_gpu)

    print("Compressed sentence is \n")
    print(prediction(test_pred_file))