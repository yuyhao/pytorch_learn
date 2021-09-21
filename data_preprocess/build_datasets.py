# -*- coding: utf-8 -*-
# @Time: 2021/9/12 16:48
# @Author: yuyinghao
# @FileName: build_datasets.py
# @Software: PyCharm

from utils.file_io import FileIo
from utils.tokenizer import Tokenizer
from tqdm import tqdm
from data_preprocess import Config


class BuildDatasets:
    def __init__(self, corpus_path, vocab_path, pad_size, special_word):
        self.corpus_path = corpus_path
        self.vocab_path = vocab_path
        self.pad_size = pad_size
        self.special_word = special_word

    def load_dataset(self, corpus_path, vocab, pad_size):
        contents = []
        with open(corpus_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                try:
                    lin = line.strip()
                    if not lin:
                        continue
                    content = ''.join(lin.split(',')[:-1])  # 内容
                    label = lin.split(',')[-1]  # 类别

                    token = Tokenizer.tokenizer(content)  # 将内容tokenize
                    seq_len = len(token)  # 该句内容的长度

                    if pad_size:  # 限制句子的最大长度
                        if len(token) < pad_size:
                            token.extend([self.special_word.get(
                                'pad')] * (pad_size - len(token)))
                        else:
                            token = token[:pad_size]
                            seq_len = pad_size

                    # 将字映射为id
                    words_line = []
                    for word in token:
                        words_line.append(
                            vocab.get(
                                word, vocab.get(
                                    self.special_word['unkown'])))
                    contents.append((words_line, int(label), seq_len))
                except BaseException:
                    pass
        return contents

    def build_datasets(self):
        # 加载字典
        vocab = FileIo.load_pickle_obj(self.vocab_path)
        print("Vocab size: {len(vocab)}")

        # 处理语料
        contens = self.load_dataset(self.corpus_path, vocab, self.pad_size)

        # 划分训练集、测试集
        train = contens[: int(len(contens) * 0.8)]
        test = contens[int(len(contens) * 0.8):]
        return vocab, train, test


if __name__ == '__main__':
    build_datasets = BuildDatasets(
        Config.preprocess_vocab_corpus_path,
        Config.vocab_path,
        Config.pad_size,
        Config.special_word)
    vocab_info, train_datasets, test_datasets = build_datasets.build_datasets()
