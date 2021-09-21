# -*- coding: utf-8 -*-
# @Time: 2021/9/12 10:02
# @Author: yuyinghao
# @FileName: build_vocab.py
# @Software: PyCharm

from tqdm import tqdm
from utils.file_io import FileIo
from utils.tokenizer import Tokenizer
from data_preprocess import Config


class BuildVocab:
    """
    通过语料制作字典
    """

    def __init__(
            self,
            corpus_path,
            vocab_path,
            mode,
            max_size,
            min_freq,
            special_word):
        self.corpus_path = corpus_path
        self.vocab_path = vocab_path
        self.mode = mode
        self.max_size = max_size
        self.min_freq = min_freq
        self.special_word = special_word  # 添加的特殊词,为()

    def build_vocab(self):
        vocab_dic = dict()
        with open(self.corpus_path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue

                content = ''.join(lin.split(',')[:-1])  # 按制表符切分，取第一部分

                for word in Tokenizer.tokenizer(content):  # 统计每个字的频数
                    vocab_dic[word] = vocab_dic.get(word, 0) + 1

            vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= self.min_freq], key=lambda x: x[1],
                                reverse=True)[
                         :self.max_size]  # 按频数排序，并截取最大字典长度

            vocab_dic = {
                word_count[0]: idx for idx, word_count in enumerate(vocab_list)}  # 将字映射到数字

            for i in self.special_word.values():
                vocab_dic.update({i: len(vocab_dic)})  # 将特殊词加在最后

        FileIo.save_2_pickle(vocab_dic, self.vocab_path)  # 字典存储


if __name__ == '__main__':
    # 将原始数据加工成构造字典需要的格式
    data_info = FileIo.get_data_from_csv(Config.raw_vocab_corpus_path)
    data_need = data_info[['productname', 'secondcategory_id']]
    data_need.to_csv(Config.preprocess_vocab_corpus_path, index=False, header=False)

    # 构造字典
    build_vocab = BuildVocab(Config.preprocess_vocab_corpus_path, Config.vocab_path, Config.mode, Config.max_size,
                             Config.min_freq, Config.special_word)
    build_vocab.build_vocab()
