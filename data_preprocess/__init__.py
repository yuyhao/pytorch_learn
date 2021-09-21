# -*- coding: utf-8 -*-
# @Time: 2021/9/11 21:42
# @Author: yuyinghao
# @FileName: __init__.py.py
# @Software: PyCharm


from utils.config import get_abs_path
import os


class Config:
    # 字典建立
    special_word = {
        'unkown': '<UNK>',
        'pad': '<PAD>'
    }  # 未知字符、pad字符

    raw_vocab_corpus_path = get_abs_path('data/raw_corpus/product_info.csv')
    preprocess_vocab_corpus_path = get_abs_path('data/preprocess_corpus/vocab_corpus_info.csv')
    vocab_path = get_abs_path('data/dict/vocab.pkl')
    mode = 'char'
    max_size = 10000
    min_freq = 0

    # 数据集构造
    pad_size = 40
