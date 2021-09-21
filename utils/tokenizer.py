# -*- coding: utf-8 -*-
# @Time: 2021/9/12 17:06
# @Author: yuyinghao
# @FileName: tokenizer.py
# @Software: PyCharm

class Tokenizer:
    @staticmethod
    def tokenizer(seq, mode='char'):
        """
        param:mode:
            char: 按字切分
            word: 按词切分, 要求语料是分好词并用空格隔开
        """
        if mode == 'char':
            return [i for i in seq]
        elif mode == 'word':
            return seq.split(' ')
