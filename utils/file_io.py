# -*- coding: utf-8 -*-
# @Time: 2021/9/12 11:04
# @Author: yuyinghao
# @FileName: file_io.py
# @Software: PyCharm

import os
import pickle
import pandas as pd


class FileIo:
    @staticmethod
    def file_reader(file_path):
        """
        按行读取文件，以迭代器的形式返回
        :param file_path:
        :return:
        """
        if not os.path.exists(file_path):
            raise FileExistsError('file is not exists.{}'.format(file_path))
        with open(file_path, 'r', encoding='utf-8') as f:
            while True:
                line = f.readline()
                if line:
                    yield line
                else:
                    break

    @staticmethod
    def data_save(data_list, save_path, sep=','):
        """
        将list数据落盘
        :param data_list: [[record], [record], [record]...]
        :param save_path:
        :param sep:
        :return:
        """
        with open(save_path, 'w', encoding='utf-8') as f:
            for line in data_list:
                line = map(lambda x: str(x).strip(), line)
                f.write(sep.join(line) + '\n')

    @staticmethod
    def dict_2_txt(dic, txt_path, mode='w', seg=', '):
        """将dict文件保存至txt将dict文件保存至txt"""
        with open(txt_path, mode=mode, newline='', encoding='utf-8') as file:
            for key, value in dic.items():
                line = str(key) + seg + str(value) + '\n'
                file.write(line)

    @staticmethod
    def save_2_pickle(obj, dump_to_file_path):
        """
        将对象经过pickle序列化，然后存储到指定文件。
        :param obj:
        :param dump_to_file_path:
        :return:
        """
        with open(dump_to_file_path, 'wb') as file:
            pickle.dump(obj, file)

    @staticmethod
    def load_pickle_obj(load_file_path):
        """载入pickle文件对象"""
        with open(load_file_path, 'rb') as file:
            return pickle.load(file)

    @staticmethod
    def file_sum_lines(file_path):
        """文件数据条数"""
        count = 0
        for _ in FileIo.file_reader(file_path):
            count += 1
        return count

    @staticmethod
    def delete_top_n_lines(file_path, n):
        """删除文件的前N行"""
        with open(file_path, 'r', encoding='utf-8-sig') as f_in:
            a = f_in.readlines()
        with open(file_path, 'w', encoding='utf-8-sig') as f_out:
            f_out.write(''.join(a[n:]))

    @staticmethod
    def delete_and_return_top_n_lines(file_path, n):
        """
        删除文件的前N行,并返回被删除的前N行数据。
        :param file_path:
        :param n:
        :return: <class 'list'> 如：['1\n', '2\n', '3\n']
        """
        with open(file_path, 'r', encoding='utf-8-sig') as f_in:
            a = f_in.readlines()
        with open(file_path, 'w', encoding='utf-8-sig') as f_out:
            f_out.write(''.join(a[n:]))
        return a[:n]

    @staticmethod
    def get_data_from_csv(file_path, rows=None):
        """从csv文件读取数据，并以panda的dataframe格式返回"""
        try:
            df = pd.read_csv(file_path, usecols=rows, encoding='gbk')
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, usecols=rows, encoding='utf-8')
        return df
