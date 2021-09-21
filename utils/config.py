# -*- coding: utf-8 -*-
# @Time: 2021/9/12 11:19
# @Author: yuyinghao
# @FileName: config.py
# @Software: PyCharm

import os
import configparser

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', '/')
config_path = root_path + '/config/'
property_path = root_path + '/config/properties.ini'
project_root_path = root_path


def get_abs_path(path, root_path_defined=root_path):
    """
    获取绝对路径（以保证文件路径不出错）
    :param path:以root_path_defined为根目录的相对路径
    :param root_path_defined:根路径的绝对路径，默认为项目根路径。
    :return:path（相对路径）的绝对路径
    """
    return os.path.normpath(os.path.join(root_path_defined, path))


class Config(object):
    def __init__(self, ini_path=property_path, section=None):
        """
        注意: 配置参数获取出来，默认为string类型的数据格式。使用时，需要转换。
        :param ini_path: 配置文件路径地址
        :param section:
        """
        self.config = configparser.ConfigParser()
        self.ini_path = ini_path
        self.section = section

        if not self.ini_path:
            pass
        elif os.path.exists(self.ini_path):
            self.config.read(self.ini_path, encoding='utf-8')
        else:
            raise FileNotFoundError("FileNotFound:ini_path = {}".format(ini_path))

        if self.section:
            if not self.config.has_section(self.section):
                raise Exception("section = '{}' is not exists in '{}'.".format(section, self.ini_path))

    def get_section_list(self):
        """获取配置文件中所有的section名，返回list列表"""
        return self.config.sections()

    def get_section_params(self, section=''):
        """获取section下的所有参数，返回dict字典"""
        params_dict = {}

        if section:
            self.section = section
        if self.config.has_section(self.section):
            params = self.config.items(section=self.section)
            for item in params:
                params_dict[item[0]] = item[1]
            return params_dict
        else:
            raise Exception("section = '{}' is not exists in '{}'.".format(section, self.ini_path))

    def add_section(self, section):
        """
        添加section
        :param section:
        :return:
        """
        if not self.config.has_section(section):  # 检查是否存在section
            self.config.add_section(section)

    def set_option(self, option, value, section=''):
        """
        修改某个option的值，如果不存在该option 则会创建
        :param option: section下的属性名
        :param value: 属性的值
        :param section:
        :return:
        """
        if not section:
            section = self.section

        if section:
            self.add_section(section=section)
        else:
            raise Exception('section is null, failed to set option.')

        self.config.set(section, option, value)  # 修改db_port的值为69
        self.config.write(open(self.ini_path, "w", encoding='utf-8'))


def config_initial():
    """
    初始化"项目根目录绝对路径"参数
    更新至：通用参数
    文件路径不存在时，创建新文件
    :return:
    """
    if not os.path.exists(property_path):
        if not os.path.exists(config_path):
            os.mkdir(config_path)
        open(property_path, 'w', encoding='utf-8')

    config = Config(ini_path=property_path)
    config.set_option(option='root_path', value=root_path, section='general_params')


config_initial()
