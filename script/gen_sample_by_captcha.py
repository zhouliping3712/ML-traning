# -*- coding: UTF-8 -*-
"""
使用captcha lib生成验证码（前提：pip install captcha）
"""
from captcha.image import ImageCaptcha
import os
import random
import time
import json


def gen_special_img(text, file_path, width, height):
    # 生成img文件
    generator = ImageCaptcha(width=width, height=height)  # 指定大小
    img = generator.generate_image(text)  # 生成图片
    img.save(file_path)  # 保存图片


def gen_ima_by_batch(root_dir, image_suffix, characters, count, char_count, width, height):
    # 判断文件夹是否存在
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for index, i in enumerate(range(count)):
        text = ""
        for j in range(char_count):
            text += random.choice(characters)

        timec = str(time.time()).replace(".", "")
        p = os.path.join(root_dir, "{}_{}.{}".format(text, timec, image_suffix))
        gen_special_img(text, p, width, height)

        print("Generate captcha image => {}".format(index + 1))


def main():
    # 配置参数
    root_dir = "img"  # 图片储存路径
    image_suffix = "jpg"  # 图片储存后缀
    characters = "0123456789"  # 图片上显示的字符集 # characters = "0123456789abcdefghijklmnopqrstuvwxyz"
    count = 600  # 生成多少张样本
    char_count = 4  # 图片上的字符数量

    # 设置图片高度和宽度
    width = 100
    height = 60

    gen_ima_by_batch(root_dir, image_suffix, characters, count, char_count, width, height)


if __name__ == '__main__':
    main()
