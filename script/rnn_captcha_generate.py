import pycapt
from PIL import Image
import time


for i in range(200):
    name,img = pycapt.do_captcha(
            my_str_list=['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
            width=80,
            height=40,
            num_of_str=4,
            font=15,
            gray_value=255,
            font_family='CHTN_.TTF')

    tst = str(time.time())
    img_name = "img/{}_{}.jpg".format(''.join(name), tst)
    print("[{}] save {}".format(i, img_name))
    img.save(img_name)

