import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
import numpy
import io
import base64


# 分类中的箱线图
def draw_box(df, value_field, y_col):
    data = pd.concat([df[value_field], df[y_col]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=df[y_col], y=value_field, data=df)
    fig.axis()
    # plt.show()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    content = buf.read()
    base64_data = base64.b64encode(content)  # 使用base64进行加密
    base64_data = base64_data.decode("utf-8")
    return base64_data


def main():
    df = pd.read_csv("../case/bank_pm_train_set.csv")
    base64_data = draw_box(df, "balance", "y")
    print(base64_data)


if __name__ == '__main__':
    main()
