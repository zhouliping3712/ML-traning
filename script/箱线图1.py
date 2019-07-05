import matplotlib.pyplot as plt
import pandas as pd


# 分类中的箱线图
def draw_box(df, value_field, y_col):
    result = df[y_col].value_counts()
    values = list(result.index)

    target_list = []
    for value in values:
        data = df.loc[df[y_col] == value, value_field]
        target_list.append(data)
        print(data.shape)

    plt.boxplot(target_list)
    plt.show()


def main():
    df = pd.read_csv("../case/bank_pm_train_set.csv")
    draw_box(df, "age", "y")


if __name__ == '__main__':
    main()
