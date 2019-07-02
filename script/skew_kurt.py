import pandas as pd


df = pd.read_csv("../case/bank_pm_train_set.csv")

col_name_list = list(df.columns.values)
skew_result = list(df.skew())

co = df.corr()

print(co)

