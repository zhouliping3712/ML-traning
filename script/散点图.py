import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("../case/bank_pm_train_set.csv")

data = pd.concat([df['age'], df['balance']], axis=1)
data.plot.scatter(x='age', y='balance', alpha=0.3)
plt.show()

data = pd.concat([df['y'], df['balance']], axis=1)
data.plot.scatter(x='y', y='balance', alpha=0.3)
plt.show()
