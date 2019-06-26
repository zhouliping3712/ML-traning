import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from sklearn.model_selection import train_test_split
import xgboost
from sklearn import metrics


def split_column(df, y="y"):
    X = df.drop(y, axis=1)
    y = pd.DataFrame(df[y], columns=[y])
    return X, y


def dumb_columns(df, columns=None):
    if not columns:
        col_name_list = df.columns.values

        obj_list = []
        for index, d in enumerate(df.dtypes):
            if d == "object":
                obj_list.append(col_name_list[index])

        columns = obj_list

    for column_name in columns:
        # 获得训练集和测试集的所有分类并排序，保持每次运行程序时哑变量数字代表的类型一致
        all_class = list(set(df[column_name]))
        all_class.sort()
        class_mapping = {label: idx for idx, label in enumerate(all_class)}
        # 数字映射到每一个类型
        df[column_name] = df[column_name].map(class_mapping).astype(int)
    return df


def main():
    df = pd.read_csv(r"D:\dataset\提供银行精准营销解决方案\bank_train_set.csv")
    df = df.drop(['ID', 'day', 'month'], axis=1)
    df = dumb_columns(df)

    print(df[:10])

    X, y = split_column(df, y="y")

    Xtrain, Xtest, ytrian, ytest = train_test_split(X, y, test_size=0.3, random_state=27)

    # param_grid = {
    #     'min_samples_leaf': np.arange(1, 20, 1),
    #     'n_estimators': np.arange(1, 300, 1),
    #     'max_depth': np.arange(1, 300, 1),
    # }

    xgb_model = xgboost.XGBClassifier(nthread=15)

    # xgb_model.fit(Xtrain, ytrian)
    # y_pred = xgb_model.predict(Xtest)
    # r = metrics.accuracy_score(y_pred, ytest)
    # print(r)

    cv_split = ShuffleSplit(n_splits=6, train_size=0.7, test_size=0.2)

    param_grid = {
        "max_depth": [4, 5, 6, 7],
        "learning_rate": np.linspace(0.03, 0.3, 10),
        "n_estimators": [100, 200]
    }

    grid = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=cv_split, scoring='neg_mean_squared_error')

    grid.fit(Xtrain, ytrian.values.ravel())

    print(grid.best_params_)
    print(grid.best_score_)


if __name__ == '__main__':
    main()