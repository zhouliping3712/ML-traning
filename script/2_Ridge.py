from sklearn.linear_model import Ridge


reg = Ridge(alpha=.5)
reg.fit([[0, 0], [0, 0], [1, 1]], [0, .1, 1])

print(reg.coef_)
print(reg.intercept_)
