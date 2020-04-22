# coding=utf-8

import numpy as np
from hmmlearn import hmm

states = ['one', 'two', 'three']  # 隐藏状态
n_states = len(states)  # 隐藏状态长度
observations = ['A', 'B', 'C', 'D']  # 观测值的状态
# 数据 2 选 1
# ------------------------- W1 ----------------------------------------------------------
x1 = [[0], [0], [1], [1], [2], [2], [3], [3]]  # AABBCCDD       # 定义训练序列  ABCD对应0123
x2 = [[0], [1], [1], [2], [1], [1], [3], [3]]  # ABBCBBDD
x3 = [[0], [2], [1], [2], [1], [2], [3]]  # ACBCBCD
x4 = [[0], [3]]  # AD
x5 = [[0], [2], [1], [2], [1], [0], [1], [2], [3], [3]]  # ACBCBABCDD
x6 = [[1], [0], [1], [0], [0], [3], [3], [3]]  # BABAADDD
x7 = [[1], [0], [1], [2], [3], [2], [2]]  # BABCDCC
x8 = [[0], [1], [3], [1], [1], [2], [2], [3], [3]]  # ABDBBCCDD
x9 = [[0], [1], [0], [0], [0], [2], [3], [2], [2], [3]]  # ABAAACDCCD
x10 = [[0], [1], [3]]  # ABD
# -------------------------- W2 ----------------------------------------------------------
y1 = [[3], [3], [2], [2], [1], [1], [0], [0]]  # DDCCBBAA
y2 = [[3], [3], [0], [1], [2], [1], [0]]  # DDABCBA
y3 = [[2], [3], [2], [3], [2], [1], [0], [1], [0]]  # CDCDCBABA
y4 = [[3], [3], [1], [1], [0]]  # DDBBA
y5 = [[3], [0], [3], [0], [2], [1], [1], [0], [0]]  # DADACBBAA
y6 = [[2], [3], [3], [2], [2], [1], [0]]  # CDDCCBA
y7 = [[1], [3], [3], [1], [2], [0], [0], [0], [0]]  # BDDBCAAAA
y8 = [[1], [1], [0], [1], [1], [3], [3], [3], [2], [3]]  # BBABBDDDCD
y9 = [[3], [3], [0], [3], [3], [1], [2], [0], [0]]  # DDADDBCAA
y10 = [[3], [3], [2], [0], [0], [0]]  # DDCAAA
#
X = np.concatenate([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
Y = np.concatenate([y1, y2, y3, y4, y5, y6, y7, y8, y9, y10])
lengths_X = [len(x1), len(x2), len(x3), len(x4), len(x5),
             len(x6), len(x7), len(x8), len(x9), len(x10)]
lengths_Y = [len(y1), len(y2), len(y3), len(y4), len(y5),
             len(y6), len(y7), len(y8), len(y9), len(y10)]

# 计算后验概率
seen_list2 = [1, 0, 3, 1, 3, 2, 1, 0]  # BADBDCBA
seen = np.array([seen_list2]).T
start_probability = np.array([0, 0, 1])  # 定义隐状态先验概率
transition_probability = np.array([[1, 0, 0],  # 定义转移概率先验
                                   [0, 1, 0],
                                   [0, 0, 1]])
emission_probability = np.array([[0.1, 0.4, 0.2, 0.3],
                                 [0.3, 0.3, 0.1, 0.3],
                                 [0.2, 0.3, 0.1, 0.4]])
model1 = hmm.MultinomialHMM(n_components=n_states, n_iter=100, tol=0.001,
                            startprob_prior=start_probability, transmat_prior=transition_probability)  # 定义模型
model1.emissionprob_ = emission_probability
model2 = hmm.MultinomialHMM(n_components=n_states, n_iter=100, tol=0.001,
                            startprob_prior=start_probability, transmat_prior=transition_probability)  # 定义模型
model2.emissionprob_ = emission_probability
model1.fit(X, lengths_X)
model2.fit(Y, lengths_Y)
print(model1.startprob_)
print(model2.startprob_)
print(model1.score(seen))
print(model2.score(seen))
# while 1:
#     model1.fit(X, lengths_X)
#     model2.fit(Y, lengths_Y)
#     if model1.score(seen)==model2.score(seen):
#         print(model1.startprob_)
#         print(model2.startprob_)
#         break




