#!/usr/bin/env python
# coding: utf-8

# In[5]:


# ...
def standardize(X):
    """特征标准化处理

    Args:
        X: 样本集
    Returns:
        标准后的样本集
    """
    m, n = X.shape
    # 归一化每一个特征
    for j in range(n):
        features = X[:,j]
        meanVal = features.mean(axis=0)
        std = features.std(axis=0)
        if std != 0:
            X[:, j] = (features-meanVal)/std
        else:
            X[:, j] = 0
    return X

def normalize(X):
    """特征归一化处理

    Args:
        X: 样本集
    Returns:
        归一化后的样本集
    """
    m, n = X.shape
    # 归一化每一个特征
    for j in range(n):
        features = X[:,j]
        minVal = features.min(axis=0)
        maxVal = features.max(axis=0)
        diff = maxVal - minVal
        if diff != 0:
           X[:,j] = (features-minVal)/diff
        else:
           X[:,j] = 0
    return X


# In[ ]:
