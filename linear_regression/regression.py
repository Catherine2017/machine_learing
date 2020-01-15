import numpy as np
import time
from functools import wraps

def exeTime(func):
    """耗时计算装饰器
    
    Arguments:
        func {func} -- 程序名称
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        back = func(*args, **kwargs)
        return back, time.time() - t0
    return wrapper


def loadDataSet(filename):
    """读取文件数据
    
    Arguments:
        filename {str} -- 文件名称
    """
    X, y = [], []
    with open(filename) as rd:
        for line in rd:
            curLine = [float(x) for x in line.rstrip('\n').split('\t')]
            X.append(curLine[:1])
            y.append(curLine[-1])
    return np.mat(X), np.mat(y).T


def h(theta, x):
    """预测函数
    
    Arguments:
        theta {vector} -- 参数向量
        x {vector} -- 特征向量
    """
    return (theta.T * x)[0, 0]
    

def J(theta, X, y):
    """代价函数计算
    
    Arguments:
        theta {vector} -- 参数向量
        X {matrix} -- 特征矩阵
        y {vector} -- 标签向量
    """
    m = len(X)
    return (X * theta - y).T * (X * theta - y) / (2 * m)
    

@exeTime
def bgd(rate, maxLoop, espilon, X, y):
    """批量梯度下降法
    
    Arguments:
        rate {int} -- 学习速率
        maxLoop {int} -- 最大迭代次数
        espilon {float} -- 收敛精度
        X {matrix} -- 样本矩阵
        y {vector} -- 标签向量
    """