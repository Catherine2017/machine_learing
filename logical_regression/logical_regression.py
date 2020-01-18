import numpy as np
import time
import os
from functools import wraps


def exeTime(func):
    """耗时计算装饰器

    Arguments:
        func {type} -- 待装饰器函数

    Returns:
        wrapper -- 装饰后的函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        back = func(*args, **kwargs)
        return back, time.time() - t0
    return wrapper


def loadDataSet(filename):
    """加载数据集

    Arguments:
        filename {str} -- 文件名称

    Returns:
        X, y -- 特征矩阵，标签矩阵
    """
    X, y = [], []
    with open(filename) as rd:
        for line in rd:
            curLine = [float(x) for x in line.rstrip(os.linesep).split('\t')]
            tmp = [1.0]
            tmp.extend(curLine[:-1])
            X.append(tmp)
            y.append(curLine[-1])
    return np.mat(X), np.mat(y).T


def sigmod(z):
    """sigmod函数

    Arguments:
        z {matrix} -- 线性函数矩阵

    Returns:
        z -- sigmod函数
    """
    return 1.0/(1.0+np.exp(-z))


def J(theta, X, y, theLambda=0):
    """逻辑回归的代价函数

    Arguments:
        theta {matrix} -- 参数矩阵
        X {matrix} -- 样本矩阵
        y {matrix} -- 标签矩阵

    Keyword Arguments:
        theLambda {float} -- 正规化参数 (default: {0})

    Returns:
        Jvalue -- 代价函数值
    """
    m, n = X.shape
    h = sigmod(X.dot(theta))
    J = (-1 / m) * (np.log(h).T.dot(y) + np.log(1 - h).T.dot((1 - y))) + \
        (theLambda / (2 * m)) * np.sum(np.square(theta[1:]))
    if np.isnan(J[0]):
        return (np.inf)
    return J.flatten()[0, 0]


@exeTime
def gradient(X, y, options):
    """逻辑回归梯度下降法
    
    Arguments:
        X {matrix} -- 特征矩阵
        y {matrix} -- 标签矩阵
        options {rate, epsilon, maxLoop, theLambda, method} -- 相关参数
    
    Returns:
        (thetas, errors, loopnum) -- 最后的参数
    """
    m, n = X.shape
    # 初始化参数矩阵
    theta = np.zeros((n, 1))
    error = float('inf')
    errors = []
    thetas = []
    rate = options.get('rate', 0.01)
    epsilon = options.get('epsilon', 0.1)
    maxLoop = options.get('maxLoop', 1000)
    theLambda = options.get('theLambda', 0)
    method = options['method']

    def _sgd(theta):
        converged = False
        for i in range(maxLoop):
            if converged:
                break
            for j in range(m):
                h = sigmod(X[j] * theta)
                diff = h - y[j]
                theta = theta - rate * (1 / m) * X[j].T * diff
                error = J(theta, X, y)
                errors.append(error)
                if error < epsilon:
                    converged = True
                    break
                thetas.append(theta)
        return thetas, errors, i + 1
        
    def _bgd(theta):
        for i in range(maxLoop):
            h = sigmod(X.dot(theta))
            diff = h - y
            theta = theta - rate*((1 / m) * X.T * diff + (theLambda / m) * np.r_[[[0]], theta[1:]])
            error = J(theta, X, y, theLambda)
            errors.append(error)
            if error < epsilon:
                break
            thetas.append(theta)
        return thetas, errors, i+1

    methods = {'sgd': _sgd, 'bgd': _bgd}
    return methods[method](theta)


def oneVsAll(X, y, options):
    """多分类参数计算
    
    Arguments:
        X {matrix} -- 特征矩阵
        y {matrix} -- 标签矩阵
        options {dict} -- 相关参数
    
    Returns:
        Thetas -- 系数矩阵
    """
    classes = set(np.ravel(y))
    Thetas = np.ones((len(classes), X.shape[1]))
    for idx, c in enumerate(classes):
        newY = np.zeros(y.shape)
        newY[np.where(y == c)] == 1
        result, timeConsumed = gradient(X, newY, options)
        thetas, errors, iterations = result
        Thetas[idx] = thetas[-1].ravel()
    return Thetas


def predictOneVsAll(X, Thetas):
    """多分类预测
    
    Arguments:
        X {matrix} -- 样本
        Thetas {matrix} -- 权值矩阵
    
    Returns:
        H -- 预测结果
    """
    H = sigmod(Thetas * X.T)
    return H