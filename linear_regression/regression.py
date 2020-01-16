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
        rate {float} -- 学习速率
        maxLoop {int} -- 最大循环次数
        espilon {float} -- 收敛精度
        X {matrix} -- 样本矩阵
        y {vector} -- 标签向量
    
    Returns:
        theta, errors, thetas -- [参数向量、错误列表，参数矩阵]
    """
    m, n = X.shape
    # 初始化theta
    theta = np.zeros((n, 1))
    converged = False
    error = float('inf')
    errors = []
    thetas = {x: [theta[x, 0]] for x in range(n)}
    for tmp in range(maxLoop):
        if converged:
            break
        for j in range(n):
            deriv = (y - X * theta).T * X[:, j] / m
            theta[j, 0] = theta[j, 0] + rate * deriv
            thetas[j].append(theta[j, 0])
        error = J(theta, X, y)
        errors.append(error[0, 0])
        if error[0, 0] < espilon:
            converged = True
    return theta, errors, thetas


@exeTime
def sgd(rate, maxLoop, epsilon, X, y):
    """随机梯度下降法
    
    Arguments:
        rate {float} -- 学习速率
        maxLoop {int} -- 最大循环次数
        espilon {float} -- 收敛精度
        X {matrix} -- 样本矩阵
        y {vector} -- 标签向量
    
    Returns:
        theta, errors, thetas -- [参数向量、错误列表，参数矩阵]
    """
    m, n = X.shape
    # 初始化theta
    theta = np.zeros((n, 1))
    converged = False
    error = float('inf')
    errors = []
    thetas = {x: [theta[x, 0]] for x in range(n)}
    for tmp in range(maxLoop):
        if converged:
            break
        for i in range(m):
            if converged:
                break
            diff = y[i, 0] - h(theta, X[i].T)
            for j in range(n):
                theta[j, 0] = theta[j, 0] + rate * diff * X[i, j]
                thetas[j].append(theta[j, 0])
            error = J(theta, X, y)
            errors.append(error[0, 0])
            if error[0, 0] < epsilon:
                converged = True
    return theta, errors, thetas


def standardize(X):
    """特征标准化处理
    
    Arguments:
        X {matrix} -- 样本标签
    
    Returns:
        X -- 标准化之后的样本标签
    """
    m, n = X.shape
    for j in range(n):
        features = X[:, j]
        meanVal = features.mean(axis=0)
        std = features.std(axis=0)
        if std != 0:
            X[:, j] = (features - meanVal) / std
        else:
            X[:, j] = 0
    return X


def normalize(X):
    """特征归一化处理
    
    Arguments:
        X {matrix} -- 样本标签
    
    Returns:
        X -- 归一化之后的样本标签
    """
    m, n = X.shape
    for j in range(n):
        features = X[:, j]
        minVal = features.min(axis=0)
        maxVal = features.max(axis=0)
        diff = maxVal - minVal
        if diff != 0:
            X[:, j] = (features - minVal) / diff
        else:
            X[:, j] = 0
    return X


def JLwr(theta, X, y, x, c):
    """局部加权线性回归的代价函数计算式
    
    Arguments:
        theta {vector} -- 相关系数矩阵
        X {matrix} -- 样本集矩阵
        y {matrix} -- 标签集矩阵
        x {vector} -- 待预测输入
        c {float} -- tau
    
    Returns:
        summerize -- 预测代价
    """
    m, n = X.shape
    summerize = 0
    for i in range(m):
        diff = (X[i] - x) * (X[i] - x).T
        w = np.exp(-diff / (2 * c * c))
        predictDiff = np.power(y[i] - X[i] * theta, 2)
        summerize += w * predictDiff
    return summerize


@exeTime
def lwr(rate, maxLoop, epsilon, X, y, x, c=1):
    m, n = X.shape
    # 初始化theta
    theta = np.zeros((n, 1))
    converged = False
    error = float('inf')
    errors = []
    thetas = {x: [theta[x, 0]] for x in range(n)}
    for tmp in range(maxLoop):
        if converged:
            break
        for j in range(n):
            deriv = (y-X*theta).T*X[:, j]/m
            theta[j, 0] = theta[j, 0]+rate*deriv
            thetas[j].append(theta[j, 0])
        error = JLwr(theta, X, y, x, c)
        errors.append(error[0, 0])
        # 如果已经收敛
        if(error[0, 0] < epsilon):
            converged = True
    return theta, errors, thetas