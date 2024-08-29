import random
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import LinearSVC,SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier,VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
import logging

"""
HLOA算法的二进制版本改进版本1
该版本相较于初始版本，需要优化的点有：
1、 问题：对于全部欧式距离小于阈值的情况，出现对应二进制向量全部为零的情况，初始版本是用随机数据取代，
    目的只是让程序能运行下去，同时连续数据向量并未更改，依旧继续迭代；
    解决方案1：思路是要用连续数据向量反映特征选择情况，可以用另一套标准，比如重新选取一个阈值
    或者再次进行计算，直到产生有效特征选择，总之就是不能随机产生，丢失迭代的数据特征
    解决方案2：如果出现全零，是否可以理解为这个局部搜索是无效的，然后将特征选择向量进行随机重置，是否
    可以认为是进行了一次全局搜索？那么该如何将这种特征反馈给连续的种群数据呢？方案一是用策略直接进行
    一次全局搜索的计算，方案二就是以大于阈值的随机数的方法在原数据上进行累加
    元启发算法做特征选择本质上就是一个二分类问题，也就是把什么数据归为1，把什么数据归为0
2、 问题：仅用一个定死的阈值是否太过于死板了？虽然对同一个数据集同一个训练测试模型，可能存在
    一个最佳阈值，但对于不同数据集和不同训练测试模型来说，或许就需要更换阈值了
    解决方案：本着人工智能专业中智能二字，我觉得是不是可以设计一种算法，让它根据适应度的值
    自己调整阈值，可能会增加运算量，但会获得极佳的泛用性，这样也会显得更‘智能’一点，至于具体
    算法思路，可以参考深度学习里面的反向传播算法，通过测试结果，反过来对阈值进行修改
3、 问题：是否可以再对算法的局部搜索思路进行修改呢？HLOA是用的随机搜索方法，那是否可以
    尝试采用梯度下降算法的思路，来进行局部的探幽呢，就是顺着梯度降低最快的方向进行局部搜索？

以上三个问题，可以在该版本中进行尝试实现，并进行结果实验
"""

def data_Processing_split(filename,trainProportion=0.8):
    """
    用来将一个完整的数据，按照输入的比例，分成两份数据，并以csv文件的形式保存下来
    :param filename: 需要拆分的文件名
    :param trainProportion: 拆分的训练集数据的比例
    :return: 返回训练集+测试集的名字
    """
    df=pd.read_csv(filename)
    split_index=int(len(df)*trainProportion)
    train_df=df.iloc[:split_index]
    test_df=df.iloc[split_index:]
    filename_without_ext = os.path.splitext(filename)[0]
    name_train=f"{filename_without_ext}_train.csv"
    name_test=f"{filename_without_ext}_test.csv"
    train_df.to_csv(name_train,index=False)
    test_df.to_csv(name_test, index=False)
    return name_train,name_test

def data_Processing_NSL_KDD(filename):
    """
    将原始数据处理成sklearn能用的数据形式，仅限NSL-KDD数据集，其他数据集格式不对
    需要重新写代码
    :param filename: 传入原始数据集
    :return: data+target,arrary数组
    """
    df = pd.read_csv(filename)
    # 去除列名中的特殊字符：单双引号以及空格
    df = df.rename(columns=lambda x: x.replace("'", "").replace('"', '')).replace(" ", "")
    # print(df.columns)
    df_data = df.iloc[:, 5:-1]  # 先将1至4列非数特征值搁置，稍后再处理,从第5列到倒数第二列
    df_targetNonnumerical = df.iloc[:, -1]  # 选取所有行的最后一列，
    standard = 'normal'
    df_targetNumerical = []
    for i in range(len(df_targetNonnumerical)):
        if df_targetNonnumerical[i] == standard:
            df_targetNumerical.append(1)
        elif df_targetNonnumerical[i] != standard:
            df_targetNumerical.append(0)
    df_target = df_targetNumerical
    # 要把数据处理成sklearn需要的形式
    df_data = df_data.values
    df_target = np.array(df_target)
    print(f"{'数据集：'}{filename}{'，项目文件：HLOA_Binary，函数名：data_Processing'}")
    # print(df_data)
    # print(df_target)
    return df_data, df_target

def data_Processing_arrhythmia(filename):
    """
    将原始数据处理成sklearn能用的数据形式，仅限arrhythmia数据集，其他数据集格式不对
    需要重新写代码
    :param filename: 传入原始数据集
    :return: data+target,arrary数组
    """
    df = pd.read_csv(filename,header=None)
    #df.to_csv("处理之前.csv", index=False)

    #df['is_abnormal'] = (df['data_column'] == '?').astype(int)#创建一个新列来标记异常值：1表示异常，0表示正常

    # 遍历每一列，替换非数值数据为NaN
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    for col in df.columns:
        # print(df[col])
        mean_value = df[col].mean()  #计算当前列的平均值，忽略NaN
        # print(mean_value)
        df[col].fillna(mean_value, inplace=True)  #替换NaN值为平均值
    #df.to_csv("处理之后.csv", index=False)

    # for col_idx in range(df.shape[1]):
    #     # 将?替换为NaN，以便计算平均值时可以忽略它们
    #     df.iloc[:, col_idx] = df.iloc[:, col_idx].replace('?', np.nan)
    #
    #     # 计算该列（忽略NaN）的平均值
    #     mean_value = df.iloc[:, col_idx].mean(skipna=True)
    #
    #     # 如果平均值是NaN（即该列全为NaN），则可以选择一个默认值（如0），或者根据具体情况处理
    #     if pd.isna(mean_value):
    #         mean_value = 0  # 或者其他你认为合适的默认值
    #
    #     # 将NaN替换为计算出的平均值
    #     df.iloc[:, col_idx].fillna(mean_value, inplace=True)



    df_data = df.iloc[:, :-1]
    df_targetNonnumerical = df.iloc[:, -1]  # 选取所有行的最后一列，
    standard = 1
    df_targetNumerical = []
    for i in range(len(df_targetNonnumerical)):
        if df_targetNonnumerical[i] == standard:
            df_targetNumerical.append(1)
        elif df_targetNonnumerical[i] != standard:
            df_targetNumerical.append(0)
    df_target = df_targetNumerical
    # 要把数据处理成sklearn需要的形式
    df_data = df_data.values
    df_target = np.array(df_target)
    print(f"{'数据集：'}{filename}{'，项目文件：HLOA_Binary，函数名：data_Processing'}")
    # print(df_data)
    # print(df_target)
    return df_data, df_target

def Initialization(search_agents, dim, ub, lb):
    """
    根据search_agents的数量，初始化整个族群
    :param search_agents: 搜索代理的数量，或者说生成位置矩阵的行数，或者说需要生成的种群的个数
    :param dim:空间维度，指的是问题需要的解的个数，或者说生成位置矩阵的列数
    :param ub:搜索空间上界，也就是说问题的各个解的上限，长度是等于dim的
    :param lb:搜索空间下届，也就是说问题的各个解的下限，长度等于dim
    :return:返回一个位置矩阵，也就是初始化之后族群，根据初始化之后的族群开始进行搜索策略
    """
    boundary_no = ub.shape[0]
    positions = np.zeros((search_agents, dim))
    if boundary_no == 1:
        positions = np.random.rand(search_agents, dim) * (ub - lb) + lb
    else:
        for i in range(dim):
            ub_i = ub[i]
            lb_i = lb[i]
            positions[:, i] = np.random.rand(search_agents) * (ub_i - lb_i) + lb_i
    return positions


def sigmoidChange(position):
    """
    将传入位置矩阵，用sigmoid函数计算，然后返回对应行列数的新矩阵
    :param position:传入的位置矩阵
    :return:使用sigmoid函数计算后的新矩阵
    """
    # if position.ndim == 2:
    #     newArrary = np.zeros((position.shape[0], position.shape[1]))
    #     for i in range(position.shape[0]):
    #         for j in range(position.shape[1]):
    #             newArrary[i, j] = 1
    # elif position.ndim == 1:
    #     newArrary = np.zeros((len(position)))
    #     for i in range(len(position)):
    #         newArrary[i] = 1
    return 1 / (1 + np.exp(-position))


def euclideanDistance(position1, position2,logger):
    """
    根据传入的两个矩阵，计算他们相对应位置的两个数的欧氏距离，
    将对应计算完成的存放欧氏距离的矩阵返回
    :param position1: 传入np.arrary矩阵1
    :param position2:传入np.arrary矩阵2
    :param logger:日志记录logger函数
    :return:一个存放欧氏距离的矩阵
    """

    # 先判断两个矩阵的尺寸
    if position1.shape != position2.shape:
        raise SystemExit("传入矩阵尺寸不相符，无法计算，出错函数euclideanDistance")
    if position1.ndim == 2:
        euclideanDistanceArrary = np.zeros((position1.shape[0], position1.shape[1]))
        for i in range(position1.shape[0]):
            for j in range(position1.shape[1]):
                euclideanDistanceArrary[i, j] = np.sqrt(np.sum((position1[i, j] - position2[i, j]) ** 2))
    elif position1.ndim == 1:
        euclideanDistanceArrary = np.zeros((len(position1)))
        for i in range(len(position1)):
            euclideanDistanceArrary[i] = np.sqrt(np.sum((position1[i] - position2[i]) ** 2))

    euclideanDistanceArrary = np.array(euclideanDistanceArrary)
    # logger.info(f"欧式距离为：{euclideanDistanceArrary}")
    # print(type(euclideanDistanceArrary))
    return euclideanDistanceArrary

def k_means_simplify(euclideanDistance):
    """
    这是一个利用聚类思想，对数轴上的点按照类“内距离最小，类间距离最大”的原则进行二分类
    目的就是将数轴上的点分成两类，目前暂定，数轴上属于左边小值的那一类为0，右边大值那一类为1
    :param euclideanDistance:传入的一维矩阵
    :return:一个binaryArrary,np.arrary数组
    """
    #先做一个判断，只能对一维矩阵进行处理
    if euclideanDistance.ndim == 2:
        print(f"传入了二维矩阵，只能接受一维矩阵")
        return
    # 第一步，找到数轴上的最大值和最小值
    e_max=np.max(euclideanDistance)
    e_min=np.min(euclideanDistance)

    binaryArrary=np.zeros(len(euclideanDistance))

    #第二步，遍历每一个数据，并计算其与最大值和最小值的差值，归属于差值较小的那一类
    for i in range(len(euclideanDistance)):
        distance_min=euclideanDistance[i]-e_min
        distance_max=e_max-euclideanDistance[i]
        if distance_max<=distance_min:#属于左边小值的那一类为0，右边大值那一类为1
            binaryArrary[i]=1
    #print(binaryArrary)
    return binaryArrary

def continuous_to_discreate(position,logger, judgementCriteria=0):
    """
    生成search_agents行dim列的二进制数组，且其数据是根据对应种群生成
    在这一部分，对特征选择向量出现全零情况之后的处理步骤进行优化
    方法1：抛弃该次迭代，重新计算，直到特性选择向量不再出现全零的情况
    方法2：用另一套标准进行计算，算出一个新的特征选择向量
    总之就是不能在迭代的过程中用随机
    :param position:种群矩阵，或者欧式距离矩阵
    :param logger:日志记录logger函数
    :param judgementCriteria:比其小则取0，大则取1，默认值是0
    :return:返回一个离散矩阵，尺寸和position对应
    """
    """
        方法1：
        判断特征选择是否全为0，如果是，则重新进行迭代，则就又有两种玩法
        1、从几种策略里面随机选择一个，这个比较好实现
        2、继续沿用上一个迭代策略重新进行计算，这样的话可能会出现死循环的情况
        其中方法1对应开头提到的思路2，解释起来，就是说本次的局部探索是一次无效探索，需要进行一次全局探索来
        让程序继续跑下去，但这样又会延伸出一个问题，就是局部探索的有效性跟阈值的选择关系比较大
    """
    """
        方法2：
        判断特征选择是否全为0，如果是，则选择一个新的评判标准，重新进行特征选择向量的生成
        1、可以根据某个公式生成一个新标准，而且必然会产生一个非全零的特征选择向量，但该方法主观性太强，需要
        找通过大量实验找到合适的公式；（可以尝试对欧氏距离做聚类，然后进行计算，再根据准确率，判断
        以哪一类为1，哪一类为0，需要对代码进行大改）
        2、随机在欧式距离向量中挑选一个数值，以此为标准，凡大于等于该数值的全部选一，这必然不是全零向量，该
        方法不需要过多的实验，主打一个简单方便；
    """
    # search_agents=position.shape[0]
    # dim=position.shape[1]

    # 下面是计算位置对应的sigmoid函数值
    # position=sigmoidChange(position)
    if position.ndim == 2:
        binaryArrary = np.zeros((position.shape[0], position.shape[1]))
        for i in range(position.shape[0]):
            for j in range(position.shape[1]):
                if position[i, j] >= judgementCriteria:
                    binaryArrary[i, j] = 1
                else:
                    binaryArrary[i, j] = 0
    elif position.ndim == 1:
        binaryArrary = np.zeros((len(position)))
        for i in range(len(position)):
            if position[i] >= judgementCriteria:
                binaryArrary[i] = 1
            else:
                binaryArrary[i] = 0

    #下面实现的是方法2中的第一个方法,对欧氏距离使用聚类k-means，k=2的思路做一个简单的二分类
    if (binaryArrary == 0).all():
        #logger.info(f"因为binaryArrary:{binaryArrary}全部是0，开始对欧氏距离进行聚类计算")
        logger.info(f"因为binaryArrary:全部是0，开始对欧氏距离进行聚类计算")
        binaryArrary=k_means_simplify(position)

    # #下面实现的是方法2中的第二个方法，随机选取原始数据中的某个值做阈值
    # if (binaryArrary == 0).all():
    #     logger.info(f"因为binaryArrary:{binaryArrary}全部是0，随机选取原始数据中的某个值作为阈值，重新进行向量生成")
    #     if position.ndim == 2:
    #         judgementCriteria=random.choice(random.choice(position))
    #         # print(f"维度2")
    #         # print(f"new_judgementCriteria:{judgementCriteria}")
    #         binaryArrary = np.zeros((position.shape[0], position.shape[1]))
    #         for i in range(position.shape[0]):
    #             for j in range(position.shape[1]):
    #                 if position[i, j] >= judgementCriteria:
    #                     binaryArrary[i, j] = 1
    #                 else:
    #                     binaryArrary[i, j] = 0
    #     elif position.ndim == 1:
    #         judgementCriteria = random.choice(position)
    #         # print(f"维度1")
    #         # print(f"new_judgementCriteria:{judgementCriteria}")
    #         binaryArrary = np.zeros((len(position)))
    #         for i in range(len(position)):
    #             if position[i] >= judgementCriteria:
    #                 binaryArrary[i] = 1
    #             else:
    #                 binaryArrary[i] = 0

    # #下面实现的是方法2中的第二个方法，使用原始数据中的中位数做阈值
    # if (binaryArrary == 0).all():
    #     logger.info(f"因为binaryArrary:{binaryArrary}全部是0，选取原始数据中的中位数做阈值，重新进行向量生成")
    #     if position.ndim == 2:
    #         judgementCriteria=np.median(position)
    #         logger.info(f"维度2")
    #         # print(f"new_judgementCriteria:{judgementCriteria}")
    #         binaryArrary = np.zeros((position.shape[0], position.shape[1]))
    #         for i in range(position.shape[0]):
    #             for j in range(position.shape[1]):
    #                 if position[i, j] >= judgementCriteria:
    #                     binaryArrary[i, j] = 1
    #                 else:
    #                     binaryArrary[i, j] = 0
    #     elif position.ndim == 1:
    #         judgementCriteria = np.median(position)
    #         # print(f"维度1")
    #         # print(f"new_judgementCriteria:{judgementCriteria}")
    #         binaryArrary = np.zeros((len(position)))
    #         for i in range(len(position)):
    #             if position[i] >= judgementCriteria:
    #                 binaryArrary[i] = 1
    #             else:
    #                 binaryArrary[i] = 0

    # logger.info(f"binaryArrary:{binaryArrary},{type(binaryArrary)}")

    return binaryArrary


def modified_formula(positions):
    """
    输入[0,2π]，输出[0,1]，再输入需要加入一个修正公式（本次选用y=x*2π），
    让[0,1]再次变成[0,2π]，才能保证迭代正常，目前来说暂时用不到
    :param positions: 一轮迭代计算完的位置矩阵
    :return: 修正之后的位置矩阵
    """
    position_aftermodified = np.zeros((positions.shape[0], positions.shape[1]))
    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            position_aftermodified[i, j] = positions[i, j] * 2 * np.pi
    return position_aftermodified


def randomGenerate(search_agents):
    """
    生成0到search_agents-1代理搜索数量之间的随机数，用于从搜索代理中获得随机索引
    :param search_agents:搜索代理的数量或者种群大小
    :return:四个互不相同的随机数
    """
    band = 1
    while band:
        r1 = round((search_agents - 1) * np.random.rand())
        r2 = round((search_agents - 1) * np.random.rand())
        r3 = round((search_agents - 1) * np.random.rand())
        r4 = round((search_agents - 1) * np.random.rand())
        if (r1 != r2) and (r2 != r3) and (r1 != r3) and (r4 != r3) and (r4 != r2) and (r1 != r4):
            band = 0
    return r1, r2, r3, r4


def getColor(colorPalette):
    """
    从colorPalette这个数组中，随机选取两个不同的数值
    :param colorPalette: 一个np.arrary数组，一行的
    :return: 两个介于0到len(colorPalette)不相等的随机数值
    """
    band = 1
    while band:
        c1 = colorPalette[np.random.randint(0, len(colorPalette))]
        c2 = colorPalette[np.random.randint(0, len(colorPalette))]
        if c1 != c2:
            band = 0
    return c1, c2


def getBinary():
    """
    随机返回0或1，评价概率
    :return:
    """
    if np.random.rand() < 0.5:
        val = 0
    else:
        val = 1
    return val


def CauchyRand(m, c):
    """
    生成一个柯西分布的随机数
    :param m: 柯西分布的位置参数，即分布的中心
    :param c:柯西分布的尺度参数，即分布的宽度
    :return:
    """
    cauchy = c * np.tan(np.pi * (np.random.rand() - 0.5)) + m
    return cauchy


def checkBoundaries(X, lb, ub):
    """
    检查搜索代理的位置是否超出了搜索空间的边界，如果超出了，就将其设置为搜索空间的边界
    :param X: 搜索代理当前的位置
    :param lb: 搜索空间的下界(目前接受一个int数，如果要传arr数组，需要改代码）
    :param ub: 搜索空间的上界（目前接受一个int数，如果要传arr数组，需要改代码）
    :return: 判定后的搜索代理的位置
    """
    # if not X.ndim == 2:
    #     raise ValueError('X must be a 2D array')

    n = X.shape[0]
    # print(f'n: {n}')
    bound = lb.shape[0]
    # print(f'bound: {bound}')
    if bound == 1:
        ubx = np.full(n, ub)
        lbx = np.full(n, lb)
    else:
        ubx = [ub[i] for i in range(n)]
        lbx = [lb[i] for i in range(n)]
    # print(f'ubx: {ubx}')
    # print(f'lbx: {lbx}')
    X = np.where(X > ub, ubx, X)
    X = np.where(X < lb, lbx, X)
    return X


def mimicry(Xbest, X, Max_iter, SearchAgents, t):
    """
    实现角蜥蜴的伪装策略，也即隐蔽策略
    :param Xbest:第t次的最好的搜索代理
    :param X:当前的位置矩阵，全部信息
    :param Max_iter:最大迭代次数
    :param SearchAgents:搜索代理的数量
    :param t:当前迭代次数
    :return:第t+1次最新的代理搜索位置，也就是说返回一个1行dim列的向量
    """
    colorPalette = np.array(
        [0, 0.00015992, 0.001571596, 0.001945436, 0.002349794, 0.00353364, 0.0038906191, 0.003906191, 0.199218762,
         0.19999693, 0.247058824, 0.39999392, 0.401556397, 0.401559436, 0.498039216, 0.498046845, 0.499992341,
         0.49999997, 0.601556397, 0.8, 0.900000447, 0.996093809, 0.996109009, 0.996872008, 0.998039245, 0.998046875,
         0.998431444, 0.999984801, 0.999992371, 1])
    Delta = 2
    r1, r2, r3, r4 = randomGenerate(SearchAgents)
    c1, c2 = getColor(colorPalette)
    Xnext = Xbest + (Delta - Delta * t / Max_iter) * (c1 * (np.sin(X[r1, :]) - np.cos(X[r2, :])) \
                                                      - ((-1) ** getBinary()) * c2 * (
                                                                  np.cos(X[r3, :]) - np.sin(X[r4, :])))
    return Xnext


def Skin_darkening_or_lightening(Xbest, X, SearchAgents):
    """
    实现角蜥蜴皮肤的变量和变暗策略
    :param Xbest:找到的最好代理
    :param X:搜索代理矩阵，包含全部信息
    :param SearchAgents:搜索代理数量
    :return:找到的最差的代理
    """
    darkening = [0.0, 0.4046661]
    lightening = [0.5440510, 1.0]

    dark1 = darkening[0] + (darkening[1] - darkening[0]) * np.random.rand()
    dark2 = darkening[0] + (darkening[1] - darkening[0]) * np.random.rand()

    light1 = lightening[0] + (lightening[1] - lightening[0]) * np.random.rand()
    light2 = lightening[0] + (lightening[1] - lightening[0]) * np.random.rand()

    r1, r2, r3, r4 = randomGenerate(SearchAgents)

    if getBinary() == 1:
        # Xworst指找到的最差的代理
        Xworst = Xbest + light1 * np.sin((X[r1, :] - X[r2, :]) / 2) \
                 - ((-1) ** getBinary()) * light2 * np.sin((X[r3, :] - X[r4, :]) / 2)
    else:
        Xworst = Xbest + dark1 * np.sin((X[r1, :] - X[r2, :]) / 2) \
                 - ((-1) ** getBinary()) * dark2 * np.sin((X[r3, :] - X[r4, :]) / 2)

    return Xworst


def shootBloodstream(Xbest, X, Max_iter, t):
    """
    实现角蜥蜴的血液射击策略
    :param Xbest:第t次的最好的搜索代理
    :param X:  本轮循环的搜索代理
    :param Max_iter: 最大迭代次数
    :param t:当前迭代次数
    :return:第t+1次最新的代理搜索位置，也就是说返回一个1行dim列的向量
    """
    g = 0.009807  # 9.807 m/s2
    epsilon = 1E-6
    Vo = 1  # 1E-2
    Alpha = np.pi / 2

    Xnext = (Vo * np.cos(Alpha * t / Max_iter) + epsilon) * Xbest + \
            (Vo * np.sin(Alpha - Alpha * t / Max_iter) - g + epsilon) * X
    return Xnext


def randomWalk(Xbest, X):
    """
    实现角蜥蜴的随机行走策略
    :param Xbest:第t次的最好的搜索代理
    :param X:本轮循环的搜索代理
    :return:第t+1次最新的代理搜索位置，也就是说返回一个1行dim列的向量
    """
    e = CauchyRand(0, 1)
    walk = -1 + 2 * np.random.rand()  # -1 < d < 1
    Xnext = Xbest + walk * (0.5 - e) * X
    return Xnext


def alpha_melanophore(fit, min, max):
    """
    计算α促黑激素
    :param fit:搜索代理当前的适应度值（目前fit的形态规格不清楚，应该是SearchAgents行1列的数组）
    :param min:当前最优的适应度值
    :param max:当前最差的适应度值
    :return:每个搜索代理对应的α促黑激素组成的向量（目前返回值好像不太对）
    """
    return np.array([(max - fit[i]) / (max - min) for i in range(fit.shape[0])])


def remplaceSearchAgent(Xbest, X, SearchAgents_no):
    """
    替换搜索代理，根据最好的搜索代理和当前的搜索代理，生成新的搜索代理
    :param Xbest: 最佳搜索代理
    :param X: 当前搜索代理的位置
    :param SearchAgents_no: 搜索代理的数量
    :return:生成的新的搜索代理，一行dim列的向量
    """
    band = 1
    while band:
        r1 = np.random.randint(0, SearchAgents_no)
        r2 = np.random.randint(0, SearchAgents_no)
        if r1 != r2:
            band = 0

    Xnew = Xbest + (X[r1, :] - ((-1) ** getBinary()) * X[r2, :]) / 2
    return Xnew


def fitness(binaryArrary_row, errorRate, a=0.5, b=0.5):
    """
    目标函数，作用就是找出令该目标函数取最小值的解，每个搜索代理都需要计算一次
    :param binaryArrary_row:需要计算的搜索代理对应的那一行的二进制向量，也就是一个1行dim列的数组
    :param errorRate:该行特征向量训练之后，再进行预测测试得到的错误率
    :param a:计算目标函数时错误率的权重值，初始0.5
    :param b:计算目标函数时选择特征数量/全部特征数量比值的权重，初始0.5
    :return:返回计算好的目标函数
    """
    selectfeature = 0
    # for element in binaryArrary_row:
    #     if element==1:
    #         selectfeature=selectfeature+1
    selectfeature = np.sum(binaryArrary_row)
    # print(f"selectfeature={selectfeature}")
    return a * errorRate + b * (selectfeature / len(binaryArrary_row))


def feature_selection(binaryArray_row, train_data):
    """
    根据传入的二进制解向量，在训练数据中选择特征，然后将选择后的特征返回
    :param binaryArray_row:传入一个二进制解向量，也就是一行dim列的向量
    :param train_data:传入完整的训练集
    :return:返回选择好的训练集
    """
    feature_number = 0
    feature_selected = []
    feature_selected_index = []
    if len(binaryArray_row) != train_data.shape[1]:
        print(f"解向量的维数{len(binaryArray_row)}和数据的列数{train_data.shape[1]}不相同")
        return
    # 先搞个中间列表，记录下来哪些列被选中，然后再根据记录列表来挑选特征
    for i in range(train_data.shape[1]):
        if binaryArray_row[i] == 1:
            feature_selected_index.append(i)
            feature_number = feature_number + 1
    for row in train_data:
        feature = [row[index] for index in feature_selected_index]
        feature_selected.append(feature)

    # print(feature_selected)
    # print(type(feature_selected))
    feature_selected = np.array(feature_selected)
    # print(feature_selected)
    # print(type(feature_selected))
    return feature_selected


def HLOA(SearchAgents_no, Max_iter, lb, ub, dim, judgementCriteria, a, filename_train, filename_test,logger,clf):
    """
    实现角蜥蜴优化算法
    :param SearchAgents_no:搜索代理的数量
    :param Max_iter:最大迭代次数
    :param lb:下届
    :param ub:上界
    :param dim:问题纬度
    :param judgementCriteria:二进制转换的标准
    :param a:适应度函数中错误率对应的权重，特征数量的权重为1-a
    :param filename_train:用到的训练集
    :param filename_test:用到的测试集
    :param logger:记录的logger函数
    :param clf:传入的训练和预测使用的模型
    :return:
    """



    # 初始化搜索代理的位置，行数是搜索代理的数量SearchAgents_no，列数是搜索空间的维度dim
    Positions = Initialization(SearchAgents_no, dim, ub, lb)
    #logger.info(f"positons={Positions}")
    # 生成搜索代理对应的二进制矩阵，因为第一轮矩阵没有对应的前代，无法计算欧氏距离，所以直接转化
    binaryArrary = continuous_to_discreate(Positions,logger, 0)
    # 先进行一遍训练加测试，计算出目标函数以及最好最差
    train_data, train_target = data_Processing_arrhythmia(filename_train)
    test_data, test_target = data_Processing_arrhythmia(filename_test)
    Fitness = []  # 存储目标函数值的数组
    # 这是整个迭代算法的第一步，方法一可以用全部训练集进行训练，方法二进行特征选择，选择完进行训练
    # 因为要找到最佳和最差代理，所以只能用第二个方法
    for row in binaryArrary:
        # 对训练集和测试集进行特征选择
        train_data_selected = feature_selection(row, train_data)
        test_data_selected = feature_selection(row, test_data)
        # 接下来开始训练，开始测试
        # 下面是用knn计算的
        # clf = KNeighborsClassifier(n_neighbors=5)

        #下面是决策树
        #clf=tree.DecisionTreeClassifier(random_state=0)

        clf.fit(train_data_selected, train_target)
        y_predict = clf.predict(test_data_selected)
        error_rate = 1 - accuracy_score(test_target, y_predict)
        # print(f"error_rate={error_rate}")
        # 接下来开始计算适应度的值（即目标函数值）
        Fitness.append(fitness(row, error_rate, a, 1 - a))
        logger.info(f"feature_number={train_data_selected.shape[1]}")
        logger.info(f"error_rate={error_rate}")

    #logger.info(f"Fitness={Fitness}")
    Fitness = np.array(Fitness)
    vMin_idx = np.argmin(Fitness)  # 返回最小值的索引
    vMin = Fitness[vMin_idx]
    theBestVct = Positions[vMin_idx, :]  # 将最小值定为最好的搜索代理
    # print(f"theBestVct={theBestVct}")
    vMax_idx = np.argmax(Fitness)  # 返回最大值的索引，且是最差的搜索代理
    Convergence_curve = np.zeros(Max_iter)
    Convergence_curve[0] = vMin  # 收敛曲线，记录每一轮的最佳值，也就是目标函数最小值，用于分析

    alphaMelanophore = alpha_melanophore(Fitness, Fitness[vMin_idx], Fitness[vMax_idx])
    # alphaMelanophore注定会有一个1一个0,1是fit[i]取到min，0是fit[i]取到max
    #logger.info(f"alphaMelanophore={alphaMelanophore}")
    # 下列大循环就是迭代过程
    for t in range(1, Max_iter + 1):
        # 这个小循环每次迭代过程中，每个搜索代理进行的算法计算
        logger.info(f"第{t}轮迭代中")
        for r in range(SearchAgents_no):
            #logger.info(f"第{r + 1}个代理")
            if np.random.rand() > 0.5:  # 50%的概率，执行伪装策略或者血液射击与随机行走策略
                v = mimicry(theBestVct, Positions, Max_iter, SearchAgents_no, t)
                #logger.info(f"进入了伪装策略")
            else:
                if t % 2 == 0:  # 若迭代次数是偶数，执行血液射击策略
                    v = shootBloodstream(theBestVct, Positions[r, :], Max_iter, t)
                    #logger.info(f"执行了血液射击策略")
                else:  # 若迭代次数是奇数，执行随机行走策略
                    v = randomWalk(theBestVct, Positions[r, :])
                    #logger.info(f"执行了随机行走策略")
            Positions[vMax_idx, :] = Skin_darkening_or_lightening(theBestVct, Positions, SearchAgents_no)
            #logger.info(f"最差是第{vMax_idx + 1}个代理，执行皮肤变化策略")
            if alphaMelanophore[r] <= 0.3:
                # print(f"第{r+1}个alphaMelanophore值是{alphaMelanophore[r]}，执行替换逃离策略")
                v = remplaceSearchAgent(theBestVct, Positions, SearchAgents_no)

            # v = checkBoundaries(v, lb, ub)
            # 以上已经确定好了一个搜索代理
            # 接下来计算新的搜索代理的适应度函数
            # print(v)
            # 先将得到的新代理二进制化，这个临界值有待商榷

            #logger.info(f"第{t}轮迭代中的，第{r + 1}个代理是：{v}")

            # 下面代码计算欧氏距离并转换二进制版本
            v_euclideanDistance = euclideanDistance(v, Positions[r, :],logger)
            v_binary = continuous_to_discreate(v_euclideanDistance,logger, judgementCriteria)

            # 下面代码利用sigmoid函数转化二进制版本
            # v_binary=continuous_to_discreate(v,0)

            # 用得到的二进制进行特征选择，然后训练并计算适应度函数
            train_data_selected = feature_selection(v_binary, train_data)
            test_data_selected = feature_selection(v_binary, test_data)
            # 接下来开始训练，开始测试
            #下面是用knn进行计算的
            #knn = KNeighborsClassifier(n_neighbors=5)

            #下面是用决策树进行计算
            #clf=tree.DecisionTreeClassifier(random_state=0)

            clf.fit(train_data_selected, train_target)
            y_predict = clf.predict(test_data_selected)
            error_rate = 1 - accuracy_score(test_target, y_predict)
            # 接下来开始计算适应度的值（即目标函数值）
            Fnew = fitness(v_binary, error_rate, a, 1 - a)

            # 接下来需要对新的搜索代理进行公式修正，以免其数值收敛于0

            if Fnew <= Fitness[r]:
                Positions[r, :] = v
                Fitness[r] = Fnew

            if Fnew <= vMin:
                theBestVct = v
                vMin = Fnew

            logger.info(f"feature_number={train_data_selected.shape[1]}")
            logger.info(f"error_rate={error_rate}")
            logger.info(f"fnew={Fnew}")
            #logger.info(f"Fitness={Fitness}")
            #logger.info('   ')

        vMax_idx = np.argmax(Fitness)
        alphaMelanophore = alpha_melanophore(Fitness, vMin, Fitness[vMax_idx])
        Convergence_curve[t - 1] = vMin

    logger.info(f"Convergence_curve={Convergence_curve}")
    logger.info(f"vMin={vMin}")
    #logger.info(f"theBestVct={theBestVct}")

    return vMin, theBestVct, Convergence_curve


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO,
    #                     filename='实验记录.txt',
    #                     filemode='w',
    #                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # logger = logging.getLogger(__name__)
    # console_handler = logging.StreamHandler()
    # console_handler.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

    # 创建日志记录器
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # 设置日志级别

    # 创建文件处理器
    file_handler = logging.FileHandler('实验记录.txt', mode='w')
    file_handler.setLevel(logging.INFO)
    # 为文件处理器设置格式化器，不包含日期、时间等前缀
    file_formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # 为控制台处理器设置格式化器，包含日期、时间、模块名和日志级别
    console_formatter = logging.Formatter('%(asctime)s-%(name)-12s-%(levelname)-8s-%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    ub = np.array([20])
    lb = np.array([-20])
    #pos1=Initialization(4,6,ub,lb)
    # pos2=np.array([[1,1,2],[10,20,30],[5,6,7],[19,20,21]])
    # pos3=np.array([7,6,5,4,3,1])
    # 下面是用knn进行计算的
    # clf = KNeighborsClassifier(n_neighbors=6)
    # 下面是用决策树进行计算
    #clf=RandomForestClassifier(min_samples_split=5,min_samples_leaf=10)
    # 下面使用极端随机树
    #clf=ExtraTreesClassifier()
    # 下面使用adaboost模型
    clf=AdaBoostClassifier(n_estimators=100,random_state=0)
    # 下面使用bagging模型
    #clf=BaggingClassifier(n_estimators=100,random_state=0)

    name_train, name_test = data_Processing_split('arrhythmia.data', 0.8)
    for i in range(7):
        logger.info(f"第{i+1}次实验")
        HLOA(50, 100, lb, ub, 279, 15, 0.9, name_train, name_test,logger,clf)

    # results=continuous_to_discreate(pos3,5)
    # print(f"results:{results}")