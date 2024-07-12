import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,accuracy_score

"""
HLOA算法的二进制版本

1、主要是用我设定的第三种办法来将该算法应用于特征选择问题，也即生成连续数据的初始种群，
2、计算完毕后转换为二进制数据，根据二进制数据进行特征选取然后进行目标函数计算；
3、目标函数计算完毕后，根据情况判定最佳和最差，然后继续用其对应的连续数据进行计算。
备注：也就是说，只有连续数据转二进制这一步，根据二进制进行特征选择，然后计算还是用对应的连续数据
备注：再说具体一点，就是每个搜索代理对应两个数据，一个连续的一个离散的，离散的是根据连续的内容生成的，计算是用连续的，特征选择是用离散的
4、输入[0,2π]，输出[0,1]，再输入需要加入一个修正公式（y=x*2π，其实并不是线性关系，但先按线性处理），让[0,1]再次变成[0,2π]，才能保证迭代正常
"""

def data_Processing(filename):
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
    df_targetNonnumerical = df.iloc[:, -1]#选取所有行的最后一列，
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
    #print(df_data)
    #print(df_target)
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
    return 1/(1+np.exp(-position))

def euclideanDistance(position1,position2):
    """
    根据传入的两个矩阵，计算他们相对应位置的两个数的欧氏距离，
    将对应计算完成的存放欧氏距离的矩阵返回
    :param position1: 传入np.arrary矩阵1
    :param position2:传入np.arrary矩阵2
    :return:一个存放欧氏距离的矩阵
    """

    #先判断两个矩阵的尺寸
    if position1.shape!=position2.shape:
        raise SystemExit("传入矩阵尺寸不相符，无法计算，出错函数euclideanDistance")
    if position1.ndim==2:
        euclideanDistanceArrary=np.zeros((position1.shape[0],position1.shape[1]))
        for i in range(position1.shape[0]):
            for j in range(position1.shape[1]):
                euclideanDistanceArrary[i,j]=np.sqrt(np.sum((position1[i,j]-position2[i,j])**2))
    elif position1.ndim==1:
        euclideanDistanceArrary = np.zeros((len(position1)))
        for i in range(len(position1)):
            euclideanDistanceArrary[i] = np.sqrt(np.sum((position1[i]-position2[i])**2))

    euclideanDistanceArrary=np.array(euclideanDistanceArrary)
    print(f"欧式距离为：{euclideanDistanceArrary}")
    #print(type(euclideanDistanceArrary))
    return euclideanDistanceArrary


def continuous_to_discreate(position,judgementCriteria=0):
    """
    生成search_agents行dim列的二进制数组，且其数据是根据对应种群生成
    也就是前面提到的，每个搜索代理对应的离散数据
    每迭代一次都需要进行连续数据二进制化
    :param position:种群矩阵
    :param judgementCriteria:比其小则取0，大则取1，默认值是1
    :return:返回一个离散矩阵，尺寸和position对应
    """
    # search_agents=position.shape[0]
    # dim=position.shape[1]

    #下面是计算位置对应的sigmoid函数值
    # position=sigmoidChange(position)
    if position.ndim==2:
        binaryArrary=np.zeros((position.shape[0],position.shape[1]))
        for i in range(position.shape[0]):
            for j in range(position.shape[1]):
                if position[i,j]>judgementCriteria:
                    binaryArrary[i,j]=1
                else:
                    binaryArrary[i,j]=0
    elif position.ndim==1:
        binaryArrary = np.zeros((len(position)))
        for i in range(len(position)):
                if position[i] > judgementCriteria:
                    binaryArrary[i] = 1
                else:
                    binaryArrary[i] = 0
    # 判断特征选择是否全为0，如果是则随机生成一个列表
    if (binaryArrary==0).all():
        print(f"因为binaryArrary:{binaryArrary}全部是0，所以产生新的随机向量")
        length=len(binaryArrary)
        binaryArrary=np.random.randint(0,2,size=length).tolist()
    print(f"binaryArrary:{binaryArrary},{type(binaryArrary)}")
    return binaryArrary

def modified_formula(positions):
    """
    输入[0,2π]，输出[0,1]，再输入需要加入一个修正公式（本次选用y=x*2π），
    让[0,1]再次变成[0,2π]，才能保证迭代正常，目前来说暂时用不到
    :param positions: 一轮迭代计算完的位置矩阵
    :return: 修正之后的位置矩阵
    """
    position_aftermodified=np.zeros((positions.shape[0],positions.shape[1]))
    for i in range(positions.shape[0]):
        for j in range(positions.shape[1]):
            position_aftermodified[i,j]=positions[i,j]*2*np.pi
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
                                                      - ((-1) ** getBinary()) * c2 *(np.cos(X[r3, :]) - np.sin(X[r4, :])))
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



def fitness(binaryArrary_row,errorRate,a=0.5,b=0.5):
    """
    目标函数，作用就是找出令该目标函数取最小值的解，每个搜索代理都需要计算一次
    :param binaryArrary_row:需要计算的搜索代理对应的那一行的二进制向量，也就是一个1行dim列的数组
    :param errorRate:该行特征向量训练之后，再进行预测测试得到的错误率
    :param a:计算目标函数时错误率的权重值，初始0.5
    :param b:计算目标函数时选择特征数量/全部特征数量比值的权重，初始0.5
    :return:返回计算好的目标函数
    """
    selectfeature=0
    # for element in binaryArrary_row:
    #     if element==1:
    #         selectfeature=selectfeature+1
    selectfeature=np.sum(binaryArrary_row)
    #print(f"selectfeature={selectfeature}")
    return a*errorRate+b*(selectfeature/len(binaryArrary_row))



def feature_selection(binaryArray_row,train_data):
    """
    根据传入的二进制解向量，在训练数据中选择特征，然后将选择后的特征返回
    :param binaryArray_row:传入一个二进制解向量，也就是一行dim列的向量
    :param train_data:传入完整的训练集
    :return:返回选择好的训练集
    """
    feature_number=0
    feature_selected=[]
    feature_selected_index=[]
    if len(binaryArray_row) != train_data.shape[1]:
        print(f"解向量的维数{len(binaryArray_row)}和数据的列数{train_data.shape[1]}不相同")
        return
    #先搞个中间列表，记录下来哪些列被选中，然后再根据记录列表来挑选特征
    for i in range(train_data.shape[1]):
        if binaryArray_row[i]==1:
            feature_selected_index.append(i)
            feature_number=feature_number+1
    for row in train_data:
        feature=[row[index] for index in feature_selected_index]
        feature_selected.append(feature)

    #print(feature_selected)
    #print(type(feature_selected))
    feature_selected=np.array(feature_selected)
    #print(feature_selected)
    #print(type(feature_selected))
    return feature_selected



def HLOA(SearchAgents_no, Max_iter, lb, ub, dim,judgementCriteria,a,filename_train,filename_test):
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
    :return:
    """

    # 初始化搜索代理的位置，行数是搜索代理的数量SearchAgents_no，列数是搜索空间的维度dim
    Positions = Initialization(SearchAgents_no, dim, ub, lb)
    print(f"positons={Positions}")
    #生成搜索代理对应的二进制矩阵，因为第一轮矩阵没有对应的前代，无法计算欧氏距离，所以直接转化
    binaryArrary=continuous_to_discreate(Positions,0)
    #先进行一遍训练加测试，计算出目标函数以及最好最差
    train_data, train_target=data_Processing(filename_train)
    test_data, test_target = data_Processing(filename_test)
    Fitness=[]#存储目标函数值的数组
    #这是整个迭代算法的第一步，方法一可以用全部训练集进行训练，方法二进行特征选择，选择完进行训练
    #因为要找到最佳和最差代理，所以只能用第二个方法
    for row in binaryArrary:
        #对训练集和测试集进行特征选择
        train_data_selected=feature_selection(row,train_data)
        test_data_selected=feature_selection(row,test_data)
        #接下来开始训练，开始测试
        knn=KNeighborsClassifier(n_neighbors=5)
        knn.fit(train_data_selected,train_target)
        y_predict=knn.predict(test_data_selected)
        error_rate=1-accuracy_score(test_target,y_predict)
        #print(f"error_rate={error_rate}")
        #接下来开始计算适应度的值（即目标函数值）
        Fitness.append(fitness(row,error_rate,a,1-a))
        print(f"feature_number={train_data_selected.shape[1]}")
        print(f"error_rate={error_rate}")

    print(f"Fitness={Fitness}")
    Fitness=np.array(Fitness)
    vMin_idx = np.argmin(Fitness)  # 返回最小值的索引
    vMin = Fitness[vMin_idx]
    theBestVct = Positions[vMin_idx, :]  # 将最小值定为最好的搜索代理
    #print(f"theBestVct={theBestVct}")
    vMax_idx = np.argmax(Fitness)  # 返回最大值的索引，且是最差的搜索代理
    Convergence_curve = np.zeros(Max_iter)
    Convergence_curve[0] = vMin  # 收敛曲线，记录每一轮的最佳值，也就是目标函数最小值，用于分析

    alphaMelanophore = alpha_melanophore(Fitness, Fitness[vMin_idx], Fitness[vMax_idx])
    #alphaMelanophore注定会有一个1一个0,1是fit[i]取到min，0是fit[i]取到max
    print(f"alphaMelanophore={alphaMelanophore}")
    #下列大循环就是迭代过程
    for t in range(1, Max_iter+1):
        #这个小循环每次迭代过程中，每个搜索代理进行的算法计算
        print(f"第{t}轮迭代中")
        for r in range(SearchAgents_no):
            print(f"第{r+1}个代理")
            if np.random.rand() > 0.5:  # 50%的概率，执行伪装策略或者血液射击与随机行走策略
                v = mimicry(theBestVct, Positions, Max_iter, SearchAgents_no, t)
                print(f"进入了伪装策略")
            else:
                if t % 2 == 0:  # 若迭代次数是偶数，执行血液射击策略
                    v = shootBloodstream(theBestVct, Positions[r, :], Max_iter, t)
                    print(f"执行了血液射击策略")
                else:  # 若迭代次数是奇数，执行随机行走策略
                    v = randomWalk(theBestVct, Positions[r, :])
                    print(f"执行了随机行走策略")
            #这一步将最差的代理进行替换计算
            Positions[vMax_idx, :] = Skin_darkening_or_lightening(theBestVct, Positions, SearchAgents_no)
            print(f"最差是第{vMax_idx+1}个代理，执行皮肤变化策略")
            if alphaMelanophore[r] <= 0.3:
                #print(f"第{r+1}个alphaMelanophore值是{alphaMelanophore[r]}，执行替换逃离策略")
                v = remplaceSearchAgent(theBestVct, Positions, SearchAgents_no)

            #v = checkBoundaries(v, lb, ub)
            #以上已经确定好了一个搜索代理
            #接下来计算新的搜索代理的适应度函数
            #print(v)
            #先将得到的新代理二进制化，这个临界值有待商榷
            print(f"第{t}轮迭代中的，第{r + 1}个代理是：{v}")

            #下面代码计算欧氏距离并转换二进制版本
            v_euclideanDistance=euclideanDistance(v,Positions[r,:])
            v_binary=continuous_to_discreate(v_euclideanDistance,judgementCriteria)

            #下面代码利用sigmoid函数转化二进制版本
            # v_binary=continuous_to_discreate(v,0)

            #用得到的二进制进行特征选择，然后训练并计算适应度函数
            train_data_selected = feature_selection(v_binary, train_data)
            test_data_selected = feature_selection(v_binary, test_data)
            # 接下来开始训练，开始测试
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(train_data_selected, train_target)
            y_predict = knn.predict(test_data_selected)
            error_rate = 1 - accuracy_score(test_target, y_predict)
            # 接下来开始计算适应度的值（即目标函数值）
            Fnew =fitness(v_binary, error_rate, a, 1-a)

            #接下来需要对新的搜索代理进行公式修正，以免其数值收敛于0


            if Fnew <= Fitness[r]:
                Positions[r, :] = v
                Fitness[r] = Fnew

            if Fnew <= vMin:
                theBestVct = v
                vMin = Fnew

            print(f"feature_number={train_data_selected.shape[1]}")
            print(f"error_rate={error_rate}")
            print(f"fnew={Fnew}")
            print(f"Fitness={Fitness}")
            print('   ')

        vMax_idx = np.argmax(Fitness)
        alphaMelanophore = alpha_melanophore(Fitness, vMin, Fitness[vMax_idx])
        Convergence_curve[t-1] = vMin

    print(f"Convergence_curve={Convergence_curve}")
    print(f"vMin={vMin}")
    print(f"theBestVct={theBestVct}")

    return vMin, theBestVct, Convergence_curve


if __name__ == "__main__":
    print(f"运行程序来自HLOA_Binary.py")
    ub=np.array([20])
    lb=np.array([-20])
    # pos1=Initialization(10,6,ub,lb)
    # pos2=np.array([[1,1,2],[10,20,30],[5,6,7],[19,20,21]])

    HLOA(10,30,lb,ub,37,10,0.6,'KDDTrain+.csv','KDDTest+.csv')

    # xnext = np.array([1,2,3,4,5,6])
    # xbest=np.array([-14,-8,-2,4,10,16])
    # x_arrary1=[]
    # x_arrary2 =[]
    # x_arrary3 = []
    # x_arrary4 = []
    # x_arrary5 = []
    # x_arrary6 = []
    # x_1 = []
    # x_2 = []
    # x_3 = []
    # x_4 = []
    # x_5 = []
    # x_6 = []
    # for t in range(10):
    #     for i in range(30):
    #         #print(xnext)
    #         #xnext=mimicry(xbest,pos1,15,10,i)
    #         #xnext=shootBloodstream(xbest,xnext,30,i)
    #         xnext=randomWalk(xbest,xnext)
    #         #xnext=Skin_darkening_or_lightening(xbest,pos1,10)
    #         x_arrary1.append(xnext[0])
    #         x_arrary2.append(xnext[1])
    #         x_arrary3.append(xnext[2])
    #         x_arrary4.append(xnext[3])
    #         x_arrary5.append(xnext[4])
    #         x_arrary6.append(xnext[5])
    #         #xnext=remplaceSearchAgent(pos2[1,:],pos2,3)
    #     x_1.append(np.std(x_arrary1))
    #     x_2.append(np.std(x_arrary2))
    #     x_3.append(np.std(x_arrary3))
    #     x_4.append(np.std(x_arrary4))
    #     x_5.append(np.std(x_arrary5))
    #     x_6.append(np.std(x_arrary6))
    # x_1 = np.array(x_1)
    # x_2 = np.array(x_2)
    # x_3 = np.array(x_3)
    # x_4 = np.array(x_4)
    # x_5 = np.array(x_5)
    # x_6 = np.array(x_6)
    # print(f"x1迭代10次的标准差x_1为：{x_1}")
    # print(f"x2迭代10次的标准差x_2为：{x_2}")
    # print(f"x3迭代10次的标准差x_3为：{x_3}")
    # print(f"x4迭代10次的标准差x_4为：{x_4}")
    # print(f"x5迭代10次的标准差x_5为：{x_5}")
    # print(f"x6迭代10次的标准差x_6为：{x_6}")
    # print(f"x_1的平均值为：{np.mean(x_1)}")
    # print(f"x_2的平均值为：{np.mean(x_2)}")
    # print(f"x_3的平均值为：{np.mean(x_3)}")
    # print(f"x_4的平均值为：{np.mean(x_4)}")
    # print(f"x_5的平均值为：{np.mean(x_5)}")
    # print(f"x_6的平均值为：{np.mean(x_6)}")
