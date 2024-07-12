import numpy as np

"""
对该算法在特征选择上的应用，需要对算法进行一定的修改，使其能够适应特征选择的问题
"""


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
    :param X:搜索代理本次的位置
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
                                                      - ((-1) ** getBinary()) * c2 * np.cos(X[r3, :]) - np.sin(
                X[r4, :]))
    return Xnext


def Skin_darkening_or_lightening(Xbest, X, SearchAgents):
    """
    实现角蜥蜴皮肤的变量和变暗策略
    :param Xbest:找到的最好代理
    :param X:目前的种群矩阵或者说搜索代理矩阵
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
    :param X:  搜索代理本次的位置
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
    :param X:当前搜索代理的位置
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


def helper_fun(x, aSH_i, cSH_i):
    return -((x - aSH_i) @ (x - aSH_i) + cSH_i) ** -1


def get_function(F):
    """
    根据F的值，返回对应下界，上界，维度和目标函数
    :param F:需要计算的目标函数的编号
    :return:与F对应的下界，上界，维度和目标函数
    """
    aSH = np.array([[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6],
                    [3, 7, 3, 7], [2, 9, 2, 9], [5, 5, 3, 3], [8, 1, 8, 1],
                    [6, 2, 6, 2], [7, 3.6, 7, 3.6]])
    cSH = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])
    switcher = {
        1: lambda x: sum(x ** 2),  # F1
        2: lambda x: sum(abs(x)) + np.prod(abs(x)),  # F2
        3: lambda x: sum([sum(x[:i + 1]) ** 2 for i in range(len(x))]),  # F3
        4: lambda x: max(abs(x)),  # F4
        5: lambda x: sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(len(x) - 1)),  # F5
        6: lambda x: sum(abs(x + 0.5) ** 2),  # F6
        7: lambda x: sum([(i + 1) * (x[i] ** 4) for i in range(len(x))]) + np.random.randn(),  # F7
        8: lambda x: sum(-x * np.sin(np.sqrt(abs(x)))),  # F8
        23: lambda x: sum(helper_fun(x, aSH_i, cSH_i) for aSH_i, cSH_i in zip(aSH, cSH))
    }

    # 从switcher中获取F对应的函数，如果没有就返回一个字符串
    fobj = switcher.get(F, lambda x: "Invalid function called")

    boundaries_and_dims = {
        1: (-100, 100, 30),
        2: (-10, 10, 30),
        3: (-100, 100, 30),
        23: (0, 10, 4),
    }

    # 从boundaries_and_dims中获取F对应的下界，上界和维度，如果没有就返回标识字符串与对应的fobj内容
    result = boundaries_and_dims.get(F)
    if result is None:
        return "Invalid function identifier", "Invalid range", "Invalid dimension", fobj
    lb, ub, dim = result
    lb = np.array([lb])
    ub = np.array([ub])
    return lb, ub, dim, fobj

def feature_selection():
    """
    需要实现对数据集进行角蜥蜴优化算法的特征选择
    对参数的使用以及返回值，尽可能和前面的函数保持一致，所用测试模型就用sklearn中的基础模型
    （以特征值的个数作为dim，lb和ub就是0和1，公式是针对连续数据的，离散数据需要进行修改，现在的
    工作重心就是找到合适的公式，以角蜥蜴算法的思路对离散数据进行处理）
    :return:
    """
    pass

def HLOA(SearchAgents_no, Max_iter, lb, ub, dim, fobj):
    """
    实现角蜥蜴优化算法
    :param SearchAgents_no:
    :param Max_iter:
    :param lb:
    :param ub:
    :param dim:
    :param fobj:
    :return:
    """

    # 初始化搜索代理的位置，行数是搜索代理的数量SearchAgents_no，列数是搜索空间的维度dim
    Positions = Initialization(SearchAgents_no, dim, ub, lb)

    # 每个搜索代理的适应度值（即目标函数值）
    Fitness = np.array([fobj(Positions[i, :]) for i in range(SearchAgents_no)])

    vMin_idx = np.argmin(Fitness)  # 返回最小值的索引
    vMin = Fitness[vMin_idx]
    theBestVct = Positions[vMin_idx, :]  # 将最小值定为最好的搜索代理

    vMax_idx = np.argmax(Fitness)  # 返回最大值的索引，且是最差的搜索代理
    Convergence_curve = np.zeros(Max_iter)
    Convergence_curve[0] = vMin  # 收敛曲线，记录每一轮的最佳值，用于分析
    # print(f'Fitness: {Fitness}')
    alphaMelanophore = alpha_melanophore(Fitness, vMin, Fitness[vMax_idx])

    for t in range(1, Max_iter):
        for r in range(SearchAgents_no):
            if np.random.rand() > 0.5:  # 50%的概率，执行伪装策略或者血液射击与随机行走策略
                v = mimicry(theBestVct, Positions, Max_iter, SearchAgents_no, t)
            else:
                if t % 2 == 0:  # 若迭代次数是偶数，执行血液射击策略
                    v = shootBloodstream(theBestVct, Positions[r, :], Max_iter, t)
                else:  # 若迭代次数是奇数，执行随机行走策略
                    v = randomWalk(theBestVct, Positions[r, :])

            Positions[vMax_idx, :] = Skin_darkening_or_lightening(theBestVct, Positions, SearchAgents_no)

            if alphaMelanophore[r] <= 0.3:
                v = remplaceSearchAgent(theBestVct, Positions, SearchAgents_no)

            v = checkBoundaries(v, lb, ub)

            Fnew = fobj(v)
            # print(f'Fnew: {Fnew}')
            # print(f'Fitness[r]: {Fitness[r]}')
            if Fnew <= Fitness[r]:
                Positions[r, :] = v
                Fitness[r] = Fnew

            if Fnew <= vMin:
                theBestVct = v
                vMin = Fnew

        vMax_idx = np.argmax(Fitness)
        alphaMelanophore = alpha_melanophore(Fitness, vMin, Fitness[vMax_idx])
        Convergence_curve[t] = vMin

    return vMin, theBestVct, Convergence_curve


def operateHLOA():
    ## number of executions
    i = 30
    a = np.zeros(i)
    SearchAgents_no = 30
    Function_name = 1
    Max_iteration = 200

    lb, ub, dim, fobj = get_function(Function_name)

    v = np.zeros((i, dim))
    cc = np.zeros((i, Max_iteration))  # convergence curve

    for m in range(i):
        print(f'Run: {m + 1}')
        vMin, theBestVct, Convergence_curve = HLOA(SearchAgents_no, Max_iteration, lb, ub, dim, fobj)
        a[m] = vMin  # the best fitness for the m-Run
        v[m, :] = theBestVct  # the best vector (variables) for the m-Run
        cc[m, :] = Convergence_curve  # convergence curve fot the m-Run

    vMin_idx = np.argmin(a)
    theBestVct = v[vMin_idx, :]
    ConvergenceC = cc[vMin_idx, :]

    print(f'The best solution obtained by HLOA is: {theBestVct}')
    print(f'The best fitness (min f(x)) found by HLOA is: {vMin}')
    print(f'# runs: {i}')
    print(f'Mean:  {np.mean(a)}')
    print(f'Std.Dev:  {np.std(a)}')


if __name__ == "__main__":
    # search_agents_no = 10  # Number of search agents
    # dim = 3  # Dimensionality of the search space
    # ub = np.array([10.0])  # Upper bounds for each dimension
    # lb = np.array([0.0, 0.0, 0.0])  # Lower bounds for each dimension
    #
    # # Call the initialization function
    # positions = Initialization(search_agents_no, dim, ub, lb)
    # print(positions)
    operateHLOA()
