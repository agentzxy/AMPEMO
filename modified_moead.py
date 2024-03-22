from moead_basic_function import *
import random
import numpy as np
import time
import os
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
# 初始化种群
import copy

for cnt in range(20):

    P = create_pop(N)
    f_activity = cal_activity(P)
    Z = [0, 0]
    t = 0
    while t <= 40000:
        # t1=time.time()
        descendant = []  # 存储生成的子代
        cross_index = []  # 存储进行交叉变异的个体索引
        ls_pop = []  # 存储局部搜索搜到的个体，用于后面计算活性
        for i in range(N):
            k, l = random.sample(B[i], 2)
            if random.random() < 1:
                Y = crossover1(P, k, l, 1)
                cross_index.append(i)
                Y[random.randint(0, len(Y) - 1)] = amino[random.randint(0, 19)]
                descendant.append(Y)
        # 计算生成的子代活性值
        if descendant != []:
            descendant_activity = cal_activity(descendant)

        for i in cross_index:
            # 计算原来切比雪夫距离
            # qbxf_x,flag_x = cal_qbxf(Z, i, P, f_activity)
            # qbxf_x,flag_x=cal_sum(i, P, f_activity)
            pbi_p, flag_p = cal_pbi(Z, i, P, f_activity)
            # 将生成的y替换原有的p[i]
            tmp1 = P[i]
            tmp2 = f_activity[i]
            P[i] = descendant[i]
            f_activity[i] = descendant_activity[i]
            # qbxf_y,flag_y = cal_qbxf(Z, i, P, f_activity)
            # qbxf_y,flag_y=cal_sum(i, P, f_activity)
            pbi_c, flag_c = cal_pbi(Z, i, P, f_activity)

            if pbi_p < pbi_c:
                P[i] = tmp1
                f_activity[i] = tmp2
            if tmp2 < 0.1 and flag_p == 0:
                if tmp1 not in P:
                    P.append(tmp1)
                    f_activity.append(tmp2)
            if descendant_activity[i] < 0.1 and flag_c == 0:
                if descendant[i] not in P:
                    P.append(descendant[i])
                    f_activity.append(descendant_activity[i])

        eda_Y = []
        eda_activity = []
        if len(P) > N:
            E = copy.deepcopy(P[N:])
            E_ac = copy.deepcopy(f_activity[N:])
            while True:
                multi = []
                for i in E:
                    multi.append(cal_multi(i, r1, P))
                if multi != [] and max(multi) != 0:
                    eda_Y.append(E[multi.index(max(multi))])
                    eda_activity.append(E_ac[multi.index(max(multi))])
                    print(len(P))
                    print(len(f_activity))
                    del f_activity[P.index(eda_Y[-1])]
                    P.remove(eda_Y[-1])
                    del E[multi.index(max(multi))]
                    del E_ac[multi.index(max(multi))]
                else:
                    break

        eda_multi = []
        for i in eda_Y:
            eda_multi.append(cal_multi(i, r1, P))

        p_index = []  # 存储p中要与eda子代比较的个体的索引
        for i in range(len(eda_Y)):
            similarity = []
            for j in range(len(weight)):
                similarity.append(cosine_similarity([[eda_activity[i], eda_multi[i]]], [weight[j]]))
            p_index.append(similarity.index(max(similarity)))

        for i in range(len(eda_Y)):
            # 计算原来切比雪夫距离
            # qbxf_x,flag_x = cal_qbxf(Z, p_index[i], P, f_activity)
            # qbxf_x,flag_x=cal_sum(p_index[i], P, f_activity)
            pbi_p, flag_p = cal_pbi(Z, p_index[i], P, f_activity)
            # 将生成的y替换原有的p[i]
            tmp1 = P[p_index[i]]
            tmp2 = f_activity[p_index[i]]
            P[p_index[i]] = eda_Y[i]
            f_activity[p_index[i]] = eda_activity[i]
            # qbxf_y,flag_y = cal_qbxf(Z, p_index[i], P, f_activity)
            # qbxf_y,flag_y=cal_sum(p_index[i], P, f_activity)
            pbi_c, flag_c = cal_pbi(Z, p_index[i], P, f_activity)

            if pbi_p < pbi_c:
                P[p_index[i]] = tmp1
                f_activity[p_index[i]] = tmp2
            if tmp2 < 0.1 and flag_p == 0:
                if tmp1 not in P:
                    P.append(tmp1)
                    f_activity.append(tmp2)
            if eda_activity[i] < 0.1 and flag_c == 0:
                if eda_Y[i] not in P:
                    P.append(eda_Y[i])
                    f_activity.append(eda_activity[i])

        if t > 10000:
            P, ls_pop1, new_solution, valid = local_search(P, f_activity)
            # 更新多样性个体的活性
            pls = []
            for i in ls_pop1:
                pls.append(P[i])
            pls_ac = cal_activity(pls)
            for i in range(len(ls_pop1)):
                f_activity[ls_pop1[i]] = pls_ac[i]
            t = t + 200 + new_solution
        else:
            valid = []
            t = t + 200

        filename = './res/pbi_random_cm_ls_50l_' + str(cnt) + '.txt'
        f = open(filename, 'a')
        f.write(str(t) + '\n')
        f.write(str(P) + '\n')
        f.write(str(f_activity) + '\n')
        f.write(str(valid) + '\n')
        f.close()

        print(t)
        print(len(P))
        print('P:' + str(P))
        print(f_activity)
        print(len(valid))
        del filename, f, descendant, eda_activity, eda_Y, eda_multi

    del f_activity, P

