import random
import numpy as np
from cal_similarity import haiming_basic,common_substr
from vector import *
import math
import itertools
from predict_AMP import cal_activity
import copy

N=200
T=6
MINL=7
MAXL=48
amino=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
r1=0.45
#存储权重向量
mv = Mean_vector(N-1, 2)
weight=mv.test()
#存储邻居权重
B=[]
for num in range(len(weight)):
    left=num-3
    right=num+3
    if left<0:
        right=right-left
        left=0
    if right>=N:
        left=left-(right-N+1)
        right=N-1
    tmp=[]
    for jj in range(left,right+1):
        if jj!=num:
            tmp.append(jj)
    B.append(tmp)




def create_pop(num):
    global N,MAXL,amino,MINL
    P=[]
    for n in range(num):
        length = random.randint(MINL, MAXL)
        pp = []
        for i in range(length):
            pp.append(amino[random.randint(0,19)])
        P.append(pp)
    return P


def cal_multi(pi,r,P):
    global N
    #r=2
    res=0
    if pi not in P:
        P=P+pi
    for pp in range(len(P)):
        if P[pp]==pi:
            continue
        d=haiming_basic(pi,P[pp])
        if d<r1:
            res=res+(1-d/r)*(1-d/r)
    #res=format(pow(res,0.5),'.3f')
    return float(res)



def Tchebycheff_dist(w, f, z):
    # 计算切比雪夫距离
    return w * abs(f - z)

def cal_qbxf(Z,idx,P,f_activaty):
    max = 0
    multi = cal_multi(P[idx], r1, P)
    flag = 1
    if multi == 0:
        flag = 0
    ri = weight[idx]
    F_X = [f_activaty[idx],multi]
    for i in range(len(F_X)):
        fi = Tchebycheff_dist(ri[i], F_X[i], Z[i])
        if fi > max:
            max = fi
    return max,flag

def cal_sum(Z,idx,P,f_activity):
    f_m=cal_multi(P[idx],r1,P)
    flag=1
    if f_m==0:
        flag=0
    ri=weight[idx]
    return f_activity[idx]*ri[0]+f_m*ri[1],flag

def cal_pbi(Z,idx,P,f_activity):
    multi=cal_multi(P[idx],r1,P)
    flag=1
    if multi==0:
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaa')
        flag=0

    theta=2
    fx=[f_activity[idx],multi]
    d1s=(fx[0]-Z[0])*weight[idx][0]+(fx[1]-Z[1])*weight[idx][1]
    d1x=math.sqrt(weight[idx][0]*weight[idx][0]+weight[idx][1]*weight[idx][1])
    d1=d1s/d1x

    d20=fx[0]-(Z[0]-d1*weight[idx][0]/d1x)
    d21=fx[1]-(Z[1]-d1*weight[idx][1]/d1x)
    d2=math.sqrt(d20*d20+d21*d21)

    return d1+theta*d2,flag


def crossover1(P,index1,index2,cnt):
    #直接选取一个点，进行交叉，单点交叉
    global MAXL, MINL,amino,all_peptide,N
    # l=16
    c=[]
    i=0
    while i<cnt:
        if len(P[index1])==1:
            p1=1
        else:
            p1 = random.randint(1, len(P[index1]) - 1)
        if MINL!=MAXL:
            if len(P[index2])==1:
                p2=0
            else:
                p2 = random.randint(1, len(P[index2]) - 1)
            C = P[index1][0:p1] + P[index2][p2:len(P[index2])]
        else:
            C = P[index1][0:p1] + P[index2][p1:len(P[index2])]
        if len(C)>46 or len(C)<MINL or len(C)>MAXL:
            continue
        if C in c or C in P:
            continue
        c.append(C)
        #print(C)
        #c.append(C2)
        i=i+1
    if len(c)==1:
        return c[0]
    else:
        return c


def local_searchx(P,f_activity):
    # 随机选k个位置，删除或增添每个位置的氨基酸，共产生j种新的解，找多样性为0的解
    px = []
    nl = 10
    ls_pop = []
    multi = []
    j = 5
    new_solution = 0
    valid = []
    for i in range(len(P)):
        cc = cal_multi(P[i], r1, P)
        if f_activity[i] < 0.1 and cc != 0:
            multi.append(cc)
            px.append(i)
    while len(ls_pop) < nl and len(px) != 0:
        random_index=random.randint(0,len(px)-1)
        x = px[random_index]  # px中多样性最差的
        tmp_p=copy.deepcopy(P[x])
        ls_pop.append(x)  # ls_pop存储P中已经了做局部搜索的个体的索引
        per_seq={}
        for i in range(j):
            l = random.choice([i for i in range(len(P[x]))])
            if random.choice([-1,1])==-1:
                per_seq[l]=-1
            else:
                per_seq[l]=random.choice(amino)
        new_solution = new_solution + len(per_seq)
        y_mu = []  # 每种情况的多样性
        for i in list(per_seq.keys()):
            if per_seq[i]!=-1:
                P[x].insert(i,per_seq[i])
            else:
                del P[x][i]
            if P[x] in P[:x]+P[x+1:]:
                new_solution=new_solution-1
                del per_seq[i]
                print('bbbbbbbbbbbbbbbbbbbbbb')
            else:
                y_mu.append(cal_multi(P[x], r1, P))
            P[x]=copy.deepcopy(tmp_p)
        # 选择多样性最好的一种情况
        multi_min = min(y_mu)
        if multi_min!=0 or multi_min >= multi[random_index]:
            del px[random_index]
            del multi[random_index]
        else:
            print(str(multi[random_index])+' -> '+str(multi_min))
            valid.append(len(ls_pop))
            m=list(per_seq.keys())[y_mu.index(min(y_mu))]
            if per_seq[m]==-1:
                del P[x][m]
            else:
                P[x].insert(m,per_seq[m])
            # 更新px
            multi = []
            px = []
            for i in range(len(P)):
                if i in ls_pop:
                    # 如果个体已经做过局部搜索，就不用了
                    continue
                cc = cal_multi(P[i], r1, P)
                if f_activity[i] < 0.1 and cc != 0:
                    multi.append(cc)
                    px.append(i)
    return P, ls_pop, new_solution,valid












