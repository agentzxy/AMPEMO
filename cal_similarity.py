import random
import copy
import numpy as np
import scipy.misc
#from keras.models import load_model
#from keras import Sequential
#import keras
import time
import gc
import math
import Levenshtein


def str_inx(word_, string_):
    return [i for i in range(len(string_)) if string_[i] == word_]


def ab_max_inx(s_a, s_b):
    i, len_a, len_b = 0, len(s_a), len(s_b)
    while len_a > i and len_b > i and s_a[i] == s_b[i]:
        i += 1
    return i

def common_substr(s_a, s_b):
    """
    两个字符串的所有公共子串，包含长度为1的
    :param s_a:
    :param s_b:
    :return:
    """
    res = []
    if s_a:
        a0_inx_in_b = str_inx(s_a[0], s_b)
        if a0_inx_in_b:
            b_end_inx, a_end_inx = -1, 0
            for inx in a0_inx_in_b:
                if b_end_inx > inx:
                    continue
                this_inx = ab_max_inx(s_a, s_b[inx:])
                a_end_inx = max(a_end_inx, this_inx)
                res.append(s_a[:this_inx])
                b_end_inx = this_inx + inx
            res += common_substr(s_a[a_end_inx:], s_b)
        else:
            res += common_substr(s_a[1:], s_b)
    #print(res)
    return res
#print(common_substr('ACDFFFFACDA','CDAFFEA'))
def haiming_basic(s1,s2):
    #最长公共子序列和相似的个数
    # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
    #结果越大越相似
    if len(s1)>len(s2):
        #s1存储短肽，s2存储长肽
        tmp=s1
        s1=s2
        s2=tmp

    ans1 = 0
    res = common_substr(s1, s2)

    for i in range(len(res)):
        if len(res[i])==1:
            res[i]=-1
    while -1 in res:
        res.remove(-1)

    tmp_res=[] #存储没有重复的res
    for i in res:
        ss=''
        for j in i:
            ss=ss+j
        tmp_res.append(ss)
    tmp_res=list(set(tmp_res))
    tmp_res.sort()

    tmp_s2=''
    for i in s2:
        tmp_s2=tmp_s2+i
    for i in tmp_res:
        while i in tmp_s2:
            index=tmp_s2.find(i)
            tmp_s2=tmp_s2[:index]+tmp_s2[index+len(i):]
            ans1=ans1+len(i)

    tmp_s1 = ''
    for i in s1:
        tmp_s1 = tmp_s1 + i
    for i in tmp_res:
        while i in tmp_s1:
            index = tmp_s1.find(i)
            tmp_s1 = tmp_s1[:index] + tmp_s1[index + len(i):]
            ans1 = ans1 + len(i)
    ans1=ans1/(len(s1)+len(s2))


    ans2 = 0
    for i in s1:
        if i in s2:
            ans2=ans2+1
    ans2=ans2/(min(len(s1),len(s2)))

    ans=2-ans1-ans2
    return ans/2




def valley_function(s1,s2):
    s=''
    for i in s1:
        s=s+i
    s1=s
    s=''
    for i in s2:
        s=s+i
    s2=s
    dis=Levenshtein.distance(s1,s2)
    dis=(dis-abs(len(s1)-len(s2)))/(max(len(s1),len(s2))-abs(len(s1)-len(s2)))
    return dis