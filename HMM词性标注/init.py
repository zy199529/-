#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Lenovo
# @Date:   2019-06-10 10:49:01
# @Last Modified by:   Lenovo
# @Last Modified time: 2019-06-11 09:57:13
# HMM五个要素：状态数、观测数、A状态转移矩阵、B观测概率矩阵、pi初始状态值
# 中文词性标注
import codecs

# 预处理文本


def ChinesePos():
    w = {}  # 所有的词语
    pos = []  # 所有的词性
    with codecs.open("处理语料.txt", "r", "utf8")as f:
        for line in f.readlines():
            temp = line.strip().split(" ")
            for i in range(1, len(temp)):
                word = temp[i].split("/")
                if len(word) == 2:
                    if word[0] not in w:
                        w[word[0]] = 1  # 存放词语
                    if word[1] not in pos:
                        pos.append(word[1])  # 存放标签
    f.close()
    w = w.keys()
    return w, pos


def init_state(w, pos):
    pi = {}  # 初始状态值矩阵
    A = {}  # 状态转移矩阵
    B = {}  # 观测概率矩阵
    prob_pos = {}  # 每个词性出现的概率
    prob_word = {}  # 每个词语出现的概率
    for i in pos:
        pi[i] = 0
        prob_pos[i] = 0
        A[i] = {}
        B[i] = {}
        for j in pos:
            A[i][j] = 0
        for j in w:
            B[i][j] = 0
    for i in w:
        prob_word[i] = 0
    num = 0
    with codecs.open("处理语料.txt", "r", "utf8") as fr:
        for line in fr.readlines():
            if line == '\n':
                continue
            line = line.strip().split(" ")
            n = len(line)
            num = num+1
            for i in range(1, n):
                word = line[i].split("/")
                pre = line[i-1].split("/")
                prob_word[word[0]] += 1  # 统计词频数
                prob_pos[word[1]] += 1  # 统计词性频数
                if i == 1:
                    pi[word[1]] += 1
                else:
                    A[pre[1]][word[1]] += 1
                B[word[1]][word[0]] += 1
    for i in pos:
        pi[i] = float(pi[i])/num  # 初始状态的概率矩阵
        for j in pos:
            if A[i][j] == 0:
                A[i][j] = 0.5
        for j in w:
            if B[i][j] == 0:
                B[i][j] = 0.5
    for i in pos:
        for j in pos:
            A[i][j] = float(A[i][j]/prob_pos[j])
        for j in w:
            B[i][j] = float(B[i][j]/prob_word[j])
    return A, B, pi, prob_pos, prob_word
# 参数初始化完毕采用动态规划的维特比算法求解最佳路径


def viterbi(A, B, pi, pos, str_test, prob_pos, prob_word):
    # 计算文本长度
    num = len(str_test)
    # 绘制概率转移路径
    dp = [{} for i in range(0, num)]
    # 状态转移路径
    pre = [{} for i in range(0, num)]
    for k in pos:
        for j in range(num):
            dp[j][k] = 0
            pre[j][k] = ''
    # 句子初始化状态概率分布（首个词在词性的概率分布)
    for p in pos:  # 第一个词在观测序列中，得到其观测概率
        if str_test[0] in B[p]:
            dp[0][p] = pi[p]*B[p][str_test[0]]*1000
        else:
            dp[0][p] = pi[p]*0.5*1000  # 若不在其中，则0.5*初始状态概率
    for i in range(0, num):
        for j in pos:  # 假如词语在观测转移矩阵中
            if (str_test[i] in B[j]): # 状态更新
                state = B[j][str_test[i]]*1000
            else:
                state = 0.5*1000
            for k in pos: #第i个词语的状态概率
                if (dp[i][j] < dp[i-1][k]*A[k][j]*state):
                    dp[i][j] = dp[i-1][k]*A[k][j]*state
                    pre[i][j] = k
    restate = {}
    max_state = ""
    # 首先找到最后输出的最大观测值的状态
    for j in pos:
        if max_state == "" or dp[num-1][j] > dp[num-1][max_state]:
            max_state = j
    i = num-1
    while i >= 0:
        restate[i] = max_state
        max_state = pre[i][max_state]
        i = i-1
    for i in range(0, num):
        print(str_test[i]+"\\"+restate[i])


str_test = [u"北京", u"举行", u"新年", u"音乐会"]
w, pos = ChinesePos()
A, B, pi, prob_pos, prob_word = init_state(w, pos)
viterbi(A, B, pi, pos, str_test, prob_pos, prob_word)
