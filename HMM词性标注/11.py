#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: zy19950209
# @Date:   2019-06-10 15:45:33
# @Last Modified by:   zy19950209
# @Last Modified time: 2019-06-10 15:46:41
import codecs
def ChinesePOS():
    #转移概率Aij,t时刻由状态i变为状态J的频率
    #观测概率Bj(k)，由状态J观测为K的概率
    #PAI i 初始状态q出现的频率
    #先验概率矩阵
    pi = {}
    a = {}
    b = {}
#所有的词语
    ww = {}
#所有的词性
    pos = []
 
 
    #每个词性出现的频率
 
 
    frep = {}
    #每个词出现的频率
    frew = {}
    fin = codecs.open("处理语料.txt","r","utf8")
 
    for line in fin.readlines():
        temp = line.strip().split(" ")
        for i in range(1,len(temp)):
            word = temp[i].split("/")
            if len(word) == 2:
                if word[0] not in ww:
                    ww[word[0]] = 1
 
                if word[1] not in pos:
                    pos.append(word[1])
    fin.close()
    ww = ww.keys()
 
 
    for i in pos:
        #初始化相关参数
        pi[i] = 0
        frep[i] = 0
        a[i] = {}
        b[i] = {}
 
        for j in pos:
            a[i][j] = 0
        for j in ww:
            b[i][j] = 0
 
    for w in ww:
        frew[w] = 0
    line_num = 0
    #计算概率矩阵
    fin= codecs.open("处理语料.txt","r","utf8")
    for line in fin.readlines():
        if line == "\n":
            continue
        tmp = line.strip().split(" ")
        n = len(tmp)
        line_num += 1
 
        for i in range(1,n):
 
            word = tmp[i].split("/")
            pre = tmp[i-1].split("/")
            #计算词性频率和词频率
            frew[word[0]] += 1
            frep[word[1]] += 1
            if i ==1:
                pi[word[1]] += 1
            else :
                a[pre[1]][word[1]] += 1
 
            b[word[1]][word[0]] += 1
 
    for i in pos:
        #计算各个词性的初始概率
        pi[i] = float(pi[i])/line_num
 
        for j in pos:
            if a[i][j] == 0:
                a[i][j] = 0.5
 
        for j in ww:
            if b[i][j] == 0:
                b[i][j] = 0.5
    for i in pos:
 
        for j in pos:
            #求状态i的转移概率分布
            a[i][j] = float(a[i][j])/(frep[i])
 
        for j in ww:
            #求词j的发射概率分布
            b[i][j] = float(b[i][j])/(frew[j])
    return a,b,pi,pos,frew,frep
 
 
    print("game over")
def viterbi(a,b,pi,str_token,pos,frew,frep):
    # dp = {}
    #计算文本长度
    num = len(str_token)
    #绘制概率转移路径
    dp = [{} for i in range(0,num)]
    #状态转移路径
    pre = [{} for i in range(0,num)]
    for k in pos:
        for j in range(num):
            dp[j][k] = 0
            pre[j][k] = ''
#句子初始化状态概率分布（首个词在所有词性的概率分布）
    for p in pos:
 
 
        if b[p].has_key(str_token[0]):
 
            dp[0][p] = pi[p]*b[p][str_token[0]]* 1000
        else:
            dp[0][p] = pi[p]*0.5*1000
 
 
    for i in range(0,num):
        for j in pos:
            if (b[j].has_key(str_token[i])):
                sep = b[j][str_token[i]] * 1000
            else:
                #计算发射概率,这个词不存在，应该置0.5/frew[str_token[i]]，这里默认为1
                sep = 0.5 * 1000
 
            for k in pos:
                #计算本step i 的状态是j的最佳概率和step i-1的最佳状态k(计算结果为step i 所有可能状态的最佳概率与其对应step i-1的最优状态)
                #
                if (dp[i][j]<dp[i-1][k]*a[k][j]*sep):
 
                    dp[i][j] = dp[i-1][k]*a[k][j]*sep
                    #各个step最优状态转移路径
                    pre[i][j] = k
 
 
    resp = {}
    #
    max_state = ""
    #首先找到最后输出的最大观测值的状态设置为max_state
    for j in pos:
        if max_state=="" or dp[num-1][j] > dp[num-1][max_state]:
            max_state = j
    # print
 
    i = num -1
#根据最大观测值max_state和前面求的pre找到概率最大的一条。
    while i>=0:
        resp[i] = max_state
        max_state = pre[i][max_state]
        i -= 1
    for i in range(0,num):
        print(str_token[i] +"\\" +resp[i].encode("utf8"))
if __name__ == "__main__":
    a,b,pi,pos,frew,frep = ChinesePOS()
    #北京/ns 举行/v 新年/t 音乐会/n
    str_token = [u"北京",u"举行",u"新年",u"音乐会"]
    viterbi(a, b, pi, str_token, pos, frew,frep)