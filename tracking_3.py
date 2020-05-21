# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:01:19 2020

@author: kanno
output
1: 2020/05/13
"""
import json
import numpy as np
import cv2
import csv
#from tqdm import tqdm
import itertools
import math
import sys

#test
"""
frame_no = 1
coordinate =[[[2,3]],[[4,5],[11,9]],[['none']],[[2,3]],[[5,6],[4,5],[7,8],[2,2]]] #2->[4,9],[6,9]
id_list = list(range(len(coordinate[frame_no-1])))
id_max = 0
"""

#ロールを定義
def roll(arr,nt):
    for i in range(nt):
        arr_r = np.roll(arr,1)
        arr = arr_r
    result = arr
    return result

#追跡機能を定義
def tracking_function(coordinate, frame_no, id_list, id_max,human_num):
    fr = frame_no 
    #frame1~
    data = 'none'
    before_count = 1
    while data == 'none':
        h0 = np.array(coordinate[fr-before_count])
        num0 = len(h0) #h0 -> before frame
        data = h0[0][0]
        before_count += 1
    
    h1 = np.array(coordinate[fr])
    num1 = len(h1) #h1 -> current frame
    if h1[0][0] == 'none':
        id_exist = False
        id_list_new = []
        id_max = id_max
        
    else:
        id_exist = True
        
        #全ての組み合わせの長さを計算
        r_arr = []
        for j in range(num0):
            for k in range(num1):
                r = np.linalg.norm(h0[j] - h1[k])
                r_arr = np.append(r_arr,r)
        r_arr = np.array(r_arr).reshape(num0,num1)
        
        ##bestの組み合わせを選択
        x0 = len(r_arr)
        x1 = len(r_arr[0])
        
        #転置
        t = 0
        if x0 > x1:
            r_all = r_arr.transpose()
            t = 1
            z0 = len(r_all)
            z1 = len(r_all[0])
        else:
            r_all = r_arr
            z0 = len(r_all)
            z1 = len(r_all[0])
        
        #行数と項数を計算
        line = int(math.factorial(z1-1))
        term = z0
        
        seq = range(z1)
        sll = []
        sc = list(itertools.permutations(seq))
        for l in range(line*z1):
            scl = list(sc[l])
            sll.append(scl)
        
        aa = []
        for i in range(line):
            a = [0]*z1
            for j in range(term):
                a += roll(r_all[j],sll[i][j])
            aa.extend(a)  
        ind = [(np.argmin(aa)//z1),(np.argmin(aa)%z1)]
        
        #インデックス用の配列を作成
        test1 = []
        for k in range(z0):
            test1.append(list(range(z1)))
        test1 = np.array(test1)
        
        aaa1 = []
        a1 = []
        for il in range(line):
            aa1 = []
            for j in range(term):
                a1 = roll(test1[j],sll[il][j])
                if type(a1) != list:
                    a1 = a1.tolist()
                aa1.append(a1)
            aa1t = [list(x) for x in zip(*aa1)]
            aaa1.append(aa1t)
        inn = aaa1[ind[0]][ind[1]]
        
        #転置
        innn = []
        if t == 1:
            for index in range(len(inn)):
                innn.append([inn[index],index])
        else:
            for index in range(len(inn)):
                innn.append([index,inn[index]])
        
        #print(innn)
        
        #出力
        id_list_new = [0.1]*num1
        max_count = 1
        #id_max = max(id_list)
        for l in range(len(innn)):
            if r_arr[innn[l][0],innn[l][1]] < 150:
                id_list_new[innn[l][1]] = id_list[innn[l][0]]
                
        for m in range(human_num): # num1 -> human_num
            if id_list_new[m] == 0.1:
                id_list_new[m] = id_max + max_count
                max_count += 1
        
        #while 0.1 in id_list_new:id_list_new.remove(0.1)
        #debug
        #print(id_list_new)
        
        #id_maxを更新
        if id_max < max(id_list_new):
            id_max = max(id_list_new)
        
    return id_list_new, id_max, id_exist

