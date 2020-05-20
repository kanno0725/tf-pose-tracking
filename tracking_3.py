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
def tracking_function(coordinate, frame_no, id_list, id_max):
    fr = frame_no 
    #frame1~
    length = 0
    before_count = 1
    while length < 2:
        h0 = np.array(coordinate[fr-before_count])
        num0 = len(h0) #h0 -> before frame
        length = len(h0[0])
        before_count += 1
    
    h1 = np.array(coordinate[fr])
    num1 = len(h1) #h1 -> current frame
    if len(h1[0]) < 2:
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
                
        for m in range(num1):
            if id_list_new[m] == 0.1:
                id_list_new[m] = id_max + max_count
                max_count += 1
        #debug
        #print(id_list_new)
        
        #id_maxを更新
        if id_max < max(id_list_new):
            id_max = max(id_list_new)
        
    return id_list_new, id_max, id_exist

"""
id_list_new, id_max, id_exist = tracking_function(coordinate, frame_no, id_list, id_max)

if id_exist:
    id_list = id_list_new
    print(id_list) # 処理はif文の中に
    
print("end")
"""
"""
# 関節毎の描画色（とりあえず適当な配色）
colors = [(255,0,85), (255,0,0), (255,85,0), (255,170,0), (255,255,0), (170,255,0), 
          (85,255,0), (0,255,0), (0,255,85), (0,255,170), (0,255,255), (0,170,255), 
          (0,85,255), (0,0,255), (255,0,170), (170,0,255), (255,0,255), (85,0,255)]
pairs = [[1,0],[15,0],[15,17],[14,0],[14,16],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13]]
img = np.full((720, 1280, 3), 255, dtype=np.uint8)

 fr_padded0 = '%04d' % frame
    # jsonのロード
    input_file_name0 = 'new_crop4_1_00000000'+str(fr_padded0)+'_keypoints.json'
    with open(input_file_name0) as f0:
        data0 = json.load(f0)
        d0 = data0['people']
        h0 = []
        for i0 in range(len(d0)):
            kpt0 = np.array(d0[i0]['pose_keypoints_2d']).reshape((25, 3))
            h0.append([kpt0[1][0],kpt0[1][1]])
    
    with open('crop4_1_weld.csv') as f0:
        w0 = []
        for row in csv.reader(f0):
            if row[0] == 'frame'+str(frame):
                w0.append([int(row[2]),int(row[3])])            
    v0 = list()
    v0.extend(h0)
    v0.extend(w0)
    num0 = len(v0)
    v0 = np.array(v0)
    
    if num0 == 0:
        continue
    
    af = 1
    num1 = 0
    while num1 == 0:
        fr_padded1 = '%04d' % (frame+af)     
        input_file_name1 = 'new_crop4_1_00000000'+str(fr_padded1)+'_keypoints.json'
        with open(input_file_name1) as f1:
            data1 = json.load(f1)
            d1 = data1['people']
            h1 = []
            for i1 in range(len(d1)):
                kpt1 = np.array(d1[i1]['pose_keypoints_2d']).reshape((25, 3))
                h1.append([kpt1[1][0],kpt1[1][1]])
                    
        with open('crop4_1_weld.csv') as f1:
            w1 = []
            for row in csv.reader(f1):
                if row[0] == 'frame'+str(frame+af):
                    w1.append([int(row[2]),int(row[3])]) 
        v1 = list()
        v1.extend(h1)
        v1.extend(w1)
        num1 = len(v1)    
        v1 = np.array(v1)
        
        if num1 == 0:
            frame2 = frame + af
            output = open(filename, 'a')
            #output.write('frame'+str(frame2)+','+'/'+',')
            output.write("\n")
            output.close()
            af += 1
    
    #全ての組み合わせの長さを計算
    r_arr = []
    for j in range(num0):
        for k in range(num1):
            r = np.linalg.norm(v0[j] - v1[k])
            r_arr = np.append(r_arr,r)
    r_arr = np.array(r_arr).reshape(num0,num1)
    
    #debug
    #print(r_arr)
    
    ##bestの組み合わせを選択
    x0 = len(r_arr)
    x1 = len(r_arr[0])
    
    t = 0
    #転置
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
    if t == 1:
        innn = []
        for index in range(len(inn)):
            innn.append([inn[index],index])
    else:
        innn = []
        for index in range(len(inn)):
            innn.append([index,inn[index]])
    
    #出力
    id_list_new = [0.1]*num1
    max_count = 1
    #debug
    #print(num0)
    #print(num1)
    #print(innn)
    #tqdm
    for l in range(len(innn)):
        if r_arr[innn[l][0],innn[l][1]] < 150:
            id_list_new[innn[l][1]] = id_list[innn[l][0]]
            
    for m in range(num1):
        if id_list_new[m] == 0.1:
            id_list_new[m] = id_max + max_count
            max_count += 1
    id_list = id_list_new
    #debug
    #print(id_list)
    #print(af)
    id_all.extend(id_list)
    id_max = max(id_all)
    frame3 = frame + af
    output = open(filename, 'a')
    #output.write('frame'+str(frame3)+',')
    h_w = 0
    for x in id_list:
        if h_w == len(h1):
            output.write('/'+',')
        output.write(str(x)+',')
        h_w += 1
    output.write("\n")
    output.close()
    #
    print("end / "+str(frame))
    
    #画像に出力
    img = np.full((720, 1280, 3), 255, dtype=np.uint8)
    i = 0
    for i in range(len(d1)):
        kpt = np.array(d1[i]['pose_keypoints_2d']).reshape((25, 3))
        #関節のつながりをリスト化
        for p in range(len(pairs)):
            if kpt[pairs[p][0],2] != 0 and kpt[pairs[p][1],2] != 0:
                cv2.line(img,tuple(map(int,kpt[pairs[p][0],0:2])),tuple(map(int,kpt[pairs[p][1],0:2])),colors[p], thickness=2, lineType=cv2.LINE_4)
                cv2.putText(img, 'id:'+str(id_list[i]), (int(kpt[1,0]),int(kpt[1,1])+10),cv2.FONT_HERSHEY_PLAIN,1,(255, 0, 0), 1, cv2.LINE_AA)
                
    with open('crop4_1_weld.csv') as f1:
        wn = 0
        for row in csv.reader(f1):
            if row[0] == 'frame'+str(frame+af):
                cv2.circle(img, (int(row[2]),int(row[3])), int(row[4]), (255, 0, 0),3)
                cv2.putText(img, 'id:'+str(id_list[len(d1)+wn]), (int(row[2]),int(row[3])+10),cv2.FONT_HERSHEY_PLAIN,1,(255, 0, 0), 1, cv2.LINE_AA)
                wn += 1
                        
    cv2.imwrite(str(input_file_name1)+'add_weld_01.jpg', img)
"""
