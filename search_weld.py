# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:36:37 2020

@author: kanno
"""
import cv2

def search_weld(frame):
    # 画像読み込み
    gray_img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # 2値化
    ret, binary = cv2.threshold(gray_img, 225, 255, cv2.THRESH_BINARY)
    #test 245->255
    
    # 輪郭抽出
    _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    weld_space = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 500 < area :
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            center = (int(x),int(y))
            radius = int(radius)
            weld_space.append(list([center[0],center[1]]))
   
    return weld_space
        
