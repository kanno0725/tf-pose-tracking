# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:07:08 2020

@author: kanno
"""
import cv2
import pandas as pd
#import argparse

#from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from tracking_3 import roll,tracking_function

# 読み込む動画のパス
movie_file = 'test.mp4'
output_file = 'test_result3.mp4'

# tf-poseの準備
model = 'cmu'
w, h = model_wh('432x368')
if w == 0 or h == 0:
    e = TfPoseEstimator(get_graph_path(model), target_size=(432, 368))
else:
    e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

# 対象の動画を読み込む
vc = cv2.VideoCapture(movie_file)

# アウトプットの準備
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
fps = int(vc.get(cv2.CAP_PROP_FPS))
size = (
    int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
vw = cv2.VideoWriter(output_file, fourcc, fps,  size)

#args = parser.parse_args()

# get 2d estimation result
dfs = pd.DataFrame(index=[])
columns = ['frame', 'human', 'point', 'x', 'y']

# 動画のフレームが呼び出せる間ループする
frame_no = 0
coordinate = []

while True:
    # 最初のフレームを画像として呼び出す
    ret, img = vc.read()
    if not ret:
       break

    img = cv2.resize(img, size)
    # 画像から骨格推定を行い、人物のデータを取得
    humans = e.inference(img, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
    # 取得した人物データをフレームの画像に合成
    img = TfPoseEstimator.draw_humans(img, humans, imgcopy=False)

    # per human
    xx = 0
    coordinate_frame = []
    for human in humans:
        xx = xx + 1
        # per body_part
        for m in human.body_parts:
            body_part = human.body_parts[m]
            center = [int(body_part.x * w + 0.5), int(body_part.y * h + 0.5)]
            #list->parts_list
            parts_list = [[frame_no, xx, m, center[0],center[1]]]
            df = pd.DataFrame(data=parts_list, columns=columns)
            # add list to dfs
            dfs = pd.concat([dfs, df])
            
            if m == 1:
                coordinate_frame.append(center)
    
    if len(coordinate_frame) == 0:
        coordinate.append([['none']])
    else:
        coordinate.append(coordinate_frame)
    
    #id_listを作成
    if frame_no == 0:
        id_list = list(range(len(coordinate[0])))
        id_max = max(id_list) # len(coordinate[0])-1    
    else:
        # import method id_list
        id_list_new, id_max, id_exist = tracking_function(coordinate, frame_no, id_list, id_max)
       
        if id_exist:
            id_list = id_list_new
            
            # 取得した人物のidをフレーム画像に描画
            for human_no in range(xx):
                img = cv2.putText(img,'id:'+str(id_list[human_no]),(coordinate_frame[human_no][0],coordinate_frame[human_no][1]+5),cv2.FONT_HERSHEY_PLAIN,1,(100, 255, 100), 1, cv2.LINE_AA)
            
    # 合成した画像をフレームとしてアウトプットに追加
    vw.write(img)
    
    frame_no += 1
            
# 書き込み処理
vw.release()

# result output
#dfs.to_csv(args.datas + args.dataname +'_'+ args.movie[-len(args.movie)+8:-4]  +  '_2ddata.csv', encoding="utf-8")
dfs.to_csv('_3ddata.csv', encoding="utf-8")