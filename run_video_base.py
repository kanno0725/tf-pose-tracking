# -*- coding: utf-8 -*-
"""
Created on Sun May 17 15:07:08 2020

@author: kanno
"""
import cv2
import pandas as pd
import argparse

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

# 読み込む動画のパス
movie_file = 'test.mp4'
output_file = 'test_result.mp4'

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
columns = ['flame', 'human', 'point', 'x', 'y']

# 動画のフレームが呼び出せる間ループする
frame_no = 0
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
    # 合成した画像をフレームとしてアウトプットに追加
    vw.write(img)
    
    # per human
    xx = 0
    for human in humans:
        xx = xx + 1

        # per body_part
        for m in human.body_parts:
            body_part = human.body_parts[m]
            center = (int(body_part.x * w + 0.5), int(body_part.y * h + 0.5))
            list = [[frame_no, xx, m, center[0],center[1]]]
            df = pd.DataFrame(data=list, columns=columns)

            # add list to dfs
            dfs = pd.concat([dfs, df])
    
    # import method id_list
    
    frame_no += 1
            
# 書き込み処理
vw.release()

# result output
#dfs.to_csv(args.datas + args.dataname +'_'+ args.movie[-len(args.movie)+8:-4]  +  '_2ddata.csv', encoding="utf-8")
dfs.to_csv(movie_file+'_2ddata.csv', encoding="utf-8")
