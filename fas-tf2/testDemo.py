# @Auther : wuwuwu 
# @Time : 2020/8/20 
# @File : testDemo.py
# @Description :

import tensorflow as tf
from generator_network import get_FaceMapNet
import cv2 as cv
import numpy as np
import sys

sys.setrecursionlimit(100000000)  # 例如这里设置为一百万

model = get_FaceMapNet()
# model.compile(optimizer=optimizers.Adam(lr=1e-4),
#               loss=CDLloss(),
#               metrics=[fasAccuracy()])
# model.load_weights('model_save/checkpoint')
model.load_weights('model1/checkpoint')


# def face_predict(img):
#     img = cv.resize(img, (256, 256))
#     img = np.expand_dims(img, axis=0)
#     depth = np.ones((1, 32, 32, 1), dtype=np.float32)
#     img = img / 255.0
#     x_test = {'images': img, 'depths': depth}
#     output = model(x_test)
#     depth = output[:, :, :, 0]
#     depth = tf.expand_dims(depth, axis=-1)
#     logit = tf.reduce_mean(depth, axis=1)
#     logit = tf.reduce_mean(logit, axis=1)
#     logit = tf.reduce_mean(logit, axis=1, keepdims=True)
#     logit = tf.concat([logit, 1 - logit], axis=1)
#
#     pred = tf.argmax(logit, axis=-1, output_type=tf.int32).numpy()[0]
#     if pred == 0:
#         return 'False'
#     elif pred == 1:
#         return 'True'
#
#     return None


def face_predict(img):
    img = cv.resize(img, (256, 256))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    output, logit = model(img)

    pred = tf.argmax(logit, axis=-1, output_type=tf.int32).numpy()[0]
    prob = logit.numpy()[0][pred]
    if pred == 0:
        return 'False : ' + str(prob)
    elif pred == 1:
        return 'True : ' + str(prob)

    return None


# 框出人脸
def face_test(img):
    # 联级分类器，haarcascade_frontalface_default.xml为储存了人脸特征的xml文件
    faces = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    # 找出人脸的位置
    face = faces.detectMultiScale(img, 1.1, 5)
    # 坐标点
    for x, y, w, h in face:
        face_img = img[y : y + h, x : x + w]
        text = face_predict(face_img)
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)  # 画出框
        cv.putText(img, text, (x, y), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
        # cv.imshow("face1", face_img)
    cv.imshow('face', img)  # 显示


if __name__ == '__main__':
    video = cv.VideoCapture(0)  # 打开摄像头
    while True:
        ret, img = video.read()  # 读取图片
        if ret is False: break
        face_test(img)  # 调用函数
        # 保持画面的连续，按esc键退出
        if cv.waitKey(1) & 0xFF == 27:
            break

    video.release()
    cv.destroyAllWindows()