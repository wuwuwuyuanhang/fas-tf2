# @Auther : wuwuwu 
# @Time : 2020/8/11 
# @File : generator_network.py
# @Description :

import numpy as np
import cv2 as cv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, optimizers, metrics, Sequential
import glob
import os
import random

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class residual_gradient_conv(layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        self.units = units  # 输出通道数
        self.activation = layers.Activation(activation)
        super(residual_gradient_conv, self).__init__(**kwargs)
        pass

    def build(self, input_shape):
        # sobel x 方向算子，不需要训练，tf.constant 常数表示
        sobel_plane_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_plane_x = np.expand_dims(sobel_plane_x, axis=-1)
        sobel_plane_x = np.repeat(sobel_plane_x, input_shape[-1], axis=-1)
        sobel_plane_x = np.expand_dims(sobel_plane_x, axis=-1)
        sobel_plane_x = np.repeat(sobel_plane_x, self.units, axis=-1)
        self.sobel_kernel_x = tf.constant(sobel_plane_x, dtype=tf.float32)

        # sobel y 方向算子，不需要训练，tf.constant 常数表示
        sobel_plane_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_plane_y = np.expand_dims(sobel_plane_y, axis=-1)
        sobel_plane_y = np.repeat(sobel_plane_y, input_shape[-1], axis=-1)
        sobel_plane_y = np.expand_dims(sobel_plane_y, axis=-1)
        sobel_plane_y = np.repeat(sobel_plane_y, self.units, axis=-1)
        self.sobel_kernel_y = tf.constant(sobel_plane_y, dtype=tf.float32)

        # 卷积参数，需要训练
        self.conv = self.add_weight(name='kernel', shape=(3, 3, input_shape[-1], self.units), dtype=tf.float32)
        super(residual_gradient_conv, self).build(input_shape)
        pass

    def call(self, inputs, **kwargs):
        # 卷积结果
        conv = tf.nn.conv2d(inputs, self.conv, strides=1, padding=[[0, 0], [1, 1], [1, 1], [0, 0]])

        # sobel 空间梯度
        gradient_x = tf.nn.conv2d(inputs, self.sobel_kernel_x, strides=1, padding=[[0, 0], [1, 1], [1, 1], [0, 0]])
        gradient_y = tf.nn.conv2d(inputs, self.sobel_kernel_y, strides=1, padding=[[0, 0], [1, 1], [1, 1], [0, 0]])
        gradient_gabor = tf.sqrt(tf.pow(gradient_x, 2) + tf.pow(gradient_y, 2) + 1e-8)

        # 批量归一化
        gradient_gabor = tf.nn.batch_normalization(gradient_gabor, mean=0.0, variance=1.0, offset=None, scale=None,
                                                   variance_epsilon=1e-8)
        gradient_gabor = gradient_gabor / tf.reduce_max(gradient_gabor)
        # print("conv", conv)
        # print("gradient_gabor", gradient_gabor)

        # RSGB 模块相加
        net = conv + gradient_gabor
        net = tf.nn.batch_normalization(net, mean=0.0, variance=1.0, offset=None, scale=None, variance_epsilon=1e-8)
        net = net / tf.reduce_max(net)
        net = self.activation(net)
        return net


def get_FaceMapNet():
    multipler = 1
    inputs = keras.Input(shape=(256, 256, 3))
    output = residual_gradient_conv(units=64, activation='relu', input_shape=(256, 256, 3))(inputs)
    output = residual_gradient_conv(units=64 * multipler, activation='relu')(output)
    output = residual_gradient_conv(units=96 * multipler, activation='relu')(output)
    output = residual_gradient_conv(units=64 * multipler, activation='relu')(output)
    pooling1 = layers.MaxPooling2D((2, 2), strides=2)(output)

    output = residual_gradient_conv(units=64 * multipler, activation='relu')(pooling1)
    output = residual_gradient_conv(units=96 * multipler, activation='relu')(output)
    output = residual_gradient_conv(units=64 * multipler, activation='relu')(output)
    pooling2 = layers.MaxPooling2D((2, 2), strides=2)(output)

    output = residual_gradient_conv(units=64 * multipler, activation='relu')(pooling2)
    output = residual_gradient_conv(units=96 * multipler, activation='relu')(output)
    output = residual_gradient_conv(units=64 * multipler, activation='relu')(output)
    pooling3 = layers.MaxPooling2D((2, 2), strides=2)(output)

    pooling1 = tf.image.resize(pooling1, size=(32, 32))
    pooling2 = tf.image.resize(pooling2, size=(32, 32))
    output = tf.concat([pooling1, pooling2, pooling3], axis=-1)

    output = residual_gradient_conv(units=64 * multipler, activation='relu')(output)
    output = residual_gradient_conv(units=32 * multipler, activation='relu')(output)
    output = layers.Conv2D(1, (3, 3), strides=1, padding='same', activation='relu', name='depths_output')(output)

    logit = layers.Flatten()(output)
    logit = layers.Dense(128, activation='relu')(logit)
    logit = layers.Dense(2, activation='softmax', name='logits_output')(logit)

    net = keras.Model(inputs=inputs, outputs=[output, logit], name='face_map_net')
    return net


def contrast_depth_conv(inputs):
    # 对比卷积核
    kernel_filter_list = [
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]], [[0, 1, 0], [0, -1, 0], [0, 0, 0]], [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [1, -1, 0], [0, 0, 0]], [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, -1, 0], [1, 0, 0]], [[0, 0, 0], [0, -1, 0], [0, 1, 0]], [[0, 0, 0], [0, -1, 0], [0, 0, 1]]
    ]
    kernel_filter = np.array(kernel_filter_list, np.float32)
    kernel_filter = np.expand_dims(kernel_filter, axis=-1)
    kernel_filter = kernel_filter.transpose([1, 2, 3, 0])
    kernel_filter_tf = tf.constant(kernel_filter, dtype=tf.float32)

    contrast_depth = tf.nn.conv2d(inputs, kernel_filter_tf,
                                  strides=1, padding=[[0, 0], [1, 1], [1, 1], [0, 0]])
    print("contrast_depth", contrast_depth.shape)
    return contrast_depth


class CDLloss(losses.Loss):
    def call(self, y_true, y_pred):
        contrast_pred = contrast_depth_conv(y_pred)
        contrast_depth = contrast_depth_conv(y_true)

        # 回归损失函数
        depth_loss = tf.pow(contrast_pred - contrast_depth, 2)
        depth_loss = tf.reduce_mean(depth_loss)

        return depth_loss


def random_float(f_min, f_max):
    return f_min + (f_max - f_min) * random.random()


def Contrast_and_Brightness(img, alpha=None, gamma=None):
    gamma = random.randint(-40, 40)
    alpha = random_float(0.5, 1.5)
    dst = cv.addWeighted(img, alpha, img, 0, gamma)
    return dst


def get_cut_out(img, length=50):
    h, w = img.shape[0], img.shape[1]  # Tensor [1][2],  nparray [0][1]
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)
    length_new = np.random.randint(1, length)

    y1 = np.clip(y - length_new // 2, 0, h)
    y2 = np.clip(y + length_new // 2, 0, h)
    x1 = np.clip(x - length_new // 2, 0, w)
    x2 = np.clip(x + length_new // 2, 0, w)

    mask[y1: y2, x1: x2] = 0
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate((mask, mask, mask), axis=-1)
    img *= np.array(mask, np.uint8)
    return img


def imageEnhance(img, depth):
    """
    数据增强
    :param img: 输入图片
    :param depth: 输入深度图片
    :return: 输出图片，深度图
    """
    if random_float(0.0, 1.0) < 0.5:
        img = cv.flip(img, 1)
        depth = cv.flip(depth, 1)
    if random_float(0.0, 1.0) < 0.5:
        img = Contrast_and_Brightness(img)
    if random_float(0.0, 1.0) < 0.5:
        img = get_cut_out(img)
    return img, depth


def loadData(path_list=None):
    """
    :param path_list: ['data/train_images', 'data/train_depth]
    :return: x -> 'images' :  (batch_size, 256, 256, 3)
             y -> ['depths' : (batch_size, 32, 32, 1), labels : (batch_size, 1)]
    """
    if path_list is None:
        print("No dataset!!!")
        exit(-1)
    train_images, train_depths = path_list[0], path_list[1]
    images_list = []
    depths_list = []
    labels_list = []
    # ['data/train_images/...']
    images_paths = glob.glob(os.path.join(train_images, "*"))
    for images_path in images_paths:
        name_pure = os.path.split(images_path)[-1]
        label = name_pure.split('_')[-1]
        label = [0, 1] if label == '1' else [1, 0]
        depths = glob.glob(os.path.join(train_depths, name_pure, "*.jpg"))
        # # ['data/train_images/.../...']
        images = glob.glob(os.path.join(images_path, "*.jpg"))
        for image, depth in zip(images, depths):
            img = cv.imread(image)
            map = cv.imread(depth)
            img = cv.resize(img, (256, 256))
            map = cv.resize(map, (32, 32))
            shape = map.shape
            if len(shape) == 3 and shape[-1] == 3:
                map = cv.cvtColor(map, cv.COLOR_BGR2GRAY)
                map = np.expand_dims(map, axis=-1)

            img, map = imageEnhance(img, map)
            if len(map.shape) == 2:
                map = np.expand_dims(map, axis=-1)
            # 归一化
            img = img / 255.0
            # img = np.expand_dims(img, axis=0)
            map = map / 255.0
            # map = np.expand_dims(map, axis=0)
            images_list.append(img)
            labels_list.append(label)
            depths_list.append(map)

    # 拼接
    imgs = np.stack(images_list, axis=0)
    maps = np.stack(depths_list, axis=0)
    labels = np.stack(labels_list, axis=0)
    x_train = imgs
    y_train = [maps, labels]

    # print("images", imgs.shape)
    # print("labels", labels.shape)
    # print("depth", maps.shape)

    return x_train, y_train


# def loadTestData():
#     img = cv.imread('test1.jpg')
#     depth = np.zeros((32, 32, 1), dtype=np.float32)
#     # depth = cv.cvtColor(depth, cv.COLOR_BGR2GRAY)
#     # depth = np.expand_dims(depth, axis=-1)
#     depth = np.expand_dims(depth, axis=0)
#     img = np.expand_dims(img, axis=0)
#     depth = depth / 255.0
#     img = img / 255.0
#     print('image', img.shape)
#     print('depth', depth.shape)
#     x_test = img
#     y_test = [depth, [[1]]]
#     return x_test, y_test


if __name__ == '__main__':
    train_list = ['data/train_images', 'data/train_depths']
    # 加载测试数据集
    x_train, y_train = loadData(train_list)
    # x_test, y_test = loadTestData()
    model = get_FaceMapNet()
    # tf.keras.utils.plot_model(model, 'my_model.png')
    model.compile(optimizer=optimizers.Adam(lr=1e-5),
                  loss={
                      'depths_output' : CDLloss(),
                      'logits_output' : 'binary_crossentropy'
                  },
                  loss_weights = {
                      'depths_output' : 0.2,
                      'logits_output' : 0.8
                  },
                  metrics=['accuracy'])
    model.load_weights('model1/checkpoint')
    model.fit(x_train, y_train, epochs=20, batch_size=4)
    model.save_weights('model1/checkpoint')

    # model.load_weights('model_save/checkpoint')
    # output, logit = model(x_test)
    # print(output.numpy().max())
    # print(logit)
    #
    # x_test, y_test = loadTestData()
    # output = model(x_test)
    # depth = output[:, :, :, 0]
    # depth = tf.expand_dims(depth, axis=-1)
    # logit = tf.reduce_mean(depth, axis=1)
    # logit = tf.reduce_mean(logit, axis=1)
    # logit = tf.reduce_mean(logit, axis=1, keepdims=True)
    # logit = tf.concat([1 - logit, logit], axis=1)
    # print("输出概率", logit)
    # print("预测类型", tf.argmax(logit, axis=-1, output_type=tf.int32).numpy())

    # model.build(input_shape=(None, 256, 256, 3))
    # model.summary()
    # weights = model.get_weights()
    # for i, weight in enumerate(weights):
    #     print(i, weight.shape)
