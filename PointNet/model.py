import tensorflow as tf
import numpy as np
import keras
from keras import optimizers
from keras.layers import Input
from keras.models import Model
from keras.layers import Dense, Flatten, Reshape, Dropout
from keras.layers import Convolution1D, MaxPooling1D, BatchNormalization
from keras.layers import Lambda

def mat_mul(A, B):
    """
    :param A: operand A
    :param B: operand B
    :returns: A tf.Tensor type value (A × B)
    """
    return tf.matmul(A, B)


def build_model(num_points=68, cat=7):
    """
    :param num_points: number of points in each sample (default=53215)
    :param cat: number of categories (default=7)
    :returns: A Kears.model
    """

    # define optimizer
    adam = optimizers.Adam(lr=0.001, decay=0.7)

    # input_Transformation_net
    input_points = Input(shape=(num_points, 3))
    x = Convolution1D(64, 1, activation='relu', input_shape=(num_points, 3))(input_points)
    x = BatchNormalization()(x)
    x = Convolution1D(128, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(1024, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=num_points)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
    input_T = Reshape((3, 3))(x)

    # forward net
    g = Lambda(mat_mul, arguments={'B': input_T})(input_points)
    g = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
    g = BatchNormalization()(g)

    # feature transform net
    f = Convolution1D(64, 1, activation='relu')(g)
    f = BatchNormalization()(f)
    f = Convolution1D(128, 1, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Convolution1D(1024, 1, activation='relu')(f)
    f = BatchNormalization()(f)
    f = MaxPooling1D(pool_size=num_points)(f)
    f = Dense(512, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(256, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
    feature_T = Reshape((64, 64))(f)

    # forward net
    g = Lambda(mat_mul, arguments={'B': feature_T})(g)
    g = Convolution1D(64, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(128, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(1024, 1, activation='relu')(g)
    g = BatchNormalization()(g)

    # global_feature
    global_feature = MaxPooling1D(pool_size=num_points)(g)

    # point_net_cls
    c = Dense(512, activation='relu')(global_feature)
    c = BatchNormalization()(c)
    c = Dropout(rate=0.7)(c)
    c = Dense(256, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Dropout(rate=0.7)(c)
    c = Dense(cat, activation='softmax')(c)
    prediction = Flatten()(c)

    model = Model(inputs=input_points, outputs=prediction)

    return model
