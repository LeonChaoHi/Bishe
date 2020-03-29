from keras.layers import Input
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Lambda
from keras.models import Model
from keras import optimizers
from keras.utils import plot_model
from keras import backend as K

# VGG16
import tensorflow as tf
from keras import Model, Sequential
from keras.layers import Flatten, Dense, Conv2D, GlobalAveragePooling2D
from keras.layers import Input, MaxPooling2D, GlobalMaxPooling2D


def head_pose_estimator(input_shape=(32, 32, 3), output_shape=3):
    input_ = Input(shape=input_shape)
    # block1
    x = Conv2D(16, (5, 5), activation="relu", padding="same", name="block1_conv")(input_)
    x = MaxPooling2D((3, 3), strides=(2, 2), name="block1_pool")(x)
    # block2
    x = Conv2D(20, (3, 3), activation="relu", padding="same", name="block2_conv")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)
    # regression block
    x = Flatten(name='flatten')(x)
    x = Dense(32, activation='relu', name='fc1')(x)
    output_ = Dense(output_shape, activation='relu', name='fc2')(x)
    model = Model(inputs=input_, outputs=output_)
    model.summary()

    opti_sgd = optimizers.sgd(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=opti_sgd, metrics=['mean_absolute_error'])
    return model


def VGG16(input_shape=(224, 224, 3), nclass=8):
    input_ = Input(shape=input_shape)
    # block1
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1")(input_)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)
    # block2
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)
    # block3
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)
    # block4
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)
    # block5
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    output_ = Dense(nclass, activation='softmax', name='fc3')(x)
    model = Model(inputs=input_, outputs=output_)
    model.summary()
    opti_sgd = optimizers.sgd(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=opti_sgd, metrics=['accuracy'])
    return model


def LRN(alpha=1e-4, k=2, beta=0.75, n=5):
    """
    LRN for cross channel normalization in the original Alexnet
    parameters default as original paper.
    """

    def f(X):
        b, r, c, ch = X.shape
        half = n // 2
        square = K.square(X)
        extra_channels = K.spatial_2d_padding(square, ((0, 0), (half, half)), data_format='channels_first')
        scale = k
        for i in range(n):
            scale += alpha * extra_channels[:, :, :, i:i + int(ch)]
        scale = scale ** beta
        return X / scale

    return Lambda(f, output_shape=lambda input_shape: input_shape)


def alexnet(input_shape=(224, 224, 3), nclass=8):
    """
    build Alexnet model using keras with TensorFlow backend.
    :param input_shape: input shape of network, default as (224,224,3)
    :param nclass: numbers of class(output shape of network), default as 1000
    :return: Alexnet model
    """
    # conv1
    input_ = Input(shape=input_shape)
    x = Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu')(input_)
    x = LRN()(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
    # conv2
    x = Conv2D(256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same')(x)
    x = LRN()(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
    # conv3
    x = Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    # conv4
    x = Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    # conv5
    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = Flatten()(x)
    # fc6
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    # fc7
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    # fc8
    output_ = Dense(nclass, activation='softmax')(x)
    model = Model(inputs=input_, outputs=output_)
    model.summary()
    opti_sgd = optimizers.sgd(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=opti_sgd, metrics=['accuracy'])
    return model


if __name__ == "__main__":
    model = head_pose_estimator()
    plot_model(model, "model.png", show_shapes=False)  # 保存模型图

