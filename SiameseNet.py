from tensorflow.keras.layers import Dense, Lambda, Conv3D, Dropout, Input, MaxPooling3D, GlobalAveragePooling3D, BatchNormalization, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

from DataLoader import DataLoader

import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


class SiameseNet:
    def __init__(self):
        self.data_loader = DataLoader()

        self.image_size = 128
        self.num_channels = 1
        self.num_images = 10
        self.embedding_size = 256

        self.init_filters = 64

        self.batch_size = 4
        self.epochs = 5000

        self.siamese_model = self._build_siamese()
        self.siamese_model.summary()

    def _feature_extractor(self):
        input_tensor = Input((self.image_size, self.image_size, self.num_images, self.num_channels))

        x = Conv3D(self.init_filters, 3, strides=2, padding="same", activation="relu")(input_tensor)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Conv3D(self.init_filters * 2, 3, strides=2, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Conv3D(self.init_filters * 4, 3, strides=2, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Conv3D(self.init_filters * 8, 3, strides=2, padding="same", activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Flatten()(x)
        outputs = Dense(self.embedding_size)(x)

        model = Model(input_tensor, outputs)
        model.summary()

        return model

    def _build_siamese(self):

        def euclidean_distance(vectors):
            featsA, featsB = vectors
            sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
            return K.sqrt(K.maximum(sumSquared, K.epsilon()))

        input_tensor_A = Input((self.image_size, self.image_size, self.num_images, self.num_channels))
        input_tensor_B = Input((self.image_size, self.image_size, self.num_images, self.num_channels))

        feat_model = self._feature_extractor()

        featsA = feat_model(input_tensor_A)
        featsB = feat_model(input_tensor_B)
        distance = Lambda(euclidean_distance)([featsA, featsB])

        outputs = Dense(1, activation="sigmoid")(distance)
        model = Model(inputs=[input_tensor_A, input_tensor_B], outputs=outputs)

        return model

    def build_and_train(self):
        optimizer = Adam(lr=0.0005, beta_1=0.5)
        self.siamese_model.compile(optimizer, loss='binary_crossentropy')
        start = time.time()
        losses = []
        last_time = 0
        for epoch in range(self.epochs):
            X, y = self.data_loader.load_batch(self.batch_size)
            loss = self.siamese_model.train_on_batch([X[:, 0, :, :, :, :], X[:, 1, :, :, :, :]], y)
            now = time.time()
            print(f'[Epoch: {epoch}/{self.epochs}]\tLoss: {round(loss, 4)}\t[Elapsed: {round(now - start)}s.\tCurr: {round(now - last_time, 2)}\tLeft: {round((now - last_time) * (self.epochs - epoch), 2)}]')
            last_time = now
            losses.append(loss)
            plt.cla()
            plt.plot(losses)
            plt.pause(0.1)
            if epoch % 100 == 0:
                self.siamese_model.save(f'./saved/siamese_model_{epoch}.h5')
        self.siamese_model.save('siamese_model.h5')


net = SiameseNet()
net.build_and_train()

