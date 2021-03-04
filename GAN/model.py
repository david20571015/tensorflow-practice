import tensorflow as tf
from tensorflow.keras.layers import (BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dense, Dropout,
                                     Embedding, Flatten, LeakyReLU, Reshape)


class Generator(tf.keras.Model):
    def __init__(self, noise_dim, n_class=10):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.n_class = n_class

        self.embedding = Embedding(n_class, 64)
        self.flatten = Flatten()
        self.expand_cond = Dense(128, use_bias=False, activation='relu')

        self.fc0 = Dense(64, activation='relu')
        self.fc1 = Dense(7 * 7 * 256, use_bias=False)
        self.bn1 = BatchNormalization()
        self.lrelu1 = LeakyReLU()
        self.reshape = Reshape((7, 7, 256))

        self.conv2Dt1 = Conv2DTranspose(128, (5, 5),
                                        strides=(1, 1),
                                        padding='same',
                                        use_bias=False)
        self.bn2 = BatchNormalization()
        self.lrelu2 = LeakyReLU()

        self.conv2Dt2 = Conv2DTranspose(64, (5, 5),
                                        strides=(2, 2),
                                        padding='same',
                                        use_bias=False)
        self.bn3 = BatchNormalization()
        self.lrelu3 = LeakyReLU()

        self.conv2Dt3 = Conv2DTranspose(1, (5, 5),
                                        strides=(2, 2),
                                        padding='same',
                                        use_bias=False)

    def call(self, condition, noise, training=True):
        condition = self.embedding(condition)
        condition = self.flatten(condition)
        condition = self.expand_cond(condition)

        x = tf.concat([condition, noise], axis=-1)

        x = self.fc0(x)
        x = self.fc1(x)
        x = self.bn1(x, training)
        x = self.lrelu1(x)
        x = self.reshape(x)

        x = self.conv2Dt1(x)
        x = self.bn2(x, training)
        x = self.lrelu2(x)

        x = self.conv2Dt2(x)
        x = self.bn3(x, training)
        x = self.lrelu3(x)

        return self.conv2Dt3(x)


class Discriminator(tf.keras.Model):
    def __init__(self, n_class=10):
        super(Discriminator, self).__init__()
        self.n_class = n_class

        self.conv2D1 = Conv2D(64, (5, 5), strides=(2, 2), padding='same')
        self.lrelu1 = LeakyReLU()
        self.dropout1 = Dropout(0.3)

        self.conv2D2 = Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        self.lrelu2 = LeakyReLU()
        self.dropout2 = Dropout(0.3)

        self.flatten1 = Flatten()

        self.fc1 = Dense(64, activation='relu')

        self.embedding = Embedding(n_class, 64)
        self.flatten2 = Flatten()

        self.fc2 = Dense(32, activation='relu')
        self.fc3 = Dense(1)

    def call(self, condition, image, training=True):
        x = self.conv2D1(image)
        x = self.lrelu1(x)
        x = self.dropout1(x, training)

        x = self.conv2D2(x)
        x = self.lrelu2(x)
        x = self.dropout2(x, training)

        x = self.flatten1(x)

        x = self.fc1(x)

        condition = self.embedding(condition)
        condition = self.flatten2(condition)

        x = tf.concat([x, condition], axis=-1)
        x = self.fc2(x)
        return self.fc3(x)
