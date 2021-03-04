import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose


class Encoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        self.conv2D_1 = Conv2D(32, 3, 2, activation='relu')
        # shape = (batch_size, 32, 32, 16)
        self.conv2D_2 = Conv2D(64, 3, 2, activation='relu')
        # shape = (batch_size, 16, 16, 32)
        self.conv2D_3 = Conv2D(128, 3, 2, activation='relu')
        # shape = (batch_size, 8, 8, 64)

        self.flatten = Flatten()
        self.latent = Dense(latent_dim * 2)

    def call(self, x):
        x = self.conv2D_1(x)
        x = self.conv2D_2(x)
        x = self.conv2D_3(x)

        x = self.flatten(x)
        x = self.latent(x)
        return x


class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()

        self.from_latent = Dense(8 * 8 * 128, activation='relu')
        self.reshape = Reshape((8, 8, 128))

        self.conv2DT_1 = Conv2DTranspose(64,
                                         3,
                                         2,
                                         activation='relu',
                                         padding='same')
        self.conv2DT_2 = Conv2DTranspose(32,
                                         3,
                                         2,
                                         activation='relu',
                                         padding='same')
        self.conv2DT_3 = Conv2DTranspose(3, 3, 2, padding='same')

    def call(self, x):
        x = self.from_latent(x)
        x = self.reshape(x)

        x = self.conv2DT_1(x)
        x = self.conv2DT_2(x)
        x = self.conv2DT_3(x)
        return x


class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder()

    def call(self, x):
        mean, logvar = self.encode(x)
        latent = self.compute_latent(mean, logvar)
        return self.decode(latent)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def compute_latent(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, latent, apply_sigmoid=False):
        logits = self.decoder(latent)
        return tf.sigmoid(logits) if apply_sigmoid else logits
