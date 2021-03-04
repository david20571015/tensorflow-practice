import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, filter_nums):
        super(Encoder, self).__init__()

        seq = []
        for i, filter_num in enumerate(filter_nums):
            if i != len(filter_nums) - 1:
                seq.append(tf.keras.layers.Conv2D(
                    filter_num, 3, padding='same', activation='relu'))
                seq.append(tf.keras.layers.MaxPool2D((2, 2), padding='same'))
            else:
                seq.append(tf.keras.layers.Conv2D(
                    filter_num, 3, padding='same'))

        self.model = tf.keras.Sequential(seq)

    def call(self, x):
        return self.model(x)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, filter_nums):
        super(Decoder, self).__init__()

        seq = []
        for i, filter_num in enumerate(filter_nums):
            if i != len(filter_nums) - 1:
                seq.append(tf.keras.layers.Conv2DTranspose(
                    filter_num, 3, padding='same', activation='relu'))
                seq.append(tf.keras.layers.UpSampling2D((2, 2)))
            else:
                seq.append(tf.keras.layers.Conv2DTranspose(
                    filter_num, 3, padding='same'))

        self.model = tf.keras.Sequential(seq)

    def call(self, x):
        return self.model(x)


class AutoEncoder(tf.keras.Model):
    def __init__(self, encoder_filter_nums, input_size):
        assert input_size[0] >= 2**(len(encoder_filter_nums)-1)
        assert input_size[0] % 2**(len(encoder_filter_nums)-1) == 0

        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(encoder_filter_nums)

        decoder_filter_nums = encoder_filter_nums[-2::-1] + [1]
        self.decoder = Decoder(decoder_filter_nums)

    def call(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
