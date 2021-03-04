import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential


class ResBlock(layers.Layer):
    def __init__(self, filter_nums, strides=1):
        super(ResBlock, self).__init__()

        self.conv_1 = layers.Conv2D(
            filter_nums, (3, 3), strides=strides, padding='same')
        self.bn_1 = layers.BatchNormalization()
        self.relu_1 = layers.ReLU()

        self.conv_2 = layers.Conv2D(
            filter_nums, (3, 3), strides=1, padding='same')
        self.bn_2 = layers.BatchNormalization()
        self.relu_2 = layers.ReLU()

        if strides != 1:
            self.block = Sequential()
            self.block.add(layers.Conv2D(filter_nums, (1, 1), strides=strides))
        else:
            self.block = lambda x: x

    def call(self, input, training=None):
        x = self.bn_1(input, training=training)
        x = self.relu_1(x)
        x = self.conv_1(x)

        x = self.bn_2(x, training=training)
        x = self.relu_2(x)
        x = self.conv_2(x)

        identity = self.block(input)

        output = layers.add([x, identity])
        return output


class ResNet(tf.keras.Model):
    def __init__(self, layers_dims, class_nums=10):
        super(ResNet, self).__init__()

        self.model = Sequential([layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same'),
                                 layers.BatchNormalization(),
                                 layers.Activation('relu'),
                                 layers.MaxPooling2D(pool_size=(
                                     3, 3), strides=(2, 2), padding='same')
                                 ])

        self.layer_1 = self.ResNet_build(64, layers_dims[0])
        self.layer_2 = self.ResNet_build(128, layers_dims[1], strides=2)
        self.layer_3 = self.ResNet_build(256, layers_dims[2], strides=2)
        self.layer_4 = self.ResNet_build(512, layers_dims[3], strides=2)
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(class_nums)

    def call(self, input):
        x = self.model(input)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

    def ResNet_build(self, filter_nums, block_nums, strides=1):
        build_model = Sequential()
        build_model.add(ResBlock(filter_nums, strides))
        for _ in range(1, block_nums):
            build_model.add(ResBlock(filter_nums, strides=1))
        return build_model
