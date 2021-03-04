try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
except ImportError:
    print('import os error')

import time
import tensorflow as tf
from tensorflow.keras import datasets, layers
from resnet import ResNet

(train_images, train_labels), (test_images,
                               test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images/255.0, test_images/255.0

print(train_images.shape, train_labels.shape)


class CNN(tf.keras.Model):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        self.conv1 = layers.Conv2D(32, 3, activation=tf.nn.relu)
        self.maxpool1 = layers.MaxPooling2D((2, 2))
        self.conv2 = layers.Conv2D(64, 3, activation=tf.nn.relu)
        self.maxpool2 = layers.MaxPooling2D((2, 2))
        self.conv3 = layers.Conv2D(64, 3, activation=tf.nn.relu)

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(64, activation=tf.nn.relu)
        self.fc2 = layers.Dense(64, activation=tf.nn.relu)
        self.fc3 = layers.Dense(10)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# model = ResNet([3, 4, 6, 3], 10)
model = CNN()
model(tf.zeros([1, *(train_images[0].shape)]))
model.summary()

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_obj(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    loss = loss_obj(labels, predictions)

    test_loss(loss)
    test_accuracy(labels, predictions)


train_ds = tf.data.Dataset.from_tensor_slices(
    (train_images, train_labels)).shuffle(1000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices(
    (test_images, test_labels)).batch(32)

# tf.summary.trace_on(graph=True, profiler=True)
# train_log_dir = 'logs/train'
# test_log_dir = 'logs/test'
# train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# test_summary_writer = tf.summary.create_file_writer(test_log_dir)

start = time.time()
EPOCHS = 20
for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    progbar = tf.keras.utils.Progbar(len(train_ds))
    for i, (images, labels) in enumerate(train_ds):
        train_step(images, labels)
        progbar.update(i+1)
    # with train_summary_writer.as_default():
    #     tf.summary.scalar('loss', train_loss.result(), step=epoch)
    #     tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

    for images, labels in test_ds:
        test_step(images, labels)
    # with test_summary_writer.as_default():
    #     tf.summary.scalar('loss', test_loss.result(), step=epoch)
    #     tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100))
total_time = time.time()-start
print(f'total time :　{total_time}')
