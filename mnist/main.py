from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import tensorflow as tf
import time

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(1000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)


class SimpleCNN(Model):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = Conv2D(32, 3, activation=tf.nn.relu)
        self.conv2 = Conv2D(64, 3, activation=tf.nn.relu)
        self.flatten = Flatten()
        self.fc1 = Dense(32, activation=tf.nn.relu)
        self.fc2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


model = SimpleCNN()
# _ = model(tf.zeros([1, *(x_train[0].shape)]))
# model.summary()

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


# tf.summary.trace_on(graph=True, profiler=True)
# train_log_dir = 'logs/train'
# test_log_dir = 'logs/test'
# train_summary_writer = tf.summary.create_file_writer(train_log_dir)
# test_summary_writer = tf.summary.create_file_writer(test_log_dir)


EPOCHS = 5
for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    start = time.time()

    for images, labels in train_ds:
        train_step(images, labels)
    # with train_summary_writer.as_default():
    #     tf.summary.scalar('loss', train_loss.result(), step=epoch)
    #     tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
    # with test_summary_writer.as_default():
    #     tf.summary.scalar('loss', test_loss.result(), step=epoch)
    #     tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

    end = time.time()

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}, time: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100,
                          test_loss.result(),
                          test_accuracy.result() * 100,
                          end-start))
