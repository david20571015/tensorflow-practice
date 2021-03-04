import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model import Discriminator, Generator

(train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
# print(train_images.shape)
# print(train_labels.shape)

train_images = train_images / 255.
train_images = tf.reshape(train_images,
                          shape=(train_images.shape[0], 28, 28, 1))
train_images = tf.cast(train_images, dtype=tf.float32)
train_images = tf.concat([train_images, train_images], axis=0)
print('train_images: ', train_images.shape, train_images.dtype)

train_labels = tf.reshape(train_labels, shape=(train_labels.shape[0], 1))
train_labels = tf.cast(train_labels, dtype=tf.int32)
train_labels_fake = tf.random.uniform(train_labels.shape,
                                      minval=1,
                                      maxval=9,
                                      dtype=tf.int32)
train_labels_fake = tf.math.mod(train_labels + train_labels_fake, 9)
train_labels = tf.concat([train_labels, train_labels_fake], axis=0)
print('train_labels: ', train_labels.shape, train_labels.dtype)

train_match = tf.concat([tf.ones([60000, 1]), tf.zeros([60000, 1])], axis=0)
print('train_match: ', train_match.shape, train_match.dtype)

SAMPLE_SIZE = 16

sample_images = train_images[:SAMPLE_SIZE, :, :, :]
sample_labels = train_labels[:SAMPLE_SIZE]
print(sample_images.shape, sample_labels.shape)

fig = plt.figure()
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    ax.imshow(sample_images[i], cmap='gray')
    ax.set_title(sample_labels[i, 0].numpy())
plt.show()

BUFFER_SIZE = 60000
BATCH_SIZE = 256
RANDOM_SEED = 0

train_images_dataset = tf.data.Dataset.from_tensor_slices(
    train_images).shuffle(BUFFER_SIZE, seed=RANDOM_SEED).repeat(1).batch(
        BATCH_SIZE, drop_remainder=True)
train_labels_dataset = tf.data.Dataset.from_tensor_slices(
    train_labels).shuffle(BUFFER_SIZE, seed=RANDOM_SEED).repeat(1).batch(
        BATCH_SIZE, drop_remainder=True)
train_match_dataset = tf.data.Dataset.from_tensor_slices(train_match).shuffle(
    BUFFER_SIZE, seed=RANDOM_SEED).repeat(1).batch(BATCH_SIZE,
                                                   drop_remainder=True)
train_dataset = tf.data.Dataset.zip(
    (train_images_dataset, train_labels_dataset, train_match_dataset))

dataset_size = tf.data.experimental.cardinality(train_labels_dataset).numpy()


def gen_loss(fake_output):
    loss = tf.losses.binary_crossentropy(tf.ones_like(fake_output),
                                         fake_output,
                                         from_logits=True)
    return loss


def dis_loss(real_output, fake_output, matches):
    real_loss = tf.losses.binary_crossentropy(matches,
                                              real_output,
                                              from_logits=True)
    fake_loss = tf.losses.binary_crossentropy(tf.zeros_like(fake_output),
                                              fake_output,
                                              from_logits=True)
    return real_loss + fake_loss


gen_optimizer = tf.optimizers.Adam(1e-4)
dis_optimizer = tf.optimizers.Adam(1e-4)

log_dir = 'output'
ckpt_dir = os.path.join(log_dir, 'checkpoint')

EPOCHS = 50
DIS_TRAIN_REPEAT = 2
NOISE_DIM = 32
SAMPLE_SIZE = 256

generator = Generator(NOISE_DIM)
discriminator = Discriminator()

ckpt = tf.train.Checkpoint(generator=generator,
                           discriminator=discriminator,
                           generator_optimizer=gen_optimizer,
                           discriminator_optimizer=dis_optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=10)


@tf.function
def dis_train_step(images, labels, matches, noise):
    with tf.GradientTape() as dis_tape:
        fake_images = generator(labels, noise, training=False)

        real_output = discriminator(labels, images, training=True)
        fake_output = discriminator(labels, fake_images, training=True)

        dis_l = dis_loss(real_output, fake_output, matches)

    dis_grads = dis_tape.gradient(dis_l, discriminator.trainable_variables)
    dis_optimizer.apply_gradients(
        zip(dis_grads, discriminator.trainable_variables))


@tf.function
def gen_train_step(labels, noise):
    with tf.GradientTape() as gen_tape:
        fake_images = generator(labels, noise, training=True)

        # real_output = discriminator(labels, images)
        fake_output = discriminator(labels, fake_images, training=False)

        gen_l = gen_loss(fake_output)

    gen_grads = gen_tape.gradient(gen_l, generator.trainable_variables)
    gen_optimizer.apply_gradients(zip(gen_grads,
                                      generator.trainable_variables))


def generate_images(generator, condition, noise=None):
    if noise is None:
        noise = tf.random.normal([condition.shape[0], NOISE_DIM])
    generated_images = generator(condition, noise, training=False)

    fig = plt.figure(figsize=(5, 5))
    for i in range(min(16, condition.shape[0])):
        ax = plt.subplot(4, 4, i + 1)
        ax.imshow(generated_images[i], cmap='gray')
        ax.set_title(condition[i, 0].numpy())
    plt.show()


noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
for _, labels, _ in train_dataset.take(1):
    gen_train_step(labels, noise)

generator.summary()
discriminator.summary()


def train(dataset, epochs):
    for epoch in range(epochs):
        print(f'epoch: {epoch}')
        dis_probar = tf.keras.utils.Progbar(dataset_size * DIS_TRAIN_REPEAT)
        gen_probar = tf.keras.utils.Progbar(dataset_size)

        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

        for _ in range(DIS_TRAIN_REPEAT):
            for images, labels, matches in dataset:
                dis_train_step(images, labels, matches, noise)
                dis_probar.add(1)

        for _, labels, _ in dataset:
            gen_train_step(labels, noise)
            gen_probar.add(1)

        if (epoch + 1) % 10 == 0:
            generate_images(generator, sample_labels)


train(train_dataset, EPOCHS)
