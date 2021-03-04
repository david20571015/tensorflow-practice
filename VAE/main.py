import math
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pathlib

from model import VAE

image_dir = 'data'
image_dir = pathlib.Path(image_dir)

BATCH_SIZE = 32
IMG_HEIGHT = IMG_WIDTH = 64

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    image_dir,
    label_mode=None,
    validation_split=0.2,
    subset='training',
    seed=0,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    image_dir,
    label_mode=None,
    validation_split=0.2,
    subset='validation',
    seed=0,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

train_ds = train_ds.map(lambda x: x / 255).cache().prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE)
val_ds = val_ds.map(lambda x: x / 255).cache().prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE)
sample_images = next(iter(train_ds))[:16]


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    log_normal = -0.5 * (
        (sample - mean)**2. * tf.exp(-logvar) + logvar + log2pi)
    return tf.reduce_sum(log_normal, axis=raxis)


def compute_loss(model, x, kl_factor=1.):
    mean, logvar = model.encode(x)
    z = model.compute_latent(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit,
                                                        labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    kl_term = kl_factor * (logpz - logqz_x)
    return -tf.reduce_mean(logpx_z + kl_term)


epochs = 100
num_examples_to_generate = 16
latent_dim = 256
kl_factor = 10.0
save_interval = 5

model = VAE(latent_dim)
model_id = f'dim_{model.latent_dim}_kl_{int(kl_factor)}'

optimizer = tf.optimizers.Adam(1e-4)

# output_dir = 'output'
# checkpoint_path = os.path.join(output_dir, 'checkpoints', model_id)
# log_dir = os.path.join(output_dir, 'logs', model_id)
# images_save_dir = os.path.join(output_dir, 'images', model_id)

# train_loss = tf.keras.metrics.Mean(name='train_ELBO')
# val_loss = tf.keras.metrics.Mean(name='val_ELBO')
# summary_writer = tf.summary.create_file_writer(log_dir)
# tf.summary.trace_on(graph=True, profiler=False)

# ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

# if ckpt_manager.latest_checkpoint:
#     ckpt.restore(ckpt_manager.latest_checkpoint)
#     last_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1]) * 5
#     print(f'Read checkpoint, {last_epoch} epoches have beed trained.')
# else:
#     last_epoch = 0
#     print('Checkpoint not found.')

# random_vector_for_generation = tf.random.normal(
#     shape=(num_examples_to_generate, model.latent_dim))

# @tf.function
# def train_step(model, x):
#     with tf.GradientTape() as tape:
#         loss = compute_loss(model, x, kl_factor)
#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#     train_loss(loss)

# @tf.function
# def val_step(model, x):
#     loss = compute_loss(model, x, kl_factor)

#     val_loss(loss)

image = next(iter(train_ds))
print(image.shape)
model(image)

# with summary_writer.as_default():
#     func_graph = val_step.get_concrete_function(model, image).graph
#     tf.python.ops.summary_ops_v2.graph(func_graph.as_graph_def(), step=0)

# model.summary()

# def generate_and_save_images(model, epoch, test_sample):
#     mean, logvar = model.encode(test_sample)
#     latent = model.compute_latent(mean, logvar)
#     predictions = model.decode(latent, apply_sigmoid=True)

#     if not os.path.exists(images_save_dir):
#         os.makedirs(images_save_dir)

#     fig = plt.figure(figsize=(7, 7))
#     num_of_images = predictions.shape[0]
#     egde_len = math.ceil(math.sqrt(num_of_images))
#     for i in range(num_of_images):
#         plt.subplot(egde_len, egde_len, i+1)
#         plt.imshow(predictions[i])
#     plt.savefig(os.path.join(images_save_dir, f'epoch_{epoch}.png'))
#     # plt.show()
#     plt.close()

# iter_len = tf.data.experimental.cardinality(train_ds).numpy()
# print(f'iter_len: {iter_len}')

# if not ckpt_manager.latest_checkpoint:
#     generate_and_save_images(model, 0, sample_images)

# for epoch in range(last_epoch+1, epochs+1):
#     train_loss.reset_states()
#     val_loss.reset_states()

#     probar = tf.keras.utils.Progbar(iter_len)

#     for train_x in train_ds:
#         train_step(model, train_x)
#         probar.add(1)

#     for val_x in val_ds:
#         val_step(model, val_x)

#     print(f'Epoch {epoch}\
#          Loss {train_loss.result():.4f}\
#          Val_Loss {val_loss.result(): .4f}')

#     if epoch % save_interval == 0:
#         ckpt_save_path = ckpt_manager.save()
#         print(f'Save checkpoint for epoch {epoch} at {ckpt_save_path}.')
#         generate_and_save_images(model, epoch, sample_images)

#     with summary_writer.as_default():
#         tf.summary.scalar('train_ELBO', train_loss.result(), step=epoch)
#         tf.summary.scalar('val_ELBO', val_loss.result(), step=epoch)
