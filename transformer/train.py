import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from learningrate import CustomSchedule
from model import Transformer
from config import Config

output_dir = Config['output_dir']
en_vocab_file = os.path.join(output_dir, 'en_vocab')
zh_vocab_file = os.path.join(output_dir, 'zh_vocab')
checkpoint_path = os.path.join(output_dir, 'checkpoints')
log_dir = os.path.join(output_dir, 'logs')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


ds_config = tfds.translate.wmt.WmtConfig(
    version='1.0.0',
    language_pair=('zh', 'en'),
    subsets={
        tfds.Split.TRAIN: ['newscommentary_v14']
    }
)
builder = tfds.builder('wmt_translate', config=ds_config)
builder.download_and_prepare()
train_examples, val_examples = builder.as_dataset(
    split=['train[:30%]', 'train[30%:31%]'], as_supervised=True)

print('-' * 50)
try:
    subword_encoder_en = tfds.features.text.SubwordTextEncoder.load_from_file(
        en_vocab_file)
    print(f'Load builded corpus: {en_vocab_file}')
except:
    print(f'Build corpus: {en_vocab_file}')
    subword_encoder_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for en, _ in train_examples),
        target_vocab_size=2**13
    )
    subword_encoder_en.save_to_file(en_vocab_file)

print(f'Size of corpus: {subword_encoder_en.vocab_size}')
input_vocab_size = subword_encoder_en.vocab_size + 2
print("input_vocab_size:", input_vocab_size)

try:
    subword_encoder_zh = tfds.features.text.SubwordTextEncoder.load_from_file(
        zh_vocab_file)
    print(f'Load builded corpus: {zh_vocab_file}')
except:
    print(f'Build corpus: {zh_vocab_file}')
    subword_encoder_zh = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (zh.numpy() for _, zh in train_examples),
        target_vocab_size=2**13,
        max_subword_length=3
    )
    subword_encoder_zh.save_to_file(zh_vocab_file)

print(f'Size of corpus: {subword_encoder_zh.vocab_size}')
target_vocab_size = subword_encoder_zh.vocab_size + 2
print("target_vocab_size:", target_vocab_size)
print('-' * 50)

MAX_LENGTH = Config['max_length']
BUFFER_SIZE = Config['buffer_size']
BATCH_SIZE = Config['batch_size']


def encode(en_t, zh_t):
    en_vocab_size = subword_encoder_en.vocab_size
    zh_vocab_size = subword_encoder_zh.vocab_size
    en_indices = [en_vocab_size] + \
        subword_encoder_en.encode(en_t.numpy()) + [en_vocab_size + 1]
    zh_indices = [zh_vocab_size] + \
        subword_encoder_zh.encode(zh_t.numpy()) + [zh_vocab_size + 1]
    return en_indices, zh_indices


def tf_encode(en_t, zh_t):
    return tf.py_function(encode, [en_t, zh_t], [tf.int64, tf.int64])


def filter_max_length(en, zh, max_length=MAX_LENGTH):
    return tf.logical_and(tf.size(en) <= max_length, tf.size(zh) <= max_length)


train_dataset = (train_examples
                 .map(tf_encode)
                 .filter(filter_max_length)
                 .padded_batch(BATCH_SIZE, padded_shapes=(MAX_LENGTH, MAX_LENGTH))
                 .cache()
                 .shuffle(BUFFER_SIZE)
                 .prefetch(tf.data.experimental.AUTOTUNE))

val_dataset = (val_examples
               .map(tf_encode)
               .filter(filter_max_length)
               .padded_batch(BATCH_SIZE, padded_shapes=(MAX_LENGTH, MAX_LENGTH))
               .cache()
               .shuffle(BUFFER_SIZE)
               .prefetch(tf.data.experimental.AUTOTUNE))

num_layers = Config['num_layers']
d_model = Config['d_model']
dff = Config['dff']
num_heads = Config['num_heads']
dropout_rate = Config['dropout_rate']

print('-' * 50)
print(f'num_layers: {num_layers}\nd_model: {d_model}\ndff: {dff}\nnum_heads: {num_heads}\ndropout_rate: {dropout_rate}')
print('-' * 50)

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, dropout_rate)

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.logical_not(tf.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_padding_mask(seq):
    mask = tf.cast(tf.equal(seq, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')
val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')


run_id = f"{num_layers}layers_{d_model}d_{num_heads}heads_{dff}dff"
checkpoint_path = os.path.join(checkpoint_path, run_id)
log_dir = os.path.join(log_dir, run_id)


train_summary_writer = tf.summary.create_file_writer(
    os.path.join(log_dir, 'train'))
val_summary_writer = tf.summary.create_file_writer(
    os.path.join(log_dir, 'val'))

tf.summary.trace_on(graph=True, profiler=False)


@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(
            inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


@tf.function
def val_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        inp, tar_inp)

    predictions, _ = transformer(
        inp, tar_inp, False, enc_padding_mask, combined_mask, dec_padding_mask)
    loss = loss_function(tar_real, predictions)

    val_loss(loss)
    val_accuracy(tar_real, predictions)


(inp, tar) = next(iter(train_dataset))

with train_summary_writer.as_default():
    func_graph = val_step.get_concrete_function(inp, tar).graph
    tf.python.ops.summary_ops_v2.graph(func_graph.as_graph_def(), step=0)

transformer.summary()


ckpt = tf.train.Checkpoint(model=transformer, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    last_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1]) * 5
    print(f'Read checkpoint, {last_epoch} epoches have beed trained.')
else:
    last_epoch = 0
    print('Checkpoint not found.')
print('-' * 50)

iter_num = len(list(train_dataset))

EPOCHS = Config['epochs']

for epoch in range(last_epoch, EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()
    val_loss.reset_states()
    val_accuracy.reset_states()

    probar = tf.keras.utils.Progbar(iter_num)
    for (step_idx, (inp, tar)) in enumerate(train_dataset):
        train_step(inp, tar)
        probar.add(1)

    for (step_idx, (inp, tar)) in enumerate(val_dataset):
        val_step(inp, tar)

    if (epoch + 1) % Config['save_interval'] == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path))

    with train_summary_writer.as_default():
        tf.summary.scalar("train_loss", train_loss.result(), step=epoch)
        tf.summary.scalar("train_acc", train_accuracy.result(), step=epoch)

    with val_summary_writer.as_default():
        tf.summary.scalar("val_loss", val_loss.result(), step=epoch)
        tf.summary.scalar("val_acc", val_accuracy.result(), step=epoch)

    print(f'Epoch {epoch}\
         Loss {train_loss.result():.4f}\
         Accuracy {train_accuracy.result():.4f}\
         Val_Loss {val_loss.result():.4f}\
         Val_Accuracy {val_accuracy.result():.4f}')

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
