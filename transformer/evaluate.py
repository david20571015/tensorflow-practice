import os
try:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
except:
    pass
import tensorflow as tf
import tensorflow_datasets as tfds
import string

from model import Transformer
from config import Config


output_dir = Config['output_dir']
en_vocab_file = os.path.join(output_dir, 'en_vocab')
zh_vocab_file = os.path.join(output_dir, 'zh_vocab')
checkpoint_path = os.path.join(output_dir, 'checkpoints')
log_dir = os.path.join(output_dir, 'logs')


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


print('-' * 50)

try:
    subword_encoder_en = tfds.features.text.SubwordTextEncoder.load_from_file(
        en_vocab_file)
    print(f'Load builded corpus: {en_vocab_file}')
    print(f'Size of corpus: {subword_encoder_en.vocab_size}')
    input_vocab_size = subword_encoder_en.vocab_size + 2
    print("input_vocab_size:", input_vocab_size)
except:
    print(f'Builded en corpus not found.')
    exit()

try:
    subword_encoder_zh = tfds.features.text.SubwordTextEncoder.load_from_file(
        zh_vocab_file)
    print(f'Load builded corpus: {zh_vocab_file}')
    print(f'Size of corpus: {subword_encoder_zh.vocab_size}')
    target_vocab_size = subword_encoder_zh.vocab_size + 2
    print("target_vocab_size:", target_vocab_size)
except:
    print(f'Builded zh corpus not found.')
    exit()

print('-' * 50)

model_name = '2layers_256d_8heads_512dff'
checkpoint_id = 'ckpt-4'

model_info = [i for i in model_name if i not in string.ascii_letters]
model_info = ''.join(model_info).split('_')

num_layers, d_model, num_heads, dff = [int(info) for info in model_info]

transformer = Transformer(num_layers, d_model, num_heads,
                          dff, input_vocab_size, target_vocab_size)

checkpoint = tf.train.Checkpoint(model=transformer)
checkpoint.restore(
    'nmt/checkpoints/' + model_name + '/' + checkpoint_id)

MAX_LENGTH = 100


def evaluate(inp_sentence):
    start_token = [subword_encoder_en.vocab_size]
    end_token = [subword_encoder_en.vocab_size + 1]
    inp_sentence = start_token + \
        subword_encoder_en.encode(inp_sentence) + end_token

    encoder_input = tf.expand_dims(inp_sentence, 0)
    decoder_input = [subword_encoder_zh.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for _ in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        predictions = predictions[:, -1:, :]

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id, subword_encoder_zh.vocab_size + 1):
            return tf.squeeze(output, axis=0), attention_weights

        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


if __name__ == '__main__':
    sentence = 'That means she must figure out how to get each child logged into their classes on time. It means she follows along to make sure the kids stay focused. It means she provides their only outside help.'

    predicted_seq, _ = evaluate(sentence)

    target_vocab_size = subword_encoder_zh.vocab_size
    predicted_seq_without_bos_eos = [
        idx for idx in predicted_seq if idx < target_vocab_size]
    predicted_sentence = subword_encoder_zh.decode(
        predicted_seq_without_bos_eos)

    print("=" * 20)
    print("original_sentence:", sentence)
    print("-" * 20)
    print("predicted_sentence:", predicted_sentence)
    print("=" * 20)
