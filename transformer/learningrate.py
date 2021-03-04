import tensorflow as tf


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_setps=4000) -> None:
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)

        self.warmup_step = warmup_setps

    def __call__(self, step) -> None:
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_step ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.minimum(arg1, arg2)
