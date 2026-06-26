"""
Ablation variant: D1 + D3
  - D1 ON  : deeper per-channel feature extractor (2 conv layers)
  - D2 OFF : no bottleneck Dropout
  - D3 ON  : luminance-injection weight is a trainable scalar
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import keras


class MSEFBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm     = tf.keras.layers.LayerNormalization(axis=-1)
        self.depthwise_conv = layers.DepthwiseConv2D(kernel_size=(3, 3), padding='same')
        self.se_attn        = SEBlock(filters)
    def call(self, inputs):
        x = self.layer_norm(inputs)
        return layers.Add()([layers.Multiply()([self.depthwise_conv(x), self.se_attn(x)]), inputs])


class SEBlock(layers.Layer):
    def __init__(self, input_channels, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.pool = layers.GlobalAveragePooling2D()
        self.fc1  = layers.Dense(input_channels // reduction_ratio, activation='relu')
        self.fc2  = layers.Dense(input_channels, activation='tanh')
    def call(self, inputs):
        x = self.fc2(self.fc1(self.pool(inputs)))
        return inputs * tf.reshape(x, [-1, 1, 1, inputs.shape[-1]])


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size; self.num_heads = num_heads
        self.head_dim   = embed_size // num_heads
        self.query_dense = layers.Dense(embed_size); self.key_dense = layers.Dense(embed_size)
        self.value_dense = layers.Dense(embed_size); self.combine_heads = layers.Dense(embed_size)
    def split_heads(self, x, bs):
        return tf.transpose(tf.reshape(x, (bs,-1,self.num_heads,self.head_dim)), [0,2,1,3])
    def call(self, inputs):
        bs = tf.shape(inputs)[0]; h = tf.shape(inputs)[1]; w = tf.shape(inputs)[2]
        q = self.split_heads(self.query_dense(inputs), bs)
        k = self.split_heads(self.key_dense(inputs),   bs)
        v = self.split_heads(self.value_dense(inputs), bs)
        attn = tf.matmul(tf.nn.softmax(tf.matmul(q,k,transpose_b=True)/
               tf.math.sqrt(tf.cast(tf.shape(k)[-1],tf.float32)),axis=-1),v)
        out = self.combine_heads(tf.reshape(tf.transpose(attn,[0,2,1,3]),[bs,-1,self.embed_size]))
        return tf.reshape(out,[bs,h,w,self.embed_size])


class Denoiser(Model):
    """D2 OFF — no Dropout."""
    def __init__(self, num_filters, kernel_size=3, activation='relu'):
        super().__init__()
        self.conv1 = layers.Conv2D(num_filters, kernel_size, strides=1, padding='same', activation=activation)
        self.conv2 = layers.Conv2D(num_filters, kernel_size, strides=2, padding='same', activation=activation)
        self.conv3 = layers.Conv2D(num_filters, kernel_size, strides=2, padding='same', activation=activation)
        self.conv4 = layers.Conv2D(num_filters, kernel_size, strides=2, padding='same', activation=activation)
        self.bottleneck   = MultiHeadSelfAttention(embed_size=num_filters, num_heads=4)
        self.up2 = layers.UpSampling2D(2); self.up3 = layers.UpSampling2D(2); self.up4 = layers.UpSampling2D(2)
        self.output_layer = layers.Conv2D(1, kernel_size, strides=1, padding='same', activation='tanh')
        self.res_layer    = layers.Conv2D(1, kernel_size, strides=1, padding='same', activation='tanh')
    def call(self, inputs):
        x1=self.conv1(inputs); x2=self.conv2(x1); x3=self.conv3(x2); x4=self.conv4(x3)
        x=self.bottleneck(x4); x=self.up4(x); x=self.up3(x3+x); x=self.up2(x2+x); x=x+x1
        return self.output_layer(self.res_layer(x)+inputs)


class LYT(Model):
    """D1 ON, D2 OFF, D3 ON."""
    def __init__(self, filters=32, denoiser_cb=None, denoiser_cr=None):
        super().__init__()
        self.process_y  = self._make(filters)   # D1 ON
        self.process_cb = self._make(filters)
        self.process_cr = self._make(filters)
        self.denoiser_cb = denoiser_cb; self.denoiser_cr = denoiser_cr
        self.lum_pool = layers.MaxPooling2D(8)
        self.lum_mhsa = MultiHeadSelfAttention(embed_size=filters, num_heads=4)
        self.lum_up   = layers.UpSampling2D(8)
        self.lum_conv = layers.Conv2D(filters, (1,1), padding='same')
        self.ref_conv = layers.Conv2D(filters, (1,1), padding='same')
        self.msef     = MSEFBlock(filters)
        self.lum_weight = tf.Variable(0.2, trainable=True,   # D3 ON
                                       name='lum_injection_weight', dtype=tf.float32)
        self.recombine         = layers.Conv2D(filters, (3,3), activation='relu', padding='same')
        self.final_adjustments = layers.Conv2D(3, (3,3), activation='tanh', padding='same')
    def _make(self, filters):                    # D1 ON: 2 conv layers
        return keras.Sequential([
            layers.Conv2D(filters, (3,3), activation='relu', padding='same'),
            layers.Conv2D(filters, (3,3), activation='relu', padding='same'),
        ])
    def call(self, inputs):
        y, cb, cr = tf.split(tf.image.rgb_to_yuv(inputs), 3, axis=-1)
        cb = self.denoiser_cb(cb) + cb
        cr = self.denoiser_cr(cr) + cr
        lum  = self.process_y(y)
        ref  = self.ref_conv(tf.concat([self.process_cb(cb), self.process_cr(cr)], axis=-1))
        lum  = lum + self.lum_up(self.lum_mhsa(self.lum_pool(lum)))
        shortcut = ref
        ref  = self.msef(ref + self.lum_weight * self.lum_conv(lum)) + shortcut  # D3 ON
        return self.final_adjustments(self.recombine(tf.concat([ref, lum], axis=-1)))
