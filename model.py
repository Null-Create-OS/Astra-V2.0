import tensorflow as tf

class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d, heads, dropout=0.1):
        super().__init__()
        self.d, self.heads = d, heads
        self.head_dim = d // heads
        self.q = tf.keras.layers.Dense(d)
        self.k = tf.keras.layers.Dense(d)
        self.v = tf.keras.layers.Dense(d)
        self.out = tf.keras.layers.Dense(d)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        q = self.q(x); k = self.k(x); v = self.v(x)
        b = tf.shape(x)[0]; s = tf.shape(x)[1]
        def split(z):
            z = tf.reshape(z, [b, s, self.heads, self.head_dim])
            return tf.transpose(z, [0, 2, 1, 3])
        q, k, v = split(q), split(k), split(v)
        scores = tf.matmul(q, k, transpose_b=True)
        scores /= tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        mask = tf.linalg.band_part(tf.ones([s, s]), -1, 0)
        scores = tf.where(mask[None, None, :, :] > 0, scores, tf.constant(-1e9))
        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.dropout(weights, training=training)
        y = tf.matmul(weights, v)
        y = tf.transpose(y, [0, 2, 1, 3])
        y = tf.reshape(y, [b, s, self.d])
        return self.out(y)

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d, heads, ff, dropout=0.1):
        super().__init__()
        self.n1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn = CausalSelfAttention(d, heads, dropout)
        self.n2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(ff, activation="gelu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(d),
        ])
        self.drop = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        x = x + self.drop(self.attn(self.n1(x), training=training), training=training)
        x = x + self.drop(self.ff(self.n2(x), training=training), training=training)
        return x

class Astra(tf.keras.Model):
    def __init__(self, vocab_size, seq, d=128, layers=3, heads=4, ff=256, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq = seq
        self.d = d
        self.token_emb = tf.keras.layers.Embedding(vocab_size, d)
        self.pos_emb = tf.keras.layers.Embedding(seq, d)
        self.blocks = [TransformerBlock(d, heads, ff, dropout) for _ in range(layers)]
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.lm_head = tf.keras.layers.Dense(vocab_size)

    def call(self, ids, training=False):
        length = tf.shape(ids)[1]
        pos = tf.range(length)
        x = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x, training=training)
        return self.lm_head(self.norm(x))
