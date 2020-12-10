# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 11:33:48 2020

@author: Varchita Lalwani
"""

import tensorflow as tf
import time
from transformers import BertTokenizer
from rouge.rouge import rouge_n_sentence_level
import numpy as np

tensor = tf.compat.v1.train.NewCheckpointReader("bert_model.ckpt")
print(tensor.get_variable_to_shape_map())

with tf.compat.v1.Session() as sess:
    saver = tf.compat.v1.train.import_meta_graph('bert_model.ckpt.meta')
    saver.restore(sess, "bert_model.ckpt")
    
    embedding_matrix = sess.run('bert/embeddings/word_embeddings:0')
    
    query_kernel_layer_0 = sess.run('bert/encoder/layer_0/attention/self/query/kernel:0')
    query_bias_layer_0 = sess.run('bert/encoder/layer_0/attention/self/query/bias:0')
    key_kernel_layer_0 = sess.run('bert/encoder/layer_0/attention/self/key/kernel:0')
    key_bias_layer_0 = sess.run('bert/encoder/layer_0/attention/self/key/bias:0')
    value_kernel_layer_0 = sess.run('bert/encoder/layer_0/attention/self/value/kernel:0')
    value_bias_layer_0 = sess.run('bert/encoder/layer_0/attention/self/value/bias:0')
    bias_layer_0 = sess.run('bert/encoder/layer_0/attention/output/dense/bias:0')
    kernel_layer_0 = sess.run('bert/encoder/layer_0/attention/output/dense/kernel:0')
    
    query_kernel_layer_1 = sess.run('bert/encoder/layer_1/attention/self/query/kernel:0')
    query_bias_layer_1 = sess.run('bert/encoder/layer_1/attention/self/query/bias:0')
    key_kernel_layer_1 = sess.run('bert/encoder/layer_1/attention/self/key/kernel:0')
    key_bias_layer_1 = sess.run('bert/encoder/layer_1/attention/self/key/bias:0')
    value_kernel_layer_1 = sess.run('bert/encoder/layer_1/attention/self/value/kernel:0')
    value_bias_layer_1 = sess.run('bert/encoder/layer_1/attention/self/value/bias:0')
    bias_layer_1 = sess.run('bert/encoder/layer_1/attention/output/dense/bias:0')
    kernel_layer_1 = sess.run('bert/encoder/layer_1/attention/output/dense/kernel:0')
    
    query_kernel_layer_2 = sess.run('bert/encoder/layer_2/attention/self/query/kernel:0')
    query_bias_layer_2 = sess.run('bert/encoder/layer_2/attention/self/query/bias:0')
    key_kernel_layer_2 = sess.run('bert/encoder/layer_2/attention/self/key/kernel:0')
    key_bias_layer_2 = sess.run('bert/encoder/layer_2/attention/self/key/bias:0')
    value_kernel_layer_2 = sess.run('bert/encoder/layer_2/attention/self/value/kernel:0')
    value_bias_layer_2 = sess.run('bert/encoder/layer_2/attention/self/value/bias:0')
    bias_layer_2 = sess.run('bert/encoder/layer_2/attention/output/dense/bias:0')
    kernel_layer_2 = sess.run('bert/encoder/layer_2/attention/output/dense/kernel:0')
    
    query_kernel_layer_3 = sess.run('bert/encoder/layer_3/attention/self/query/kernel:0')
    query_bias_layer_3 = sess.run('bert/encoder/layer_3/attention/self/query/bias:0')
    key_kernel_layer_3 = sess.run('bert/encoder/layer_3/attention/self/key/kernel:0')
    key_bias_layer_3 = sess.run('bert/encoder/layer_3/attention/self/key/bias:0')
    value_kernel_layer_3 = sess.run('bert/encoder/layer_3/attention/self/value/kernel:0')
    value_bias_layer_3 = sess.run('bert/encoder/layer_3/attention/self/value/bias:0')
    bias_layer_3 = sess.run('bert/encoder/layer_3/attention/output/dense/bias:0')
    kernel_layer_3 = sess.run('bert/encoder/layer_3/attention/output/dense/kernel:0')
    
    
    query_kernel_layer_4 = sess.run('bert/encoder/layer_4/attention/self/query/kernel:0')
    query_bias_layer_4 = sess.run('bert/encoder/layer_4/attention/self/query/bias:0')
    key_kernel_layer_4 = sess.run('bert/encoder/layer_4/attention/self/key/kernel:0')
    key_bias_layer_4 = sess.run('bert/encoder/layer_4/attention/self/key/bias:0')
    value_kernel_layer_4 = sess.run('bert/encoder/layer_4/attention/self/value/kernel:0')
    value_bias_layer_4 = sess.run('bert/encoder/layer_4/attention/self/value/bias:0')
    bias_layer_4 = sess.run('bert/encoder/layer_4/attention/output/dense/bias:0')
    kernel_layer_4 = sess.run('bert/encoder/layer_4/attention/output/dense/kernel:0')
    
    query_kernel_layer_5 = sess.run('bert/encoder/layer_5/attention/self/query/kernel:0')
    query_bias_layer_5 = sess.run('bert/encoder/layer_5/attention/self/query/bias:0')
    key_kernel_layer_5 = sess.run('bert/encoder/layer_5/attention/self/key/kernel:0')
    key_bias_layer_5 = sess.run('bert/encoder/layer_5/attention/self/key/bias:0')
    value_kernel_layer_5 = sess.run('bert/encoder/layer_5/attention/self/value/kernel:0')
    value_bias_layer_5 = sess.run('bert/encoder/layer_5/attention/self/value/bias:0')
    bias_layer_5 = sess.run('bert/encoder/layer_5/attention/output/dense/bias:0')
    kernel_layer_5 = sess.run('bert/encoder/layer_5/attention/output/dense/kernel:0')
    
    
    query_kernel_layer_6 = sess.run('bert/encoder/layer_6/attention/self/query/kernel:0')
    query_bias_layer_6 = sess.run('bert/encoder/layer_6/attention/self/query/bias:0')
    key_kernel_layer_6 = sess.run('bert/encoder/layer_6/attention/self/key/kernel:0')
    key_bias_layer_6 = sess.run('bert/encoder/layer_6/attention/self/key/bias:0')
    value_kernel_layer_6 = sess.run('bert/encoder/layer_6/attention/self/value/kernel:0')
    value_bias_layer_6 = sess.run('bert/encoder/layer_6/attention/self/value/bias:0')
    bias_layer_6 = sess.run('bert/encoder/layer_6/attention/output/dense/bias:0')
    kernel_layer_6 = sess.run('bert/encoder/layer_6/attention/output/dense/kernel:0')
    
    query_kernel_layer_7 = sess.run('bert/encoder/layer_7/attention/self/query/kernel:0')
    query_bias_layer_7 = sess.run('bert/encoder/layer_7/attention/self/query/bias:0')
    key_kernel_layer_7 = sess.run('bert/encoder/layer_7/attention/self/key/kernel:0')
    key_bias_layer_7 = sess.run('bert/encoder/layer_7/attention/self/key/bias:0')
    value_kernel_layer_7 = sess.run('bert/encoder/layer_7/attention/self/value/kernel:0')
    value_bias_layer_7 = sess.run('bert/encoder/layer_7/attention/self/value/bias:0')
    bias_layer_7 = sess.run('bert/encoder/layer_7/attention/output/dense/bias:0')
    kernel_layer_7 = sess.run('bert/encoder/layer_7/attention/output/dense/kernel:0')
    
    query_kernel_layer_8 = sess.run('bert/encoder/layer_8/attention/self/query/kernel:0')
    query_bias_layer_8 = sess.run('bert/encoder/layer_8/attention/self/query/bias:0')
    key_kernel_layer_8 = sess.run('bert/encoder/layer_8/attention/self/key/kernel:0')
    key_bias_layer_8 = sess.run('bert/encoder/layer_8/attention/self/key/bias:0')
    value_kernel_layer_8 = sess.run('bert/encoder/layer_8/attention/self/value/kernel:0')
    value_bias_layer_8 = sess.run('bert/encoder/layer_8/attention/self/value/bias:0')
    bias_layer_8 = sess.run('bert/encoder/layer_8/attention/output/dense/bias:0')
    kernel_layer_8 = sess.run('bert/encoder/layer_8/attention/output/dense/kernel:0')
    
    query_kernel_layer_9 = sess.run('bert/encoder/layer_9/attention/self/query/kernel:0')
    query_bias_layer_9 = sess.run('bert/encoder/layer_9/attention/self/query/bias:0')
    key_kernel_layer_9 = sess.run('bert/encoder/layer_9/attention/self/key/kernel:0')
    key_bias_layer_9 = sess.run('bert/encoder/layer_9/attention/self/key/bias:0')
    value_kernel_layer_9 = sess.run('bert/encoder/layer_9/attention/self/value/kernel:0')
    value_bias_layer_9 = sess.run('bert/encoder/layer_9/attention/self/value/bias:0')
    bias_layer_9 = sess.run('bert/encoder/layer_9/attention/output/dense/bias:0')
    kernel_layer_9 = sess.run('bert/encoder/layer_9/attention/output/dense/kernel:0')
    
    query_kernel_layer_10 = sess.run('bert/encoder/layer_10/attention/self/query/kernel:0')
    query_bias_layer_10 = sess.run('bert/encoder/layer_10/attention/self/query/bias:0')
    key_kernel_layer_10 = sess.run('bert/encoder/layer_10/attention/self/key/kernel:0')
    key_bias_layer_10 = sess.run('bert/encoder/layer_10/attention/self/key/bias:0')
    value_kernel_layer_10 = sess.run('bert/encoder/layer_10/attention/self/value/kernel:0')
    value_bias_layer_10 = sess.run('bert/encoder/layer_10/attention/self/value/bias:0')
    bias_layer_10 = sess.run('bert/encoder/layer_10/attention/output/dense/bias:0')
    kernel_layer_10 = sess.run('bert/encoder/layer_10/attention/output/dense/kernel:0')
    
    query_kernel_layer_11 = sess.run('bert/encoder/layer_11/attention/self/query/kernel:0')
    query_bias_layer_11 = sess.run('bert/encoder/layer_11/attention/self/query/bias:0')
    key_kernel_layer_11 = sess.run('bert/encoder/layer_11/attention/self/key/kernel:0')
    key_bias_layer_11 = sess.run('bert/encoder/layer_11/attention/self/key/bias:0')
    value_kernel_layer_11 = sess.run('bert/encoder/layer_11/attention/self/value/kernel:0')
    value_bias_layer_11 = sess.run('bert/encoder/layer_11/attention/self/value/bias:0')
    bias_layer_11 = sess.run('bert/encoder/layer_11/attention/output/dense/bias:0')
    kernel_layer_11 = sess.run('bert/encoder/layer_11/attention/output/dense/kernel:0')
    
    positional_encoding = sess.run('bert/embeddings/position_embeddings:0')
    
dataset1 = tf.data.TextLineDataset(["poems.txt"])

dataset2 = tf.data.TextLineDataset(["explanations.txt"])

dataset3 = tf.data.Dataset.zip((dataset1, dataset2))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def encode(lang1, lang2):
    lang1 = tokenizer.encode(tf.compat.as_str(lang1.numpy()), add_special_tokens=True)
    lang2 = tokenizer.encode(tf.compat.as_str(lang2.numpy()), add_special_tokens=True)
    
    return lang1, lang2

def tf_encode(pt, en):
    result_pt, result_en = tf.py_function(func = encode, inp = [pt, en], Tout=[tf.int64, tf.int64])
    result_pt.set_shape([None])
    result_en.set_shape([None])
    return result_pt, result_en
def filter_max_length(x, y, max_length=510):
    return tf.logical_and(tf.size(x) <= max_length,
                        tf.size(y) <= max_length)
train_dataset = dataset3.map(tf_encode)

BUFFER_SIZE = 200
BATCH_SIZE = 64
train_dataset = train_dataset.filter(filter_max_length)
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE, 
                                                               padded_shapes=(512,512))


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask
    
  
def scaled_dot_product_attention(q,k,v,mask):
    matmulqk = tf.matmul(q, k, transpose_b = True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmulqk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights  = tf.nn.softmax(scaled_attention_logits, axis = -1)
    output = tf.matmul(attention_weights , v)
    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads,num):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        if num == 0:
            
            self.wq = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(query_kernel_layer_0),
                                            bias_initializer=tf.keras.initializers.Constant(query_bias_layer_0))
            self.wk = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(key_kernel_layer_0),
                                            bias_initializer=tf.keras.initializers.Constant(key_bias_layer_0))
            self.wv = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(value_kernel_layer_0),
                                            bias_initializer=tf.keras.initializers.Constant(value_bias_layer_0))
            self.dense = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(kernel_layer_0),
                                           bias_initializer=tf.keras.initializers.Constant(bias_layer_0))
            
        elif num == 1:
            
            self.wq = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(query_kernel_layer_1),
                                            bias_initializer=tf.keras.initializers.Constant(query_bias_layer_1))
            self.wk = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(key_kernel_layer_1),
                                            bias_initializer=tf.keras.initializers.Constant(key_bias_layer_1))
            self.wv = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(value_kernel_layer_1),
                                            bias_initializer=tf.keras.initializers.Constant(value_bias_layer_1))    
            self.dense = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(query_kernel_layer_1),
                                           bias_initializer=tf.keras.initializers.Constant(query_bias_layer_1))
        elif num == 2:
            
            self.wq = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(query_kernel_layer_2),
                                            bias_initializer=tf.keras.initializers.Constant(query_bias_layer_2))
            self.wk = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(key_kernel_layer_2),
                                            bias_initializer=tf.keras.initializers.Constant(key_bias_layer_2))
            self.wv = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(value_kernel_layer_2),
                                            bias_initializer=tf.keras.initializers.Constant(value_bias_layer_2))    
            self.dense = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(query_kernel_layer_2),
                                           bias_initializer=tf.keras.initializers.Constant(query_bias_layer_2))
        elif num == 3:
            
            self.wq = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(query_kernel_layer_3),
                                            bias_initializer=tf.keras.initializers.Constant(query_bias_layer_3))
            self.wk = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(key_kernel_layer_3),
                                            bias_initializer=tf.keras.initializers.Constant(key_bias_layer_3))
            self.wv = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(value_kernel_layer_3),
                                            bias_initializer=tf.keras.initializers.Constant(value_bias_layer_3))    
            self.dense = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(query_kernel_layer_3),
                                           bias_initializer=tf.keras.initializers.Constant(query_bias_layer_3))
        elif num == 4:
            
            self.wq = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(query_kernel_layer_4),
                                            bias_initializer=tf.keras.initializers.Constant(query_bias_layer_4))
            self.wk = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(key_kernel_layer_4),
                                            bias_initializer=tf.keras.initializers.Constant(key_bias_layer_4))
            self.wv = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(value_kernel_layer_4),
                                            bias_initializer=tf.keras.initializers.Constant(value_bias_layer_4))    
            self.dense = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(query_kernel_layer_4),
                                           bias_initializer=tf.keras.initializers.Constant(query_bias_layer_4))
        elif num == 5:
            
            self.wq = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(query_kernel_layer_5),
                                            bias_initializer=tf.keras.initializers.Constant(query_bias_layer_5))
            self.wk = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(key_kernel_layer_5),
                                            bias_initializer=tf.keras.initializers.Constant(key_bias_layer_5))
            self.wv = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(value_kernel_layer_5),
                                            bias_initializer=tf.keras.initializers.Constant(value_bias_layer_5))    
            self.dense = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(query_kernel_layer_5),
                                           bias_initializer=tf.keras.initializers.Constant(query_bias_layer_5))
        elif num == 6:
            
            self.wq = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(query_kernel_layer_6),
                                            bias_initializer=tf.keras.initializers.Constant(query_bias_layer_6))
            self.wk = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(key_kernel_layer_6),
                                            bias_initializer=tf.keras.initializers.Constant(key_bias_layer_6))
            self.wv = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(value_kernel_layer_6),
                                            bias_initializer=tf.keras.initializers.Constant(value_bias_layer_6))    
            self.dense = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(query_kernel_layer_6),
                                           bias_initializer=tf.keras.initializers.Constant(query_bias_layer_6))
        elif num == 7:
            
            self.wq = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(query_kernel_layer_7),
                                            bias_initializer=tf.keras.initializers.Constant(query_bias_layer_7))
            self.wk = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(key_kernel_layer_7),
                                            bias_initializer=tf.keras.initializers.Constant(key_bias_layer_7))
            self.wv = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(value_kernel_layer_7),
                                            bias_initializer=tf.keras.initializers.Constant(value_bias_layer_7))    
            self.dense = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(query_kernel_layer_7),
                                           bias_initializer=tf.keras.initializers.Constant(query_bias_layer_7))
        elif num == 8:
            
            self.wq = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(query_kernel_layer_8),
                                            bias_initializer=tf.keras.initializers.Constant(query_bias_layer_8))
            self.wk = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(key_kernel_layer_8),
                                            bias_initializer=tf.keras.initializers.Constant(key_bias_layer_8))
            self.wv = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(value_kernel_layer_8),
                                            bias_initializer=tf.keras.initializers.Constant(value_bias_layer_8))    
            self.dense = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(query_kernel_layer_8),
                                           bias_initializer=tf.keras.initializers.Constant(query_bias_layer_8))
        elif num == 9:
            
            self.wq = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(query_kernel_layer_9),
                                            bias_initializer=tf.keras.initializers.Constant(query_bias_layer_9))
            self.wk = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(key_kernel_layer_9),
                                            bias_initializer=tf.keras.initializers.Constant(key_bias_layer_9))
            self.wv = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(value_kernel_layer_9),
                                            bias_initializer=tf.keras.initializers.Constant(value_bias_layer_9))    
            self.dense = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(query_kernel_layer_9),
                                           bias_initializer=tf.keras.initializers.Constant(query_bias_layer_9))
        elif num == 10:
            
            self.wq = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(query_kernel_layer_10),
                                            bias_initializer=tf.keras.initializers.Constant(query_bias_layer_10))
            self.wk = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(key_kernel_layer_10),
                                            bias_initializer=tf.keras.initializers.Constant(key_bias_layer_10))
            self.wv = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(value_kernel_layer_10),
                                            bias_initializer=tf.keras.initializers.Constant(value_bias_layer_10))    
            self.dense = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(query_kernel_layer_10),
                                           bias_initializer=tf.keras.initializers.Constant(query_bias_layer_10))
        else:
            
            self.wq = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(query_kernel_layer_11),
                                            bias_initializer=tf.keras.initializers.Constant(query_bias_layer_11))
            self.wk = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(key_kernel_layer_11),
                                            bias_initializer=tf.keras.initializers.Constant(key_bias_layer_11))
            self.wv = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(value_kernel_layer_11),
                                            bias_initializer=tf.keras.initializers.Constant(value_bias_layer_11))    
            self.dense = tf.keras.layers.Dense(d_model, kernel_initializer = tf.keras.initializers.Constant(query_kernel_layer_11),
                                           bias_initializer=tf.keras.initializers.Constant(query_bias_layer_11))
                
            
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, [batch_size, -1, self.num_heads, self.depth])
        return tf.transpose(x, perm=[0, 2, 1, 3])
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)  
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size) 
        scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation = 'relu'),
        tf.keras.layers.Dense(d_model)])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff,num, rate = 0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, num)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        
    def call(self, x, training, mask):
        attn_output,_ = self.mha(x,x,x,mask)
        attn_output = self.dropout1(attn_output, training)
        out1 = self.layernorm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training = training)
        out2 = self.layernorm2(ffn_output + out1)
        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, num,rate =0.1):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads, num)
        self.mha2 = MultiHeadAttention(d_model, num_heads, num)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon = 1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weighths_block1 = self.mha1(x,x,x,look_ahead_mask)
        attn1 = self.dropout1(attn1, training = training)
        out1 = self.layernorm1(attn1 + x)
        
        attn2, attn_weighths_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training = training)
        out2 = self.layernorm2(attn2 + out1)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training = training)
        out3 = self.layernorm3(ffn_output + out2)
        
        return out3, attn_weighths_block1, attn_weighths_block2


class Encoder(tf.keras.layers.Layer):
    def __init__(self,num_layers, d_model, num_heads, dff, input_vocab_size,rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model, 
        embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix), trainable=True)
        self.pos_encoding = positional_encoding
        print(self.pos_encoding.shape)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, num, rate)
                           for num in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        self.pos = self.pos_encoding[:seq_len,:]
        self.pos = tf.expand_dims(self.pos, 0)
        x += self.pos
        x = self.dropout(x, training = training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        return x
    


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 rate = 0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model, 
        embeddings_initializer = tf.keras.initializers.Constant(embedding_matrix), trainable=True)
        self.pos_encoding = positional_encoding
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff,num, rate)
                           for num in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        self.pos = self.pos_encoding[:seq_len,:]
        self.pos = tf.expand_dims(self.pos, 0)
        x += self.pos
        
        x = self.dropout(x, training = training)
        for i in range (self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        return x, attention_weights
    

class Transformer(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                 target_vocab_size, rate = 0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                               input_vocab_size, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                           target_vocab_size, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    def call(self, inp, tar, training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        
        return final_output, attention_weights


num_layers = 12
d_model = 768
dff = 3072
num_heads = 8
EPOCHS = 2
input_vocab_size = 30522
target_vocab_size = 30522
dropout_rate = 0.1
MAX_LENGTH = 60
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps = 4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.d_model = tf.cast(self.d_model, tf.float32)
    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1 = 0.9, beta_2 = 0.98,
                                     epsilon = 1e-9)
        
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Precision(
    name='train_accuracy')


transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          rate=dropout_rate)

def create_masks(inp, tar):
    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask


def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, 
                                 True, 
                                 enc_padding_mask, 
                                 combined_mask, 
                                 dec_padding_mask)
        loss = loss_function(tar_real, predictions)
        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
        train_loss(loss)
        train_accuracy(tar_real, predictions)
    return predictions
        
for epoch in range(EPOCHS):
    predictions = []
    start = time.time()
    train_loss.reset_states()
    train_accuracy.reset_states()
    for (batch, (inp, tar)) in enumerate(train_dataset):
        predicted = train_step(inp, tar)
        if batch % 50 == 0:
            print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                epoch + 1, batch, train_loss.result(), train_accuracy.result()))
        pred_output = tokenizer.decode(np.vectorize(predicted))
        real_output = tokenizer.decode(np.vectorize(inp))
        recall, precision, rouge = rouge_n_sentence_level(pred_output.split(), real_output.split(), 3)
        print('ROUGE-3-R', recall)
        print('ROUGE-3-P', precision)
        print('ROUGE-3-F', rouge)
    
    print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                train_loss.result(), 
                                                train_accuracy.result()))
    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
    
  
def evaluate(inp_sentence):
    inp_sentence = tokenizer.encode(inp_sentence)
    encoder_input = tf.expand_dims(inp_sentence, 0)
    decoder_input = [30522]
    output = tf.expand_dims(decoder_input, 0)
    
    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
        predictions, attention_weights = transformer(encoder_input, output,False,
                                                     enc_padding_mask,combined_mask, dec_padding_mask)
        
        
    return tf.squeeze(output, axis=0), attention_weights
def translate(sentence):
    result, attention_weights = evaluate(sentence)
    predicted_sentence = tokenizer.decode(result)
    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))
    recall, precision, rouge = rouge_n_sentence_level(predicted_sentence.split(), sentence.split(), 3)
    print('ROUGE-3-R', recall)
    print('ROUGE-3-P', precision)
    print('ROUGE-3-F', rouge)
   
translate("new jersey est parfois calme pendant l' automne , et il est neigeux en avril .")
