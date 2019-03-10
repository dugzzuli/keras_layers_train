import keras.backend as K
import tensorflow as tf 
from keras.engine.topology import Layer
import numpy as np
class LayerNorm(Layer):
  def __init__(self,dims,axis = -1,offset = True,scale = True,eps = 1e-6,dtype = tf.float32,scope = None,**kwargs):
    self.offset = offset
    self.dims = dims
    self.scale = scale
    self.dtype = dtype
    self.axis = axis
    self.eps = eps
    super(LayerNorm,self).__init__(**kwargs)  ## 继承Layer中的初始化参数    
  def build(self,input_shape):
    ## create a trainable weight variable for this layer
    self.offset_var = 0
    if self.offset:
    ## 这里的name参数是不可缺的，但是如果name的字符串是固定的，代码会报错，原来的tensorflow的代码中的name是获得该节点##的名字，但是在keras里面直接获取节点的名字不太方便，所以这里就直接使用默认的参数，name = self.name + '_offset'了
      self.offset_var = tf.get_variable(self.name + 'offset',shape = [self.dims],initializer = tf.zeros_initializer(),dtype = self.dtype)
      
    scale_var = 1
    if self.scale:
      self.scale_var = tf.get_variable(self.name + '_scale',shape = [self.dims],initializer = tf.zeros_initializer(),dtype = self.dtype)
      
#     self.kernel = self.add_weight(name = 'kernel',
#                                   shape = (input_shape[1],self.output_dim),
#                                   initializer = 'uniform'
#                                   trainable = True
#                                  )
    super(LayerNorm,self).build(input_shape)
    
  def call(self,x):
    mean = tf.reduce_mean(x,axis = self.axis,keep_dims = True)
    inverse_stddev = tf.rsqrt(tf.reduce_mean(tf.square(x - mean),axis = self.axis,keep_dims = True) + self.eps)
    normed = (x - mean) * inverse_stddev
    return normed * self.scale_var + self.offset_var


import keras
from keras.layers import Input
from keras.models import Sequential,Model
filters = 128
inputs = Input(shape = (128,1))
output = LayerNorm(dims = filters)(inputs)
model = Model(input = inputs,output = output)
model.summary()



# --------------------- 
# 作者：牛丸4 
# 来源：CSDN 
# 原文：https://blog.csdn.net/baidu_36161077/article/details/84564229 
# 版权声明：本文为博主原创文章，转载请附上博文链接！