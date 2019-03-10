import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer
import numpy as np

class L2Normalization(Layer):
    '''
    Performs L2 normalization on the input tensor with a learnable scaling parameter
    as described in the paper "Parsenet: Looking Wider to See Better" (see references)
    and as used in the original SSD model.

    Arguments:
        gamma_init (int): The initial scaling parameter. Defaults to 20 following the
            SSD paper.

    Input shape:
        4D tensor of shape `(batch, channels, height, width)` if `dim_ordering = 'th'`
        or `(batch, height, width, channels)` if `dim_ordering = 'tf'`.

    Returns:
        The scaled tensor. Same shape as the input tensor.
    '''

    def __init__(self, gamma_init=20, **kwargs):
        if K.image_dim_ordering() == 'tf':
            self.axis = 3
        else:
            self.axis = 1
        self.gamma_init = gamma_init
        super(L2Normalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        gamma = self.gamma_init * np.ones((input_shape[self.axis],))
        self.gamma = K.variable(gamma, name='{}_gamma'.format(self.name))
        self.trainable_weights = [self.gamma]
        super(L2Normalization, self).build(input_shape)

    def call(self, x, mask=None):
        output = K.l2_normalize(x, self.axis)
        output *= self.gamma
        return output




# 首先说一下这个层是用来做什么的。就是对于每一个通道进行归一化，不过通道使用的是不同的归一化参数，也就是说这个参数是需要进行学习的，因此需要通过 自定义层来完成。

# 在keras中，每个层都是对象，真的，可以通过dir(Layer对象)来查看具有哪些属性。

# 具体说来：

# __init__（）：用来进行初始化的（这不是废话么），gamma就是要学习的参数。

# bulid（）：是用来创建这层的权重向量的，也就是要学习的参数“壳”。

# 33：设置该层的input_spec，这个是通过InputSpec函数来实现。

# 34：分配权重“壳”的实际空间大小

# 35,：由于底层使用的Tensorflow来进行实现的，因此这里使用Tensorflow中的variable来保存变量。

# 36：根据keras官网的要求，可训练的权重是要添加至trainable_weights列表中的

# 37：我不想说了，官网给的实例都是这么做的。

# call（）：用来进行具体实现操作的。

# 40：沿着指定的轴对输入数据进行L2正则化

# 41：使用学习的gamma来对正则化后的数据进行加权

# 42：将最后的数据最为该层的返回值，这里由于是和输入形式相同的，因此就没有了compute_output_shape函数，如果输入和输出的形式不同，就需要进行输入的调整。

# 就这样子吧。