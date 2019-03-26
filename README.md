# keras_layers_train








# https://www.jianshu.com/p/556997127319

#简单记录以下keras的自定义层的设置：
主要是三个方法的定义，以我自己的代码为例吧！
首先定义一个类，这个类继承了Keras.engine.topology.Layer。
1.首先是一些必要参数的初始化，这些参数的初始化写在def __init__(self,)中，然后是一些参数的初始化，记得最后要继承Layer中的一些初始化参数。

2.这部分主要是编写一些要更新的参数def build(self,)，如权重等，可以使用 类似self.kernel = self.add_weight(name = '....',shape = [],initializer = 'uniform',trainable = True)的方法来定义一些需要更新的参数变量，也可以使用self.kernel = tf.get_variable和tf.Variable()等来定义需要更新的参数变量。

3.最重要的是def call(self,),这部分代码包含了主要代码的实现，前面两个只是定义了一些初始化的参数以及一些需要更新的参数变量，而真正实现LayerNorm类的作用是在call方法中。可以看到call中的一系列操作是对__init__和build中变量参数的应用。
4.还有一个方法在小编的代码里面没有用到就是compute_output_shape(input_shape)，这个方法的主要作用是如果你的层更改了输入张量的形状，你应该在这里定义形状变化的逻辑，这让Keras能够自动推断各层的形状。显然，在小编需要自定义的LayerNorm中的input_shape和output_shape 是相等的，只是经过了一个norm的过程，并不改变shape。

下面是小编参照Keras中文文档编写的一个LayerNorm，读者可以对比官方文档和这个代码进行自己代码的改写。
--------------------- 
<!-- 作者：牛丸4 
来源：CSDN 
原文：https://blog.csdn.net/baidu_36161077/article/details/84564229 
版权声明：本文为博主原创文章，转载请附上博文链接！ -->




分析Dense层
下面是Keras包中实现的Desne层，我们可以看出这一层是按照官方的标准实现的。
1.初始化
每一个层都要继承基类Layer，这个类在base_layer.py中定义。
在__init__的部分，类对传入的参数进行了检查并对各种赋值参数进行了处理。像初始化、正则化等操作都是使用Keras自带的包进行了包装处理。

2.build()
build()函数首先对输入张量的大小进行了检查，对于Dense层，输入张量的大小为2即(batch_shape, sample_shape)。
然后使用self.add_weight()函数添加该层包含的可学习的参数，对于Dense层其基本操作就是一元线性回归方程y=wx+b，因此定义的两个参数kernel和bias，参数trainable=True是默认的。需要注意的是参数的大小需要我们根据输入与输出的尺寸进行定义，比如输入为n，输出为m，我们需要的参数大小即为(n, m)，偏置大小为m，这是一个矩阵乘法。
底层的所有操作都不需要我们处理，self.add_weight()函数会将各种类型的参数进行分配，Tensorflow帮我们完成自动求导和反向传播。
最后调用self.built = True完成这一层的设置，这一句是一定要有的。也可以使用super(MyLayer, self).build(input_shape)调用父类的函数进行替代。
3 .call()
这个函数式一个网络层最为核心的部分，用来进行这一层对应的运算，其接收上一层传入的张量返回这一层计算完成的张量。
output = K.dot(inputs, self.kernel)这里完成矩阵点乘的操作
output = K.bias_add(output, self.bias, data_format='channels_last')这里完成矩阵加法的操作
output = self.activation(output)这里调用激活函数处理张量
4. compute_output_shape()
compute_output_shape()函数用来输出这一层输出尺寸的大小，尺寸是根据input_shape以及我们定义的output_shape计算的。这个函数在组建Model时会被调用，用来进行前后层张量尺寸的检查。
4. get_config()
get_config()这个函数用来返回这一层的配置以及结构。

作者：洛荷
链接：https://www.jianshu.com/p/556997127319
来源：简书
简书著作权归作者所有，任何形式的转载都请联系作者获得授权并注明出处。
