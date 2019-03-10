# keras_layers_train

#对于简单的定制操作，我们或许可以通过使用layers.core.Lambda层来完成。但对于任何具有可训练权重的定制层，你应该自己来实现。@

build(input_shape)：这是定义权重的方法，可训练的权应该在这里被加入列表self.trainable_weights中。其他的属性还包括self.non_trainabe_weights（列表）和self.updates（需要更新的形如（tensor, new_tensor）的tuple的列表）。你可以参考BatchNormalization层的实现来学习如何使用上面两个属性。这个方法必须设置self.built = True，可通过调用super([layer],self).build()实现
call(x)：这是定义层功能的方法，除非你希望你写的层支持masking，否则你只需要关心call的第一个参数：输入张量
compute_output_shape(input_shape)：如果你的层修改了输入数据的shape，你应该在这里指定shape变化的方法，这个函数使得Keras可以做自动shape推断
