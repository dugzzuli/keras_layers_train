import keras
import numpy as np
from keras import backend as K
from keras.layers.core import Layer


from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model


class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


# create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)    # randomize the data
Y = 0.5 * X


# This returns a tensor
inputs = Input(shape=(1,))

# a layer instance is callable on a tensor, and returns a tensor
predictions= MyLayer(1,name="dug")(inputs)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
# choose loss function and optimizing method
model.compile(loss='mse', optimizer='sgd')

model.fit(X, Y,epochs=100)  # starts training

t=model.get_layer("dug")
print(t)
print(t.get_weights())


import keras
import numpy as np
from keras import backend as K
from keras.layers.core import Layer


from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model


class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


# create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X)    # randomize the data
Y = 0.5 * X


# This returns a tensor
inputs = Input(shape=(1,))

# a layer instance is callable on a tensor, and returns a tensor
predictions= MyLayer(1,name="dug")(inputs)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
# choose loss function and optimizing method
model.compile(loss='mse', optimizer='sgd')

model.fit(X, Y,epochs=100)  # starts training

t=model.get_layer("dug")
print(t)
print(t.get_weights())


