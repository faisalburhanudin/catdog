from keras.layers import Input, Conv2D, Flatten, Dense, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model
from catdog import resnet

# model parameter
shape = (32, 32, 3)  # cifar shape
classes = 10  # cifar10 class

# MODEL
inputs = Input(shape=shape)
x = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', name='conv1')(inputs)
x = BatchNormalization(axis=3, name="batch_normalization")(x)
x = Activation('relu')(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)

x = resnet.conv_block(x, 3, [16, 16, 64], stage=2, block='a', strides=(1, 1))

x = Flatten()(x)
x = Dense(classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=x)
model.summary()
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
