import tensorflow as tf
from tensorflow.keras.layers import ZeroPadding2D, Input, GlobalAveragePooling2D,GlobalMaxPooling2D,Dense
from tensorflow.keras.layers import Convolution2D,MaxPooling2D,BatchNormalization
from tensorflow.keras.layers import Activation,Dropout,Flatten
from tensorflow.keras.models import Model,Sequential
from keras_applications.imagenet_utils import _obtain_input_shape, get_submodules_from_kwargs
import cv2
import numpy as np
import os
import tensorflow.keras.datasets as datasets
import tensorflow_datasets as tfds

WEIGHTS_PATH = 'https://raw.githubusercontent.com/cl3m3nt/resnet/master/resnet18_cifar100_top.h5'
WEIGHTS_PATH_NO_TOP = 'https://raw.githubusercontent.com/cl3m3nt/resnet/master/resnet18_cifar100_no_top.h5'

def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3

    else: 
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block +'_branch'

    x = Convolution2D(filters1,(1,1),
               kernel_initializer='he_normal',
               name = conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis,name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(filters2, kernel_size,
               padding='same',
               kernel_initializer='he_normal',
               name = conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis,name=bn_name_base+'2b')(x)
    x = Activation('relu')(x)

    x = tf.keras.layers.add([x,input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor,kernel_size,filters,stage,block,strides=(2,2)):

    filters1, filters2 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(filters1,(1,1),strides=strides,
               kernel_initializer='he_normal',
               name = conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(bn_axis,name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)


    x = Convolution2D(filters2, kernel_size,
               padding='same',
               kernel_initializer='he_normal',
               name = conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis,name=bn_name_base+'2b')(x)
    x = Activation('relu')(x)

    
    shortcut = Convolution2D(filters2,(1,1),strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base+'1')(input_tensor)
    
    shortcut = BatchNormalization(
        axis=bn_axis,name=bn_name_base+'1')(shortcut)
    
    x = tf.keras.layers.add([x,shortcut])
    x = Activation('relu')(x)
    return x

# Those are mandatory for ResNet function to work   
backend = tf.compat.v1.keras.backend
layers = tf.keras.layers
models = tf.keras.models
utils = tf.keras.utils

# ResnNet18
def ResNet18(include_top=True,
             weights='cifar100_coarse',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=20,
             **kwargs):
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)



    if not (weights in {'cifar100_coarse', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `cifar100_coarse` '
                         '(pre-training on cifar100 coarse (super) classes), '
                         'or the path to the weights file to be loaded.')

    if weights == 'cifar100_coarse' and include_top and classes != 20:
        raise ValueError('If using `weights` as `"cifar100_coarse"` with `include_top`'
                         ' as true, `classes` should be 20')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = ZeroPadding2D(padding=(3,3),name='conv1_pad')(img_input)
    x = Convolution2D(64,(7,7),
                      strides=(2,2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = BatchNormalization(axis=bn_axis,name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1,1),name='pool1_pad')(x)
    x = MaxPooling2D((3,3),strides=(2,2))(x)

    x = identity_block(x,3,[64,64],stage=2,block='a')
    x = identity_block(x,3,[64,64],stage=2,block='b')

    x = conv_block(x,3,[128,128],stage=3,block='a')
    x = identity_block(x,3,[128,128],stage=3,block='b')

    x = conv_block(x,3,[256,256],stage=4,block='a')
    x = identity_block(x,3,[256,256],stage=4,block='b')

    x = conv_block(x,3,[512,512],stage=5,block='a')
    x = identity_block(x,3,[512,512],stage=5,block='b')


    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = layers.Dense(classes, activation='softmax', name='fc20')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
        '''
        else:
            warnings.warn('The output shape of `ResNet18(include_top=False)` '
                          'has been changed since Keras 2.2.0.')
        '''
    

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = keras_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='resnet18')

    # Load weights.
    if weights == 'cifar100_coarse':
        if include_top:
            weights_path = keras_utils.get_file(
                'resnet18_cifar100_top.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='e0798dd90ac7e0498cbdea853bd3ed7f')
        else:
            weights_path = keras_utils.get_file(
                'resnet18_cifar100_no_top.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='bfeace78cec55f2b0401c1f41c81e1dd')
        model.load_weights(weights_path)

  
    return model


def transfer_resnet18(input_shape):
    resnet18_backbone = ResNet18(include_top=False,input_shape=input_shape,backend=backend,layers=layers,models=models,utils=utils)
    resnet18_preprocess = tf.keras.applications.resnet.preprocess_input

    inputs = tf.keras.Input(shape=input_shape)
    x = resnet18_preprocess(inputs)
    x = resnet18_backbone(inputs,training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = x
    transfered_model = tf.keras.Model(inputs,outputs)
    transfered_model.summary()
    return transfered_model


def resnet18_n_class(input_shape,n_class):
    resnet18 = ResNet18(include_top=False,input_shape=input_shape,backend=backend,layers=layers,models=models,utils=utils)
    resnet18_preprocess = tf.keras.applications.resnet.preprocess_input

    inputs = tf.keras.Input(shape=input_shape)
    x = resnet18_preprocess(inputs)
    x = resnet18(inputs,training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # Extra classification layer with 128 Neurons
    x = tf.keras.layers.Dense(128,activation='relu')(x)
    # Adding Dropout to avoid from overfitting
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(n_class,activation='softmax')(x)
    transfered_model = tf.keras.Model(inputs,outputs)
    transfered_model.summary()
    return transfered_model

def compile(model):
    model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )
    return model

# Resnet18 from Pytorch applications
import torchvision.models as tvmodels
resnet18tv = tvmodels.resnet18()
resnet18tv

# Resnet18 from keras local implementation
my_resnet18 = ResNet18(include_top=False,input_shape=(32,32,3),backend=backend,layers=layers,models=models,utils=utils)
my_resnet18 = compile(my_resnet18)

# Resnet18_10 from transfer resnet18_n_class
resnet18_10c = resnet18_n_class((32,32,3),10)
resnet18_10c = compile(resnet18_10c)

# Resnet18_20 from transfer resnet18_n_class
resnet18_20c = resnet18_n_class((32,32,3),20)
resnet18_20c = compile(resnet18_20c)

# Resnet18_100 from transfer resnet18_n_class
resnet18_100 = resnet18_n_class((32,32,3),100)
resnet18_100 = compile(resnet18_100)

# CIFAR10 from tf.keras.datasets
cifar10 = datasets.cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_train = x_train/255.0
x_test = x_test/255.0

# CIFAR10 from tfds
cifar10_tfds = tfds.load('cifar10',as_supervised=True)
data_train = cifar10_tfds['train']
data_validation = cifar10_tfds['test']
data_train_ds = data_train.batch(32)
data_validation_ds = data_validation.batch(32)

# CIFAR100 from tf.keras.datasets
cifar100 = datasets.cifar100
(x_train_100,y_train_100),(x_test_100,y_test_100) = cifar100.load_data()
x_train_100 = x_train_100/255.0
x_test_100 = x_test_100/255.0

# CIFAR100 coarse from tf.keras.datasets 
cifar100_coarse = datasets.cifar100
(x_train_100_coarse,y_train_100_coarse),(x_test_100_coarse,y_test_100_coarse) = cifar100_coarse.load_data(label_mode="coarse")
x_train_100_coarse = x_train_100_coarse/255.0
x_test_100_coarse = x_test_100_coarse/255.0


# CIFAR100 from tfds
cifar100_tfds = tfds.load('cifar100',as_supervised=True)
data_train_100 = cifar100_tfds['train']
data_validation_100 = cifar100_tfds['test']
data_train_ds_100 = data_train_100.batch(32)
data_validation_ds_100 = data_validation._100batch(32)


# Training Local Resnet with CIFAR10 from tf.keras.datasets
history = my_resnet18_10.fit(x_train,y_train,
                          validation_data = (x_test,y_test),
                          epochs=1
                        )

# Training Local Resnet with CIFAR10 from tfds
history = my_resnet18_10.fit(data_train_ds,
                          validation_data = data_validation_ds,
                          epochs=1
                        )

# Training Local Resnet with CIFAR100 from tf.keras.datasets
history = my_resnet18_100.fit(x_train_100,y_train_100,
                          validation_data = (x_test_100,y_test_100),
                          epochs=2
                        )

# Training Local Resnet with CIFAR100 from tf.keras.datasets
history = my_resnet18_100_coarse.fit(x_train_100_coarse,y_train_100_coarse,
                          validation_data = (x_test_100_coarse,y_test_100_coarse),
                          epochs=2
                        )

# Training Local Resnet with CIFAR100 from tfds
history = my_resnet18_100.fit(data_train_ds_100,
                          validation_data = data_validation_ds_100,
                          epochs=2
                        )


input_shape = (32,32,3)
my_resnet18 = ResNet18(include_top=True,weights=None,classes=20,input_shape=input_shape,backend=backend,layers=layers,models=models,utils=utils)
my_resnet18 = compile(my_resnet18)
my_resnet18.summary()
