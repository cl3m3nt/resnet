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
    # resnet18
    resnet18 = ResNet18(include_top=False,weights='cifar100_coarse',input_shape=input_shape,backend=backend,layers=layers,models=models,utils=utils)
    # freezing resnet18 layers
    for layer in resnet18.layers:
        layer.trainable=False
    # preprocess input for resnet
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

## Get Data: CIFAR100 on 20 x Super Classes = coarse
# CIFAR100 coarse from tf.keras.datasets 
cifar100_coarse = datasets.cifar100
(x_train_100_coarse,y_train_100_coarse),(x_test_100_coarse,y_test_100_coarse) = cifar100_coarse.load_data(label_mode="coarse")
x_train_100_coarse = x_train_100_coarse/255.0
x_test_100_coarse = x_test_100_coarse/255.0

## Define Models
# Resnet18 from keras local implementation
resnet18 = ResNet18(include_top=True,weights=None,input_shape=(32,32,3),backend=backend,layers=layers,models=models,utils=utils)
resnet18 = compile(resnet18)
resnet18.summary()

top_weights = resnet18.weights
top_weights[0][0][0][0]

# Resnet18 from keras local implementation
notop_resnet18 = ResNet18(include_top=False,weights=None,input_shape=(32,32,3),backend=backend,layers=layers,models=models,utils=utils)
notop_resnet18 = compile(notop_resnet18)
notop_resnet18.summary()

notop_weights = notop_resnet18.weights
top_weights[0][0][0][0]



# Resnet18 from keras local implementation
resnet18 = ResNet18(include_top=True,input_shape=(32,32,3),backend=backend,layers=layers,models=models,utils=utils)
resnet18 = compile(resnet18)
resnet18.summary()

top_weights = resnet18.weights
top_weights[0][0][0][0]

# Resnet18 from keras local implementation
notop_resnet18 = ResNet18(include_top=False,input_shape=(32,32,3),backend=backend,layers=layers,models=models,utils=utils)
notop_resnet18 = compile(notop_resnet18)
notop_resnet18.summary()

notop_weights = notop_resnet18.weights
top_weights[0][0][0][0]





# ResNet18 Transfer to 10 x classes Classifier
resnet18_10c = resnet18_n_class((32,32,3),10)
resnet18_10c = compile(resnet18_10c)

notop_weights = resnet18_10c.weights
top_weights[0][0][0][0]

# ResNet18 Transfer to 20 x classes Classifier
resnet18_20c = resnet18_n_class((32,32,3),20)
resnet18_20c = compile(resnet18_20c)

notop_weights = resnet18_20c.weights
top_weights[0][0][0][0]

# ResNet18 Transfer to 100 x classes Classifier
resnet18_100c = resnet18_n_class((32,32,3),100)
resnet18_100c = compile(resnet18_100c)

notop_weights = resnet18_100c.weights
top_weights[0][0][0][0]

## Training

# Training Local Resnet with CIFAR100 from tf.keras.datasets
history = resnet18_20c.fit(x_train_100_coarse,y_train_100_coarse,
                          validation_data = (x_test_100_coarse,y_test_100_coarse),
                          epochs=1
                        )
