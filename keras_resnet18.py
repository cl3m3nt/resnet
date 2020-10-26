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
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=1000,
             **kwargs):
    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)



    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

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
        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
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
    '''
    if weights == 'imagenet':
        if include_top:
            weights_path = keras_utils.get_file(
                'resnet18_imagenet_1000.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                md5_hash='64da73012bb70e16c901316c201d9803')
        else:
            weights_path = keras_utils.get_file(
                'resnet18_imagenet_1000_no_top.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                md5_hash='318e3ac0cd98d51e917526c9f62f0b50')
        model.load_weights(weights_path)
    '''
    return model


WEIGHTS_PATH = ('https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet18_imagenet_1000.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet18_imagenet_1000_no_top.h5')


def sample_batch():
    img_path = '/Users/clement/mycar4/data/images'
    img_list = os.listdir(img_path)
    img_batch = []

    for i in range(0,32):
        img = cv2.imread(os.path.join(img_path,img_list[i]))
        img = img/255.0
        img_batch.append(img)

    batch = np.array(img_batch)
    return batch 


def transfer_resnet18(input_shape):
    resnet18 = ResNet18(include_top=False,input_shape=input_shape,backend=backend,layers=layers,models=models,utils=utils)
    resnet18_preprocess = tf.keras.applications.resnet.preprocess_input

    inputs = tf.keras.Input(shape=input_shape)
    x = resnet18_preprocess(inputs)
    x = resnet18(inputs,training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(10,activation='softmax')(x)
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
my_resnet18_10 = resnet18_n_class((32,32,3),10)
my_resnet18_10 = compile(my_resnet18_10)

# Resnet18_100 from transfer resnet18_n_class
my_resnet18_100 = resnet18_n_class((32,32,3),100)
my_resnet18_100 = compile(my_resnet18_100)

# Resnet18_100 from transfer resnet18_n_class
my_resnet18_100_coarse = resnet18_n_class((32,32,3),20)
my_resnet18_100_coarse = compile(my_resnet18_100_coarse)


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
(x_train_100_coarse,y_train_100_coarse),(x_test_100_coarse,y_test_100_coarse) = cifar100.load_data(label_mode="coarse")
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
