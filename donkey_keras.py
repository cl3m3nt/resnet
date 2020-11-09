import tensorflow as tf
import tensorflow.keras.datasets as datasets
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,GlobalAveragePooling2D,Dense,Dropout
from resnet18_keras import identity_block,conv_block,ResNet18
from abc import ABC, abstractmethod
import numpy as np
import math

# ResNet18 pre-requesite parameters
backend = tf.compat.v1.keras.backend
layers = tf.keras.layers
models = tf.keras.models
utils = tf.keras.utils

ONE_BYTE_SCALE = 1


## resnet18 Linear + Categorical

def resnet18_default_n_linear(num_outputs, input_shape=(120,60,3)):

    # Instantiate a ResNet18 model
    resnet18 = ResNet18(include_top=False,weights='cifar100_coarse',input_shape=input_shape,backend=backend,layers=layers,models=models,utils=utils)
    for layer in resnet18.layers: 
        layer.trainable=False     # Freezing resnet18 layers
    resnet18_preprocess = tf.keras.applications.resnet.preprocess_input

    # Transfer learning with Resnet18
    drop = 0.2
    img_in = Input(shape=input_shape,name='img_in')
    x = resnet18_preprocess(img_in)
    x = resnet18(img_in,training=True)
    x = GlobalAveragePooling2D()(x) # Flattening
    # Classifier
    x = Dense(128,activation='relu',name='dense_1')(x) 
    x = Dropout(drop)(x) 
    x = Dense(64,activation='relu',name='dense_2')(x)
    x = Dropout(drop)(x)

    outputs = []
    for i in range(num_outputs):
        outputs.append(
            Dense(1, activation='linear', name='n_outputs' + str(i))(x))

    model = Model(inputs=[img_in],outputs=outputs)

    return model


def resnet18_default_categorical(input_shape=(120, 60, 3)):
     # Instantiate a ResNet18 model
    resnet18 = ResNet18(include_top=False,weights='cifar100_coarse',input_shape=input_shape,backend=backend,layers=layers,models=models,utils=utils)
    for layer in resnet18.layers: 
        layer.trainable=False     # Freezing resnet18 layers
    resnet18_preprocess = tf.keras.applications.resnet.preprocess_input

    # Transfer learning with Resnet18
    drop = 0.2
    img_in = Input(shape=input_shape,name='img_in')
    x = resnet18_preprocess(img_in)
    x = resnet18(img_in,training=True)
    x = GlobalAveragePooling2D()(x) # Flattening
    # Classifier
    x = Dense(128,activation='relu',name='dense_1')(x) 
    x = Dropout(drop)(x) 
    x = Dense(64,activation='relu',name='dense_2')(x)
    x = Dropout(drop)(x)

    # Categorical output of the angle into 15 bins
    angle_out = Dense(20, activation='softmax', name='angle_out')(x)
    # categorical output of throttle into 20 bins
    throttle_out = Dense(20, activation='softmax', name='throttle_out')(x)

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])
    return model


## Utils function
def linear_unbin(arr, N=15, offset=-1, R=2.0):
    '''
    preform inverse linear_bin, taking
    one hot encoded arr, and get max value
    rescale given R range and offset
    '''
    b = np.argmax(arr)
    a = b * (R / (N + offset)) + offset
    return a

def normalize_image(img_arr_uint):
    """
    Convert uint8 numpy image array into [0,1] float image array
    :param img_arr_uint:    [0,255]uint8 numpy image array
    :return:                [0,1] float32 numpy image array
    """
    return img_arr_uint.astype(np.float32) * ONE_BYTE_SCALE


STEERING_MIN = -1.
STEERING_MAX = 1.
# Scale throttle ~ 0.5 - 1.0 depending on the steering angle
EXP_SCALING_FACTOR = 0.5
DAMPENING = 0.05

def clamp(n, min, max):
    if n < min:
        return min
    if n > max:
        return max
    return n

def _steering(input_value):
    input_value = clamp(input_value, STEERING_MIN, STEERING_MAX)
    return ((input_value - STEERING_MIN) / (STEERING_MAX - STEERING_MIN))


def throttle(input_value):
    magnitude = _steering(input_value)
    decay = math.exp(magnitude * EXP_SCALING_FACTOR)
    dampening = DAMPENING * magnitude
    return ((1 / decay) - dampening)


## Compile Function

def compile_linear(model):
    model.compile(optimizer='adam', loss='mse',metrics='mse')
    return model


def compile_categorical(model):
    # sparse_categorical_crossentropy vs categorical_crossentropy depending on data
    model.compile(optimizer='adam', metrics=['accuracy'],
                           loss={'angle_out': 'sparse_categorical_crossentropy',
                                 'throttle_out': 'sparse_categorical_crossentropy'},
                           loss_weights={'angle_out': 0.5, 'throttle_out': 0.5})
    # to check y_train format from donkey generated Data
    return model


## Resnet18 Classes Linear + Categorical + Pilot

class KerasPilot(ABC):

    def __init__(self):
        self.model = None
        self.optimizer = "adam"
 
    def load(self, model_path):
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def load_weights(self, model_path, by_name=True):
        self.model.load_weights(model_path, by_name=by_name)

    def shutdown(self):
        pass

    def compile(self):
        pass

    def set_optimizer(self, optimizer_type, rate, decay):
        if optimizer_type == "adam":
            self.model.optimizer = tf.keras.optimizers.Adam(lr=rate, decay=decay)
        elif optimizer_type == "sgd":
            self.model.optimizer = tf.keras.optimizers.SGD(lr=rate, decay=decay)
        elif optimizer_type == "rmsprop":
            self.model.optimizer = tf.keras.optimizers.RMSprop(lr=rate, decay=decay)
        else:
            raise Exception("unknown optimizer type: %s" % optimizer_type)

    def get_input_shape(self):
        assert self.model is not None, "Need to load model first"
        return self.model.inputs[0].shape

    def run(self, img_arr, other_arr=None):

        norm_arr = normalize_image(img_arr)
        return self.inference(norm_arr, other_arr)

    @abstractmethod
    def inference(self, img_arr, other_arr):
        pass

    def train(self, x_train, y_train,val_data, 
              saved_model_path, epochs=100, steps=100, train_split=0.8,
              verbose=1, min_delta=.0005, patience=5, use_early_stop=True):
        
        """
        train_gen: generator that yields an array of images an array of 
        
        """

        # checkpoint to save model after each epoch
        save_best = tf.keras.callbacks.ModelCheckpoint(saved_model_path, 
                                                    monitor='val_loss', 
                                                    verbose=verbose, 
                                                    save_best_only=True, 
                                                    mode='min')
        
        # stop training if the validation error stops improving.
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                   min_delta=min_delta, 
                                                   patience=patience, 
                                                   verbose=verbose, 
                                                   mode='auto')
        
        callbacks_list = [save_best]

        if use_early_stop:
            callbacks_list.append(early_stop)
        #hist = self.model.fit_generator(
        hist = self.model.fit(
                        x_train,y_train, 
                        steps_per_epoch=steps, 
                        epochs=epochs, 
                        verbose=1, 
                        validation_data=val_data,
                        callbacks=callbacks_list, 
                        validation_steps=steps*(1.0 - train_split))
        return hist


class Resnet18LinearKeras(KerasPilot):
    def __init__(self, num_outputs=1, input_shape=(120, 160, 3)):
        super().__init__()
        self.model = resnet18_default_n_linear(num_outputs, input_shape)
        self.optimizer = 'adam'

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss='mse',metrics='mse')

    def inference(self, img_arr, other_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        steering = outputs[0]
        return steering[0] , throttle(steering[0])

    
class Resnet18CategoricalKeras(KerasPilot):
    def __init__(self, input_shape=(120, 160, 3), throttle_range=0.5):
        super().__init__()
        self.model = resnet18_default_categorical(input_shape)
        self.optimizer = 'adam'
        self.compile()
        self.throttle_range = throttle_range

    def compile(self):
        self.model.compile(optimizer=self.optimizer, metrics=['accuracy'],
                           loss={'angle_out': 'sparse_categorical_crossentropy',
                                 'throttle_out': 'sparse_categorical_crossentropy'},
                           loss_weights={'angle_out': 0.5, 'throttle_out': 0.5})
        
    def inference(self, img_arr, other_arr):
        if img_arr is None:
            print('no image')
            return 0.0, 0.0

        img_arr = img_arr.reshape((1,) + img_arr.shape)
        angle_binned, throttle_binned = self.model.predict(img_arr)
        N = len(throttle_binned[0])
        throttle = linear_unbin(throttle_binned, N=N,
                                         offset=0.0, R=self.throttle_range)
        angle = linear_unbin(angle_binned)
        return angle, throttle



# CIFAR100 Data
cifar100_coarse = datasets.cifar100
(x_train,y_train),(x_test,y_test) = cifar100_coarse.load_data(label_mode="coarse")
x_train = x_train/255.0
x_test = x_test/255.0


## Test Transfer learning Resnet18 Linear + Categorical with CIFAR100

resnet18_linear = resnet18_default_n_linear(2,(32,32,3))
resnet18_linear = compile_linear(resnet18_linear)
resnet18_linear.summary()
notop_weights = resnet18_linear.weights
print(notop_weights[0][0][0][0])

resnet18_categorical = resnet18_default_categorical((32,32,3))
resnet18_categorical = compile_categorical(resnet18_categorical)
resnet18_categorical.summary()
notop_weights = resnet18_categorical.weights
print(notop_weights[0][0][0][0])


# Training Linear Resnet18 with CIFAR100 
history_lin = resnet18_linear.fit(x_train,y_train,
                          validation_data = (x_test,y_test),
                          epochs=1
)

# Training Categorical Resnet18 with CIFAR100 
history_cat = resnet18_categorical.fit(x_train,y_train,
                          validation_data = (x_test,y_test),
                          epochs=1
                        )


## Test KerasResnet18Linear Class: 
r18_lin = Resnet18LinearKeras()
r18_lin.compile()
r18_lin.model.summary()
history_r18_lin = r18_lin.model.fit(x_train,y_train,
                          validation_data = (x_test,y_test),
                          epochs=1
)
test_img = x_test[1]
r18_lin.inference(test_img,None)


## Test KerasResnet18Categorical Class: 
r18_cat = Resnet18CategoricalKeras()
r18_cat.compile()
r18_cat.model.summary()
history_r18_cat = r18_cat.model.fit(x_train,y_train,
                          validation_data = (x_test,y_test),
                          epochs=1
)
test_img = x_test[1]
r18_cat.inference(test_img,None)


## Test Train + Save model from KerasPilot abstract Class function

# Linear
def test_r18_linear_train():

    r18_lin = Resnet18LinearKeras()
    r18_lin.compile()
    r18_lin.model.summary()
    history_r18_lin = r18_lin.train(x_train,y_train,
                            (x_test,y_test),
                            'mymodelLinear.h5',epochs=1,steps=1563
    )
    return history_r18_lin

def test_r18_linear_inference(r18KerasLinear,test_image):
    inference = r18KerasLinear.inference(test_image,None)
    return inference

inference = test_r18_linear_inference(r18_lin,x_test[1])
print(inference)

def test_r18_linear_predict():
    model = tf.keras.models.load_model('mymodelLinear.h5')
    model.summary()
    test_img = x_test
    predictions = model.predict(test_img,None)
    return predictions

test_r18_linear_predict()

# Categorical
def test_r18_cat_train():
    r18_cat = Resnet18CategoricalKeras()
    r18_cat.compile()
    r18_cat.model.summary()
    history_r18_cat = r18_lin.train(x_train,y_train,
                            (x_test,y_test),
                            'mymodelCategorical.h5',epochs=1,steps=1563
    )
    return history_r18_cat

def test_r18_cat_inference(r18KerasCat,test_image):
    inference = r18KerasCat.inference(test_image,None)
    return inference

inference = test_r18_cat_inference(r18_cat,x_test[1])
print(inference)

def test_r18_cat_predict():
    model = tf.keras.models.load_model('mymodelCategorical.h5')
    model.summary()
    test_img = x_test
    predictions = model.predict(test_img)
    return predictions

test_r18_cat_predict()







############ DEPRECATED ############
def compile(model:Model)->Model:
    """function to compile a model for n-class classification

    Args:
        model (Model): the model to be compiled

    Returns:
        Model: a compiled model with optimizer,loss,metrics set
    """
    model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
    )
    return model
    

# resnet18_n_class classifier definition
# This is the base example to create an n-class classifier with ResNet18 Transfer Learning
def resnet18_n_class(input_shape:tuple,n_class:int)->Model:
    """resnet18_n_class function build a custom model
       based on resnet18 transfer learning, adding a Dense 
       classification layer
    Args:
        input_shape (tuple): the shape of the input data
        n_class (int): the number of class of the classifier

    Returns:
        Model: the custom model based on resnet18 transfer learning
        with extra classification layer
    """

    # Instantiate a ResNet18 model
    resnet18 = ResNet18(include_top=False,weights='cifar100_coarse',input_shape=input_shape,backend=backend,layers=layers,models=models,utils=utils)
    for layer in resnet18.layers: 
        layer.trainable=False     # Freezing resnet18 layers
    resnet18_preprocess = tf.keras.applications.resnet.preprocess_input

    # Transfer learning
    inputs = tf.keras.Input(shape=input_shape)
    x = resnet18_preprocess(inputs)
    x = resnet18(inputs,training=True)
    x = GlobalAveragePooling2D()(x) # Flattening
    x = Dense(128,activation='relu')(x) # Classifier
    x = Dropout(0.4)(x) # Drop to minimize overfitting
    x = Dense(128,activation='relu')(x) # Classifier
    x = Dropout(0.2)(x) # Drop to minimize overfitting
    outputs = Dense(n_class,activation='softmax')(x)

    model = Model(inputs,outputs)
    model.summary()

    return model