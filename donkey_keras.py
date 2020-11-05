import tensorflow as tf
import tensorflow.keras.datasets as datasets
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,GlobalAveragePooling2D,Dense,Dropout
from resnet18_keras import identity_block,conv_block,ResNet18

# ResNet18 pre-requesite parameters
backend = tf.compat.v1.keras.backend
layers = tf.keras.layers
models = tf.keras.models
utils = tf.keras.utils


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





# Transfer Learning
# Resnet18 Linear + Categorical
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


# CIFAR100 Data
cifar100_coarse = datasets.cifar100
(x_train_100_coarse,y_train_100_coarse),(x_test_100_coarse,y_test_100_coarse) = cifar100_coarse.load_data(label_mode="coarse")
x_train_100_coarse = x_train_100_coarse/255.0
x_test_100_coarse = x_test_100_coarse/255.0

# Training Linear Resnet18 with CIFAR100 
history_lin = resnet18_linear.fit(x_train_100_coarse,y_train_100_coarse,
                          validation_data = (x_test_100_coarse,y_test_100_coarse),
                          epochs=5

# Training Categorical Resnet18 with CIFAR100 
history_cat = resnet18_categorical.fit(x_train_100_coarse,y_train_100_coarse,
                          validation_data = (x_test_100_coarse,y_test_100_coarse),
                          epochs=5
                        )