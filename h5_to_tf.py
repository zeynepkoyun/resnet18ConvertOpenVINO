#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from __future__ import print_function
from keras import backend as K
from tensorflow.python.saved_model import builder as saved_model_builder
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.keras import models

IMG_SIZE = (64, 64)
NUM_CLASSES = 10
IMG_SHAPE = IMG_SIZE + (3,)

def residual_conv_block(filters, stage, block, strides=(1, 1), cut='pre'):
    def layer(input_tensor):
        x = layers.BatchNormalization(epsilon=2e-5)(input_tensor)
        x = layers.Activation('relu')(x)

        # defining shortcut connection
        if cut == 'pre':
            shortcut = input_tensor
        elif cut == 'post':
            shortcut = layers.Conv2D(filters, (1, 1), strides=strides, kernel_initializer='he_uniform',
                                     use_bias=False)(x)

        # continue with convolution layers
        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(filters, (3, 3), strides=strides, kernel_initializer='he_uniform', use_bias=False)(x)

        x = layers.BatchNormalization(epsilon=2e-5)(x)
        x = layers.Activation('relu')(x)
        x = layers.ZeroPadding2D(padding=(1, 1))(x)
        x = layers.Conv2D(filters, (3, 3), kernel_initializer='he_uniform', use_bias=False)(x)

        # add residual connection
        x = layers.Add()([x, shortcut])
        return x

    return layer

def ResNet18(input_shape=None):
    """Instantiates the ResNet18 architecture."""
    img_input = layers.Input(shape=input_shape, name='data')

    # ResNet18 bottom
    x = layers.BatchNormalization(epsilon=2e-5, scale=False)(img_input)
    x = layers.ZeroPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer='he_uniform', use_bias=False)(x)
    x = layers.BatchNormalization(epsilon=2e-5)(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)

    # ResNet18 body
    repetitions = (2, 2, 2, 2)
    for stage, rep in enumerate(repetitions):
        for block in range(rep):
            filters = 64 * (2 ** stage)
            if block == 0 and stage == 0:
                x = residual_conv_block(filters, stage, block, strides=(1, 1), cut='post')(x)
            elif block == 0:
                x = residual_conv_block(filters, stage, block, strides=(2, 2), cut='post')(x)
            else:
                x = residual_conv_block(filters, stage, block, strides=(1, 1), cut='pre')(x)
    x = layers.BatchNormalization(epsilon=2e-5)(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(NUM_CLASSES)(x)
    x = layers.Activation('softmax')(x)

    model = models.Model(img_input, x)

    return model


with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

	# loading models
    model_path = 'models/model_h5/ResNet-18_fp32saved.h5'
    model = ResNet18(input_shape=IMG_SHAPE)
	# reset learning phase
    print(K.learning_phase())
    K.set_learning_phase(0)
    print(K.learning_phase())


    model.load_weights(model_path)

    export_path_habersinif = "models/model_tf/resnet18"


    builder_habersinif = saved_model_builder.SavedModelBuilder(export_path_habersinif)

    print("- - -")
    print('habersinif output')
    print(model.output)
    print('habersinif output')
    print(model.input)
    print("- - -")
    print('habersinif output sinif sayisi')
    num_classes=model.output.shape[1]
    print(num_classes)
    print("- - -")

    x = model.input
    y = model.output
    tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(y)
    prediction_signature = (tf.saved_model.signature_def_utils.build_signature_def(
		inputs={'inputs': tensor_info_x},
		outputs={
                  'scores': tensor_info_y
                },
		method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    builder_habersinif.add_meta_graph_and_variables(
		sess, [tf.saved_model.tag_constants.SERVING],
		signature_def_map={'predict': prediction_signature,},
		legacy_init_op=legacy_init_op)
    builder_habersinif.save()

    print('model')
    print(model.summary())

