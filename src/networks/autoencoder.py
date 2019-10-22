import numpy as np
import tensorflow as tf

# ==================== autoencoder structure example =========================
# encoderStructure = {
#     'name' : 'encoder',
#     'trainable' : True,
#     'inputDim' : 2048,
#     'hiddenOutputDimList' : [512, 128],
#     'outputDim' : 128*2,
#     'hiddenActivation' : tf.nn.leaky_relu,
#     'lastLayerActivation' : None,
# }
#
# decoderStructure = {
#     'name' : 'decoder',
#     'trainable' : True,
#     'inputDim' : 128,
#     'hiddenOutputDimList' : [512],
#     'outputDim' : 2048,
#     'hiddenActivation' : tf.nn.leaky_relu,
#     'lastLayerActivation' : None,
# }

class encoder(object):
    def __init__(self, structure):
        self._trainable = structure['trainable']
        self._inputDim = structure['inputDim']
        self._hiddenOutputDimList = structure['hiddenOutputDimList']
        self._outputDim = structure['outputDim']
        self._hiddenActivation = structure['hiddenActivation']
        self._lastLayerActivation = structure['lastLayerActivation']

        self._variableScope = structure['name']
        self.variables, self.update_ops, self.saver = None, None, None
        self._reuse = False
        self._drRate = structure['drRate']

    def __call__(self, inputVector, isTraining):
        print('encoder - ' + self._variableScope)
        totalDepth = len(self._hiddenOutputDimList)
        with tf.variable_scope(self._variableScope, reuse=self._reuse):
            hidden = inputVector
            print(hidden.shape)
            for depth in range(totalDepth):
                hiddenOutputDim = self._hiddenOutputDimList[depth]
                hidden = tf.layers.dense(
                    inputs=hidden, units=int(hiddenOutputDim), activation=None, use_bias=True, trainable=self._trainable)
                hidden = self._hiddenActivation(hidden)
                hidden = tf.layers.batch_normalization(hidden, training=isTraining, trainable=self._trainable)
                if self._drRate > 0:
                    hidden = tf.layers.dropout(hidden, rate=self._drRate, training=isTraining)
                print(hidden.shape)
            hidden = tf.layers.dense(
                inputs=hidden, units=self._outputDim, activation=None, use_bias=True, trainable=self._trainable)
            if self._lastLayerActivation != None:
                hidden = self._lastLayerActivation(hidden)
            print(hidden.shape)
        self._reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._variableScope)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self._variableScope)
        self.saver = tf.train.Saver(var_list=self.variables)
        return hidden

class decoder(object):
    def __init__(self, structure):
        self._trainable = structure['trainable']
        self._inputDim = structure['inputDim']
        self._hiddenOutputDimList = structure['hiddenOutputDimList']
        self._outputDim = structure['outputDim']
        self._hiddenActivation = structure['hiddenActivation']
        self._lastLayerActivation = structure['lastLayerActivation']

        self._variableScope = structure['name']
        self.variables, self.update_ops, self.saver = None,None,None
        self._reuse = False
        self._drRate = structure['drRate']

    def __call__(self, inputVector, isTraining):
        print('decoder - ' + self._variableScope)
        totalDepth = len(self._hiddenOutputDimList)
        with tf.variable_scope(self._variableScope, reuse=self._reuse):
            hidden = inputVector
            print(hidden.shape)
            for depth in range(totalDepth):
                hiddenOutputDim = self._hiddenOutputDimList[depth]
                hidden = tf.layers.dense(
                    inputs = hidden, units = hiddenOutputDim, activation=None, use_bias=True, trainable=self._trainable)
                hidden = self._hiddenActivation(hidden)
                hidden = tf.layers.batch_normalization(hidden, training=isTraining, trainable=self._trainable)
                if self._drRate > 0:
                    hidden = tf.layers.dropout(hidden, rate=self._drRate, training=isTraining)
                print hidden.shape
            hidden = tf.layers.dense(
                inputs=hidden, units=self._outputDim, activation=None, use_bias=True, trainable=self._trainable)
            if self._lastLayerActivation != None:
                hidden = self._lastLayerActivation(hidden)
            print hidden.shape
        self._reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._variableScope)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self._variableScope)
        self.saver = tf.train.Saver(var_list=self.variables)
        return hidden














