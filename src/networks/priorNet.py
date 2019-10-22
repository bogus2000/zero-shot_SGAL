import numpy as np
import tensorflow as tf

#================= priorNet structure example ====================
# priorNetStructure = {
#     'name' : 'priorNet',
#     'trainable' : True,
#     'inputDim' : 85,
#     'hiddenLayerNum' : 2,
#     'outputDim' : 128,
#     'hiddenActivation' : tf.nn.leaky_relu,
#     'lastLayerActivation' : None,
#     'constLogVar' : None,
# }

class priorNet(object):
    def __init__(self, structure):
        self._trainable = structure['trainable']
        self._inputDim = structure['inputDim']
        self._outputDim = structure['outputDim']
        self._hiddenLayerNum = structure['hiddenLayerNum']
        self._variableScope = structure['name']
        self.variables, self.update_ops, self.saver = None, None, None
        self._coreAct, self._lastAct = structure['hiddenActivation'], structure['lastLayerActivation']
        self._constLogVar = structure['constLogVar']
        self._reuse = False
        self._drRate = structure['drRate']

    def __call__(self, inputVector, isTraining):
        print("priorNet - " + self._variableScope)
        with tf.variable_scope(self._variableScope, reuse=self._reuse):
            ratio = np.power(float(self._outputDim) / float(self._inputDim), 1.0 / float(self._hiddenLayerNum))
            layerDim = self._inputDim
            # hidden = 2.0 * inputVector - 1.0
            hidden = inputVector
            print "mean prior"
            print(hidden.shape)
            for i in range(self._hiddenLayerNum - 1):
                layerDim = layerDim * ratio
                hidden = tf.layers.dense(
                    inputs=hidden, units=int(layerDim), activation=None, use_bias=True, trainable=self._trainable)
                hidden = tf.layers.batch_normalization(hidden, training=isTraining, trainable=self._trainable)
                if self._drRate > 0.0:
                    hidden = tf.layers.dropout(hidden, rate=self._drRate, training=isTraining)
                if self._coreAct != None:
                    hidden = self._coreAct(hidden)
                print hidden.shape
            meanPrior = tf.layers.dense(
                inputs=hidden, units=self._outputDim, activation=None, use_bias=True, trainable=self._trainable)
            if self._lastAct != None:
                meanPrior = self._lastAct(meanPrior)
            print meanPrior.shape

            if self._constLogVar == None:
                print "logVar prior"
                layerDim = self._inputDim
                # hidden = 2.0 * inputVector - 1.0
                hidden = inputVector
                print(hidden.shape)
                for i in range(self._hiddenLayerNum - 1):
                    layerDim = layerDim * ratio
                    hidden = tf.layers.dense(
                        inputs=hidden, units=int(layerDim), activation=None, use_bias=True, trainable=self._trainable)
                    hidden = tf.layers.batch_normalization(hidden, training=isTraining, trainable=self._trainable)
                    if self._drRate > 0.0:
                        hidden = tf.layers.dropout(hidden, rate=self._drRate, training=isTraining)
                    if self._coreAct != None:
                        hidden = self._coreAct(hidden)
                    print hidden.shape
                logVarPrior = tf.layers.dense(
                    inputs=hidden, units=self._outputDim, activation=None, use_bias=True, trainable=self._trainable)
                if self._lastAct != None:
                    logVarPrior = self._lastAct(logVarPrior)
                print logVarPrior.shape
            elif self._constLogVar == self._constLogVar:
                print "logVar prior : constant " + str(self._constLogVar)
                logVarPrior = self._constLogVar * tf.ones_like(meanPrior)
            else:
                logVarPrior = None
        self._reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self._variableScope)
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=self._variableScope)
        self.saver = tf.train.Saver(var_list=self.variables)
        return meanPrior, logVarPrior