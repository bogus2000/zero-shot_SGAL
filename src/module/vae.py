from function import *
import tensorflow as tf
import numpy as np
import src.networks.autoencoder as AE
import src.networks.priorNet as priorNet

#==================== VAE structure example =========================
# vaeStructure = {
#     'name' : 'VAE',
#     'encoder': {
#         'name' : 'encoder',
#         'trainable' : True,
#         'inputDim' : 2048,
#         'hiddenOutputDimList' : [512, 128],
#         'outputDim' : 128*2,
#         'hiddenActivation' : tf.nn.leaky_relu,
#         'lastLayerActivation' : None,
#     },
#
#     'decoder': {
#         'name' : 'decoder',
#         'trainable' : True,
#         'inputDim' : 128,
#         'hiddenOutputDimList' : [512],
#         'outputDim' : 2048,
#         'hiddenActivation' : tf.nn.leaky_relu,
#         'lastLayerActivation' : None,
#     },
#
#     'priorNet': {
#         'name' : 'priorNet',
#         'trainable' : True,
#         'inputDim' : 85,
#         'hiddenLayerNum' : 2,
#         'outputDim' : 128,
#         'hiddenActivation' : tf.nn.leaky_relu,
#         'lastLayerActivation' : None,
#         'constLogVar' : None,
#     },
# }

class vae(object):
    def __init__(self, structure):
        self._netStructure = structure
        self._encStructure = structure['encoder']
        self._decStructure = structure['decoder']
        self._priorStructure = structure['priorNet']

        # encoder input
        self._inputVectors = tf.placeholder(tf.float32, shape=[None, self._encStructure['inputDim']])
        # encoder output
        self._mu, self._logVar = None,None

        # decoder input
        self._z = None
        # decoder output
        self._outputVectors = None
        # decoder output GT
        self._outputVectorsGT = self._inputVectors

        # priorNet input
        self._classVectors = tf.placeholder(tf.float32, shape=[None, self._priorStructure['inputDim']])
        # priorNet output
        self._muPrior, self._logVarPrior = None,None

        # for training:
        self._learningRate = tf.placeholder(tf.float32, shape=[])
        self._isTrainingEnc = tf.placeholder(tf.bool)
        self._isTrainingDec = tf.placeholder(tf.bool)
        self._isTrainingPrior = tf.placeholder(tf.bool)

        # build network
        self._buildNetwork()
        # create loss
        self._createLoss()
        # set optimizer
        self._setOptimizer()

        # init the session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.93)
        config = tf.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=config)
        # initialize variables
        init = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )
        # launch the session
        self._sess.run(init)


    def _buildNetwork(self):
        print('build network...')
        #============== encoder =====================
        self._encoder = AE.encoder(structure=self._encStructure)
        #============== decoder =====================
        self._decoder = AE.decoder(structure=self._decStructure)
        #============== prior network ===============
        self._priorNet = priorNet.priorNet(structure=self._priorStructure)

        # get encoder output
        encOut = self._encoder(
            inputVector=self._inputVectors, isTraining=self._isTrainingEnc)
        self._mu = encOut[...,:self._encStructure['outputDim']/2]
        self._logVar = encOut[...,self._encStructure['outputDim']/2:]
        self._z = sampling(mu=self._mu, logVar=self._logVar)
        # get decoder output
        self._outputVectors = self._decoder(
            inputVector=self._z, isTraining=self._isTrainingDec)

        # get prior network output
        self._muPrior, self._logVarPrior = self._priorNet(
            inputVector=self._classVectors, isTraining=self._isTrainingPrior)

        # generate unseen class datapoints
        self._zFromPrior = sampling(mu=self._muPrior, logVar=self._logVarPrior)
        self._outputVectorsGenerated = self._decoder(self._zFromPrior, isTraining=self._isTrainingDec)

    def _createLoss(self):
        print('create loss...')
        self._reconstructionLoss = tf.reduce_mean(
            tf.reduce_sum(
                tf.square(self._outputVectors - self._outputVectorsGT) , axis=-1))

        self._KLLoss = tf.reduce_mean(
            nlb_loss(mean=self._mu, logVar=self._logVar,
                     mean_target=self._muPrior, logVar_target=self._logVarPrior))

        self._priorRegulizationLoss = 0.1 * tf.reduce_mean(
            regulizer_loss(z_mean=self._muPrior, z_logVar=self._logVarPrior,
                           dist_in_z_space = 2.0*self._priorStructure['outputDim']))

        self._totalLoss = (
            self._reconstructionLoss
            +
            self._KLLoss
            +
            self._priorRegulizationLoss
        )

    def _setOptimizer(self):
        print('set optimizer...')
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learningRate)
        self._update_variables = tf.get_collection(key=None, scope=None)
        self._update_ops = tf.get_collection(key=None, scope=None)
        if self._encStructure['trainable']:
            self._update_variables += self._encoder.variables
            self._update_ops += self._encoder.update_ops
        if self._decStructure['trainable']:
            self._update_variables += self._decoder.variables
            self._update_ops += self._decoder.update_ops
        if self._priorStructure['trainable']:
            self._update_variables += self._priorNet.variables
            self._update_ops += self._priorNet.update_ops

        with tf.control_dependencies(self._update_ops):
            self._optimizer = self._optimizer.minimize(
                self._totalLoss,
                var_list = self._update_variables
            )

        self._optimizerPrior = tf.train.AdamOptimizer(learning_rate=self._learningRate)
        with tf.control_dependencies(self._priorNet.update_ops):
            self._optimizerPrior = self._optimizerPrior.minimize(
                self._muPrior,
                var_list = self._priorNet.variables
            )

    def fit(self, batchDict):
        feed_dict = {
            self._isTrainingEnc : batchDict['isTrainingEnc'],
            self._isTrainingDec : batchDict['isTrainingDec'],
            self._isTrainingPrior:batchDict['isTrainingPrior'],
            self._learningRate : batchDict['learningRate'],
            self._inputVectors : batchDict['inputVectors'],
            self._classVectors : batchDict['classVectors'],
        }
        optimizer = self._optimizer
        lossList = self._totalLoss, self._reconstructionLoss, self._KLLoss, self._priorRegulizationLoss
        opt, loss = self._sess.run([optimizer, lossList], feed_dict=feed_dict)
        return loss

    def fitPrior(self, classVectors, learningRate, isTrainingPrior=True):
        optimizer = self._optimizerPrior
        _ = self._sess.run(optimizer, feed_dict={
            self._isTrainingPrior : isTrainingPrior,
            self._learningRate : learningRate,
            self._classVectors:classVectors
        })

    def generateDataPoints(self, classVectors):
        dataPoints = self._sess.run(
            self._outputVectorsGenerated,
            feed_dict={
                self._isTrainingDec : False,
                self._isTrainingPrior : False,
                self._classVectors : classVectors
            }
        )
        return dataPoints

    def generateDataPointsWithDropout(self, classVectors, samplingNum=4):
        zFromPrior = self._sess.run(
            self._zFromPrior,
            feed_dict={
                self._isTrainingPrior : False,
                self._classVectors : classVectors,
            }
        )
        classVectorsStack = classVectors
        zFromPriorStack = zFromPrior
        for i in range(samplingNum-1):
            classVectorsStack = np.concatenate([classVectorsStack, classVectors], axis=0)
            zFromPriorStack = np.concatenate([zFromPriorStack, zFromPrior], axis=0)
        dataPoints = self._sess.run(
            self._outputVectorsGenerated,
            feed_dict={
                self._isTrainingDec : True,
                self._zFromPrior : zFromPriorStack
            }
        )
        return classVectorsStack, dataPoints

    def getFeatures(self, inputVectors):
        fMu, fLogVar = self._sess.run(
            [self._mu, self._logVar],
            feed_dict={
                self._isTrainingEnc : False,
                self._inputVectors : inputVectors
            }
        )
        return fMu, fLogVar

    def getPriorMean(self, classVectors):
        muPrior, logVarPrior = self._sess.run(
            [self._muPrior, self._logVarPrior],
            feed_dict={
                self._isTrainingPrior : False,
                self._classVectors : classVectors
            }
        )
        return muPrior, logVarPrior

    def saveEncoder(self, savePath='./'):
        ePath = os.path.join(savePath, self._netStructure['name'] + '_encoder.ckpt')
        self._encoder.saver.save(self._sess, ePath)
    def saveDecoder(self, savePath='./'):
        dPath = os.path.join(savePath, self._netStructure['name'] + '_decoder.ckpt')
        self._decoder.saver.save(self._sess, dPath)
    def savePriorNet(self, savePath='./'):
        pPath = os.path.join(savePath, self._netStructure['name'] + '_priorNet.ckpt')
        self._priorNet.saver.save(self._sess, pPath)
    def saveNetworks(self, savePath='./'):
        self.saveEncoder(savePath)
        self.saveDecoder(savePath)
        self.savePriorNet(savePath)

    def restoreEncoder(self, restorePath='./'):
        ePath = os.path.join(restorePath, self._netStructure['name'] + '_encoder.ckpt')
        self._encoder.saver.restore(self._sess, ePath)
    def restoreDecoder(self, restorePath='./'):
        dPath = os.path.join(restorePath, self._netStructure['name'] + '_decoder.ckpt')
        self._decoder.saver.restore(self._sess, dPath)
    def restorePriorNet(self, restorePath='./'):
        pPath = os.path.join(restorePath, self._netStructure['name'] + '_priorNet.ckpt')
        self._priorNet.saver.restore(self._sess, pPath)
    def restoreNetworks(self, restorePath):
        self.restoreEncoder(restorePath)
        self.restoreDecoder(restorePath)
        self.restorePriorNet(restorePath)




















