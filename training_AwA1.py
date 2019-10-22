import numpy as np
import time, sys
import src.dataLoader.AwA1 as dataLoader
import src.module.vae as vae
import tensorflow as tf

#==================== VAE structure example =========================
vaeStructure = {
    'name' : 'VAE',
    'encoder': {
        'name' : 'encoder',
        'trainable' : True,
        'inputDim' : 2048,
        # 'hiddenOutputDimList' : [1024, 256],
        'hiddenOutputDimList' : [512],
        'outputDim' : 64*2,
        'hiddenActivation' : tf.nn.leaky_relu,
        'lastLayerActivation' : None,
        'drRate' : 0.5,
    },

    'decoder': {
        'name' : 'decoder',
        'trainable' : True,
        'inputDim' : 64,
        # 'hiddenOutputDimList' : [512],
        'hiddenOutputDimList' : [],
        'outputDim' : 2048,
        'hiddenActivation' : tf.nn.leaky_relu,
        'lastLayerActivation' : None,
        'drRate' : 0.5,
    },

    'priorNet': {
        'name' : 'priorNet',
        'trainable' : True,
        'inputDim' : 85,
        'hiddenLayerNum' : 6,
        'outputDim' : 64,
        'hiddenActivation' : tf.nn.leaky_relu,
        'lastLayerActivation' : None,
        'constLogVar' : 0.0,
        'drRate' : 0.2,
    },
}

def getAcc(model, attribute, x, labelGT):
    priorMean, _ = model.getPriorMean(classVectors=attribute)
    features = []
    batchSize = 512
    dataLen = len(x)
    loopNum = int((dataLen + batchSize)/batchSize)
    for i in range(loopNum):
        dataStart = i*batchSize
        dataEnd = np.min([(i+1)*batchSize, dataLen])
        fMu, _ = model.getFeatures(inputVectors = x[dataStart:dataEnd])
        features.append(fMu)
    features = np.concatenate(features, axis=0)
    featuresTile = np.tile(np.reshape(features, [-1,1,features.shape[-1]]), [1,len(priorMean),1])
    priorMeanTile = np.tile(np.reshape(priorMean, [1,-1,priorMean.shape[-1]]), [len(features),1,1])
    distTile = np.square(featuresTile - priorMeanTile)
    dist = np.sum(distTile, axis=-1)
    classIndex = np.argmin(dist, axis=1)
    # print classIndex
    # print labelGT
    acc = np.where(classIndex==labelGT, 1.0, 0.0)
    acc = np.sum(acc)/float(len(acc))
    return classIndex, acc

def trainVAE(structure, batchSize, trainingEpoch, learningRate, savePath, restorePath=None):
    model = vae.vae(structure=structure)
    if restorePath!=None:
        print('restore networks...')
        model.restoreNetworks(restorePath=restorePath)
    dataset = dataLoader.dataLoader()

    loss = np.zeros(4)
    epoch = 0
    epochCurr = 0
    iteration = 0
    runTime = 0.0
    acchbefore = 0.0

    print('start training...')
    while epoch < trainingEpoch:
        startTime = time.time()
        batchData = dataset.getNextBatch(batchSize=batchSize)
        inputBatch = {
            'isTrainingEnc' : True,
            'isTrainingDec' : True,
            'isTrainingPrior' : True,
            'learningRate' : learningRate,
            'inputVectors' : batchData['inputVectors'],
            'classVectors' : batchData['classVectors'],
        }
        epochCurr = dataset._epoch
        dataStart = dataset._dataPointStart
        dataLength = dataset._dataPointNumTotal
        if int(epochCurr/3) != int(epoch/3):
            print('')
            # evaluation
            _, accSeen = getAcc(model, dataset.attribute, dataset._xTestSeen, dataset.labelTestSeen)
            _, accUnseen = getAcc(model, dataset.attribute, dataset._xTestUnseen, dataset.labelTestUnseen)
            accHarmonic = 2 * accSeen * accUnseen / (accSeen + accUnseen)
            print 'accSeen:{:.4f} accUnseen:{:.4f} accHarmonic:{:.4f}'.format(accSeen, accUnseen, accHarmonic)

            # reset and save
            iteration = 0
            loss = loss * 0.0
            runTime = 0.0
            if acchbefore < accHarmonic:
                acchbefore = accHarmonic
            if savePath != None:
                print('save model...')
                model.saveNetworks(savePath=savePath)

        epoch = epochCurr
        lossTemp = np.array(model.fit(batchDict=inputBatch))
        endTime = time.time()
        loss = (loss * iteration + lossTemp) / (iteration + 1.0)
        runTime = (runTime * iteration + (endTime - startTime)) / (iteration + 1.0)

        # print process
        sys.stdout.write(
            "Ep:{:04d} iter:{:04d} ".format(int(epoch+1), int(iteration+1), runTime)
        )
        sys.stdout.write(
            "curr/total:{:05d}/{:05d} ".format(dataStart, dataLength)
        )
        sys.stdout.write(
            "loss=total:{:4f},rec:{:4f},KL:{:4f},prReg:{:4f}     \r".format(loss[0], loss[1], loss[2], loss[3])
        )

        if loss[0] != loss[0]:
            print('')
            print('network diverges')
            return
        iteration += 1.0

    print('')
    print('save model...')
    model.saveNetworks(savePath=savePath)

if __name__ == "__main__":
    sys.exit(
        trainVAE(
            structure=vaeStructure,
            batchSize=64,
            trainingEpoch=1000,
            learningRate=1e-4,
            savePath='weights/vae_AwA1_init/',
            # restorePath='weights/vae_AwA1_init/',
        )
    )




















