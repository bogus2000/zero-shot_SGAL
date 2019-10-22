import scipy.io
import numpy as np
import os

'''data from
DeepEmbeddingModel_ZSL
https://github.com/lzrobots/DeepEmbeddingModel_ZSL'''

class dataLoader(object):
    def __init__(self,
                 dataPath = '/media/yonsei/4TB_HDD/downloads/data/',
                 dataset = 'AwA1_data',
                 imageEmbedding = 'res101',
                 classEmbedding = 'original_att'
                 ):
        self._epoch = 0

        imageMatContent = scipy.io.loadmat(os.path.join(dataPath, dataset, imageEmbedding+'.mat'))
        features = imageMatContent['features'].T
        labels = imageMatContent['labels'].astype(int).squeeze() - 1 # because it's from matlab - start with 1, not 0

        classMatContent = scipy.io.loadmat(os.path.join(dataPath, dataset, classEmbedding+'_splits.mat'))
        trainvalLoc = classMatContent['trainval_loc'].squeeze() - 1
        testSeenLoc = classMatContent['test_seen_loc'].squeeze() - 1
        testUnseenLoc = classMatContent['test_unseen_loc'].squeeze() - 1
        self.attribute = classMatContent['att'].T

        print('load data...')
        self._xTrain = features[trainvalLoc]
        labelTrain = labels[trainvalLoc].astype(int)
        self._attTrain = self.attribute[labelTrain]
        self._dataPointNumTotal = len(self._xTrain)
        self._dataPointIndex = np.array([k for k in range(self._dataPointNumTotal)])
        self._dataPointStart = 0
        self._dataShuffle()

        self._xTestUnseen = features[testUnseenLoc].astype('float')
        self.labelTestUnseen = labels[testUnseenLoc].astype(int)
        self._attTestUnseen = self.attribute[self.labelTestUnseen].astype('float')
        self._unseenDataPointNumTotal = len(self._xTestUnseen)
        self._unseenDataPointIndex = np.array([k for k in range(self._unseenDataPointNumTotal)])

        self._xTestSeen = features[testSeenLoc].astype('float')
        self.labelTestSeen = labels[testSeenLoc].astype(int)
        self._attTestSeen = self.attribute[self.labelTestSeen].astype('float')
        self._seenDataPointNumTotal = len(self._xTestSeen)
        self._seenDataPointIndex = np.array([k for k in range(self._seenDataPointNumTotal)])

        #should be implemented
        self.xValSeen, self.labelValSeen = None, None
        self.xValUnseen, self.labelValUnseen = None, None
        print('done!')

    def _dataShuffle(self):
        self._dataPointStart = 0
        np.random.shuffle(self._dataPointIndex)

    def getNextBatch(self, batchSize = 32):
        if self._dataPointStart + batchSize >= self._dataPointNumTotal:
            self._epoch += 1
            self._dataShuffle()
        dataStart = self._dataPointStart
        dataEnd = dataStart + batchSize
        self._dataPointStart = dataEnd
        dataPointIndex = self._dataPointIndex[dataStart:dataEnd]
        inputVectors = self._xTrain[dataPointIndex]
        classVectors = self._attTrain[dataPointIndex]

        batchData = {
            'inputVectors':inputVectors.astype('float'),
            'classVectors':classVectors.astype('float'),
        }
        return batchData

    def getRandomAttributes(self, batchSize=16, unseen=True):
        att = None
        dataPointIndex = None
        if unseen:
            att = self._attTestUnseen
            dataPointIndex = self._unseenDataPointIndex
        else:
            att = self._attTestSeen
            dataPointIndex = self._seenDataPointIndex
        classVectors = []
        for i in range(batchSize):
            np.random.shuffle(dataPointIndex)
            classVectors.append(att[dataPointIndex[0]])
        classVectors = np.array(classVectors)
        return classVectors.astype('float')















