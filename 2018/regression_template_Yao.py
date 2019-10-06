# -*- coding: utf-8 -*-

import numpy as np
import regressionData as rg
import time
import pdb
import math
from numpy import linalg as LA
#-------------------
# �N���X�̒��`�n�܂�
class linearRegression():
    #------------------------------------
    # 1) �w�K�f�[�^�����у��f���p�����[�^�̏�����
    # x: �w�K���̓f�[�^�i���̓x�N�g���̎������~�f�[�^����numpy.array�j
    # y: �w�K�o�̓f�[�^�i�f�[�^����numpy.array�j
    # kernelType: �J�[�l���̎��ށi�������Fgaussian�j
    # kernelParam: �J�[�l���̃n�C�p�[�p�����[�^�i�X�J���[�j
    def __init__(self, x, y, kernelType="linear", kernelParam=1.0):
        # �w�K�f�[�^�̐ݒ�
        self.x = x
        self.y = y
        self.xDim = x.shape[0]
        self.dNum = x.shape[1]

        # �J�[�l���̐ݒ�
        self.kernelType = kernelType
        self.kernelParam = kernelParam

    def calcDist(self,x,z):
        
        row,col=x.shape
        row1,col1=z.shape
        x=x[:,np.newaxis]
        z=z[:,np.newaxis]
        
        
        x_=np.tile(x.T,(1,row1,col1))
        z_=np.tile(z,(col,row,1))
        
        #pdb.set_trace()
        dist=np.sqrt(np.sum(pow(x_-z_,2),axis=1))
        pdb.set_trace()
        #norm=LA.norm(x_-z_)
        """
        xNum = x.shape[1]
        xDim = x.shape[0]
        zNum = z.shape[1]
        
        
        #gaussian グラム行列
        X = np.tile(np.reshape(x,[xNum,1,xDim]),[1,zNum,1])
        Z = np.tile(np.reshape(z,[1,zNum,xDim]),[xNum,1,1])
        dist = np.sum((X-Z)**2,axis=2)
        """
        return dist

    def karnel(self,x):
        calc=self.calcDist(self.x,x)
        #pdb.set_trace()
        #k=np.exp(-pow(calc,2)/2*(self.kernelParam)**2)
        k=np.exp(calc/2*(self.kernelParam)**2)
        #oku,row,col=k.shape
        #pdb.set_trace()
        #k=k.reshape(oku,col)
        #pdb.set_trace()
        return k

    def train(self):

        """for i in range(10):
            #print(self.y)
            print(self.y)"""
        length_x=0
        i=0
        """#xの要素数を求める
        for i in self.x:
            length_x+=len(i)
        """
        row,col=self.x.shape
        #print('length_x:{0}'.format(length_x))
        i=0
        x_sigma=0



        x_append=np.append(self.x,np.ones(col)[np.newaxis],axis=0)
        i=0

        #前半のシグマ部分の計算

        for i in range(col):
            x_sigma += np.dot(x_append[:,i][np.newaxis].T,(x_append[:,i][np.newaxis]))
        #print("train_x_sigma:{0}".format(x_sigma))

        x_sigma_inv = np.linalg.inv(x_sigma[np.newaxis])
        #print(self.y)
        length_y=0
        i=0

        #後半シグマ部分の計算
        yx_sigma=0
        for i in range(len(self.y)):
            yx_sigma += self.y[i]*(x_append[:,i][np.newaxis].T)



        #w_の計算
        w_ = np.dot(x_sigma_inv,yx_sigma)
        print("train_w_:{0}".format(w_))
        self.w=w_

    def trainMat(self):

        row,col=self.x.shape
        i=0
        #pdb.set_trace()
        x_append=np.append(self.x,np.ones(col)[np.newaxis],axis=0)

        front = np.matmul(x_append,x_append.T)
        #pdb.set_trace()

        back = np.matmul(x_append,self.y[np.newaxis].T)

        front=np.linalg.inv(front)
        #pdb.set_trace()
        w_ = np.matmul(front,back)
        #pdb.set_trace()
        print("train_w_:{0}".format(w_))
        self.w=w_
        #pdb.set_trace()

    def train_kernel(self,k):
        row,col=k.shape
        #pdb.set_trace()
        k=np.append(k,np.ones(col)[np.newaxis],axis=0)
        #pdb.set_trace()
        front=np.matmul(k,k.T)
        x_=np.ones_like(front)*0.001
        front=front+x_
        #pdb.set_trace()
        back=np.matmul(k,self.y[np.newaxis].T)
        front=np.linalg.inv(front)
        w_=np.matmul(front,back)
        #pdb.set_trace()
        #print("train_w_kernel:{0}".format(w_))
        self.w=w_

    def predict(self,x):
        row,col=x.shape
        #pdb.set_trace()
        x_append=np.append(x,np.ones(col)[np.newaxis],axis=0)
        #pdb.set_trace()
        return np.matmul(self.w.T,x_append)
        #return np.matmul(self.w.T,x_append)

    #------------------------------------

    #------------------------------------
    # 4) ���摹���̌v�Z
    # x: ���̓f�[�^�i���͎��� x �f�[�^���j
    # y: �o�̓f�[�^�i�f�[�^���j
    def loss(self,x,y):
        i=0
        #pdb.set_trace()
        row,col=x.shape
        #pdb.set_trace()
        loss=np.sum((y-self.predict(x))**2)/col
        #pdb.set_trace()
        return loss
    #------------------------------------
# �N���X�̒��`�I����
#-------------------

#-------------------
# ���C���̎n�܂�
if __name__ == "__main__":

    # 1) �w�K���͎�����2�̏ꍇ�̃f�[�^�[����
    myData = rg.artificial(200,100, dataType="1D",isNonlinear=True)
    #myData = rg.artificial(200,100, dataType="2D",isNonlinear=True)

    # 2) ���`���A���f��
    #regression = linearRegression(myData.xTrain,myData.yTrain)
    regression = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=1)

    # 3) �w�K�iFor���Łj

    sTime = time.time()
    regression.train()
    eTime = time.time()
    print("train with for-loop: time={0:.4} sec".format(eTime-sTime))

    # 4) �w�K�i�s���Łj
    sTime = time.time()
    regression.trainMat()
    eTime = time.time()
    print("train with matrix: time={0:.4} sec".format(eTime-sTime))

    sTime = time.time()
    regression.train_kernel(regression.karnel(myData.xTrain))
    eTime = time.time()
    print("train with kernel: time={0:.4} sec".format(eTime-sTime))


    # 5) �w�K�������f�����p���ė\��

    print("loss={0:.3}".format(regression.loss(regression.karnel(myData.xTest),myData.yTest)))

    # 6) �w�K�E�]���f�[�^�����ї\�����ʂ��v���b�g
    predict = regression.predict(regression.karnel(myData.xTest))
    print(predict)
    #pdb.set_trace()

    myData.plot(predict[0],isTrainPlot=True)

#���C���̏I����
#-------------------
