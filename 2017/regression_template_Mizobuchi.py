# -*- coding: utf-8 -*-

import numpy as np
import regressionData as rg
import time
import pdb

#-------------------
# �N���X�̒��`�n�܂�
class linearRegression():
    #------------------------------------
    # 1) �w�K�f�[�^�����у��f���p�����[�^�̏�����
    # x: �w�K���̓f�[�^�i���̓x�N�g���̎������~�f�[�^����numpy.array�j
    # y: �w�K�o�̓f�[�^�i�f�[�^����numpy.array�j
    def __init__(self, x, y):
        # �w�K�f�[�^�̐ݒ�
        self.x = x
        self.y = y
        self.xDim = x.shape[0]
        self.dNum = x.shape[1]
    #------------------------------------

    #------------------------------------
    # 2) �ŏ������@���p���ă��f���p�����[�^���œK��
    # �i�����̌v�Z��For�����p�����ꍇ�j
    def train(self):
        xTrain_d = np.append(self.x,np.array([1 for i in np.arange(self.x.shape[1])])[np.newaxis],axis=0)
        num1 = np.zeros([xTrain_d.shape[0],1])
        num2 = np.zeros([xTrain_d.shape[0],xTrain_d.shape[0]])
        for i in np.arange(self.x.shape[1]):
            num1 += self.y[i]*xTrain_d[:,i][np.newaxis].transpose()
            num2 += np.dot(xTrain_d[:,i][np.newaxis].transpose(),xTrain_d[:,i][np.newaxis])
        self.w = np.dot(np.linalg.inv(num2),num1)[np.newaxis]
        
        pdb.set_trace()
    #------------------------------------

    #------------------------------------
    # 2) �ŏ������@���p���ă��f���p�����[�^���œK���i�s�񉉎Z�ɂ��荂�����j
    def trainMat(self):
        xTrain_d = np.append(self.x,np.array([1 for i in np.arange(self.x.shape[1])])[np.newaxis],axis=0)
        num1=np.dot(self.y,xTrain_d.T)[np.newaxis].T
        num2=np.dot(xTrain_d,xTrain_d.T)
        self.w =np.dot(np.linalg.inv(num2),num1)
    #------------------------------------

    #------------------------------------
    # 3) �\��
    # x: ���̓f�[�^�i���͎��� x �f�[�^���j
    def predict(self,x):
        x_T = np.append(x,np.ones([1,x.shape[1]]),axis=0).T
        y = np.dot(x_T,self.w).T
        #pdb.set_trace()
        return y
    #------------------------------------

    #------------------------------------
    # 4) ���摹���̌v�Z
    # x: ���̓f�[�^�i���͎��� x �f�[�^���j
    # y: �o�̓f�[�^�i�f�[�^���j
    def loss(self,x,y):
        testY = self.predict(x)
        loss = np.mean(np.power(testY-y,2),axis=1)
        #pdb.set_trace()
        return loss

    #p=ハイパーパラメータ
    def kernel(self,x,p):
        k = np.power(np.dot(self.x.T,x)+1,p)
        pdb.set_trace()
        return k


    def trainMatKernel(self,k,rmd):
        xTrain_d = np.append(k,np.ones([1,k.shape[1]]),axis=0)
        num1=np.dot(self.y,xTrain_d.T)[np.newaxis].T
        num2=np.dot(xTrain_d,xTrain_d.T)+(rmd*np.eye(xTrain_d.shape[0]))
        self.w =np.dot(np.linalg.inv(num2),num1)
        pdb.set_trace()
    #------------------------------------
# �N���X�̒��`�I����
#-------------------

#-------------------
# ���C���̎n�܂�
if __name__ == "__main__":

    # 1) �w�K���͎�����2�̏ꍇ�̃f�[�^�[����
    myData = rg.artificial(200,100, dataType="1D", isNonlinear=True)

    # 2) ���`���A���f��
    regression = linearRegression(myData.xTrain,myData.yTrain)
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

    # 5) �w�K�������f�����p���ė\��
    print("loss={0:.3}".format(regression.loss(myData.xTest,myData.yTest)[0]))

    # 6) �w�K�E�]���f�[�^�����ї\�����ʂ��v���b�g
    regression.trainMatKernel(regression.kernel(myData.xTrain,3),0.1)
    predict = regression.predict(regression.kernel(myData.xTest,3))
    myData.plot(predict[0])

#���C���̏I����
#-------------------
