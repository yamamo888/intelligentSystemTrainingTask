# -*- coding: utf-8 -*-

import numpy as np
import regressionData as rg
import time
import pdb

#-------------------

class linearRegression():
    #------------------------------------
    
    def __init__(self, x, y, kernelType="linear", kernelParam=0.1):
        
        self.x = x
        self.y = y
        self.xDim = x.shape[0]
        self.dNum = x.shape[1]
        
        
        self.kernelType = kernelType
        self.kernelParam = kernelParam
        
    #------------------------------------

    #------------------------------------
    # 2) train(for)
    def train(self,x,y):
        #self.w = np.zeros([self.xDim,1])                
        self.single_x = np.append(self.x,np.ones((1,self.dNum)),axis=0)
        
        den = np.zeros((self.single_x.shape[0],1))
        mol = np.zeros((self.single_x.shape[0],self.single_x.shape[0]))


        for ind in np.arange(self.dNum):

            den += self.y[ind] * self.single_x[:,ind][np.newaxis].T
            mol += np.dot(self.single_x[:,ind][np.newaxis].T,self.single_x[:,ind][np.newaxis])
        

        self.w = np.dot(np.linalg.inv(mol),den)
        
    #------------------------------------

    #------------------------------------
    # 2) train(mat)
    def trainMat(self,x,y):
        den = np.matmul(self.single_x,self.y.T)
        mol = np.matmul(self.single_x,self.single_x.T)
        
        self.w = np.dot(np.linalg.inv(mol),den.T)
        #pdb.set_trace()
    #------------------------------------
  
    #------------------------------------
    #3.5)train(kernel)
    def trainMatkernel(self,x,y,ramda):
        #self.w = np.zeros([self.dNum + 1,1])
        single_kernel = np.append(self.kernel(x),np.ones((1,x.shape[1])),axis=0)
        
        #分母分子の計算
        den = np.matmul(single_kernel,self.y.T)
        mol = np.matmul(single_kernel,single_kernel.T)
        
        self.w = np.dot(np.linalg.inv(mol+ramda*np.eye(single_kernel.shape[0],single_kernel.shape[0])),den.T)
        #pdb.set_trace()
        
    
    # 3)テストデータを用いて、予測
    def predict(self,x):
      
        x_T = np.append(x,np.ones((1,x.shape[1])),axis=0).T
        y = np.dot(x_T,self.w.T)
        
        #pdb.set_trace()
        
        return y
    #------------------------------------

    #------------------------------------
    # 4) 二乗損失誤差
    def loss(self,x,y):
        ytest = self.predict(self.kernel(x))
        loss = np.sum((y - ytest)**2/self.dNum)
        #pdb.set_trace()
        return loss
    #------------------------------------
    # NXÌè`Iíè
    
    def kernel(self,x):
        if(self.kernelType == "gaussian"): 
            #gaussian
            kernel = np.exp(-self.trainkernelCalc(self.x,x)/ (2 * (self.kernelParam ** 2)))
        else:
            #多項式
            kernel = np.power(self.trainkernelCalc(self.x,x)+1,self.kernelParam)
            
        return kernel
#-------------------
            
    def trainkernelCalc(self,x,z):
        xNum = x.shape[1]
        xDim = x.shape[0]
        zNum = z.shape[1]
        
        if(self.kernelType == "gaussian"):
            #gaussian グラム行列
            X = np.tile(np.reshape(x,[xNum,1,xDim]),[1,zNum,1])
            Z = np.tile(np.reshape(z,[1,zNum,xDim]),[xNum,1,1])
            k = np.sum((X-Z)**2,axis=2)
            
        else:
            # 多項式 グラム行列
            X = np.tile(x,[1,zNum,1])
            Z = np.tile(z,[xNum,1,1])
            #X = np.tile(np.reshape(x,[xNum,1,xDim]),[1,zNum,1])
            #Z = np.tile(np.reshape(z,[1,zNum,xDim]),[xNum,1,1])
            
            k = np.sum(X.T,Z,axis=2)
        return k
#-------------------
# CÌnÜè
if __name__ == "__main__":
    
    # 1) 元データ
    #myData = rg.artificial(200,100, dataType="1D")
    myData = rg.artificial(200,100, dataType="2D",isNonlinear=True)
    
    # 2) 線形回帰
    #regression = linearRegression(myData.xTrain,myData.yTrain)
    # Polynomial:多項式 gaussian:ガウシアン
    regression = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=5)
    
    """
    # 3) train(for)
    sTime = time.time()
    regression.train(myData.xTrain,myData.yTrain)
    eTime = time.time()
    print("train with for-loop: time={0:.4} sec".format(eTime-sTime))
    
    # 4) train(Mat)
    sTime = time.time()
    regression.trainMat(myData.xTrain,myData.yTrain)
    eTime = time.time()
    print("train with matrix: time={0:.4} sec".format(eTime-sTime))
    """
    #4) train(kernel)
    # parameter
    ramda = 0.01
    regression.trainMatkernel(myData.xTrain,myData.yTrain,ramda)
    
    #pdb.set_trace()

    # 5) loss
    print("loss={0:.3}".format(regression.loss(myData.xTest,myData.yTest)))
    
    #6)予測
    #pdb.set_trace()
    predict = regression.predict(regression.kernel(myData.xTest))
    myData.plot(predict,isTrainPlot=False)
    


#-------------------
    