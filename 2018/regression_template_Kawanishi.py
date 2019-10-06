# -*- coding: utf-8 -*-

import numpy as np
import regressionData as rg
import time
import pdb

#-------------------
# クラスの定義始まり
class linearRegression():
    #------------------------------------
    # 1) 学習データおよびモデルパラメータの初期化
    # x: 学習入力データ（入力ベクトルの次元数×データ数のnumpy.array）
    # y: 学習出力データ（データ数のnumpy.array）
    # kernelType: カーネルの種類（文字列：gaussian）
    # kernelParam: カーネルのハイパーパラメータ（スカラー）
    def __init__(self, x, y, kernelType="linear", kernelParam=1.0):
        # 学習データの設定
        self.x = x
        self.y = y
        self.xDim = x.shape[0]
        self.dNum = x.shape[1]

        # カーネルの設定
        self.kernelType = kernelType
        self.kernelParam = kernelParam
    #------------------------------------

    #------------------------------------
    # 2) 最小二乗法を用いてモデルパラメータを最適化
    # （分子・分母の計算にFor文を用いた場合）
    def train(self):
        self.w = np.zeros([self.xDim,1])
        xTrain = np.append(self.x,np.ones([1,self.dNum]),axis=0)
        yTrain = self.y

        w_1 = np.zeros([xTrain.shape[0],xTrain.shape[0]])
        w_2 = np.zeros([xTrain.shape[0],1])
        for ind in np.arange(self.dNum):
            w_1 += np.dot(xTrain[:,ind][np.newaxis].T,xTrain[:,ind][np.newaxis])
            w_2 += np.dot(xTrain[:,ind][np.newaxis].T,yTrain[ind])
        self.w = np.dot(np.linalg.inv(w_1),w_2)
    #------------------------------------

    #------------------------------------
    # 2) 最小二乗法を用いてモデルパラメータを最適化
    # （分子・分母の計算に行列演算を用いた場合）
    def trainMat(self):
        self.w = np.zeros([self.xDim,1])
        xTrain = np.append(self.x,np.ones([1,self.dNum]),axis=0)
        w_1 = np.dot(xTrain,xTrain.T)
        w_2 = np.dot(xTrain,self.y)
        self.w = np.dot(np.linalg.inv(w_1),w_2)
    #------------------------------------


    #------------------------------------
    # 6) 2つのデータ集合間のすべての組み合わせ距離の計算
    # x: 行列（次元 x データ数）
    # z: 行列（次元 x データ数）
    def calcDist(self,x,z):
        #【行列xのデータ点x1, x2, ..., xNと、行列zのデータ点z1, z2, ..., zMとの間のMxN個の距離を計算】
        x1 = np.tile(x.T[:,np.newaxis],(1,1,1))
        X = np.tile(x1,(1,z.shape[1],1))
        z1 = np.tile(z.T,(1,1,1))
        Z = np.tile(z1,(x.shape[1],1,1))
        dist = np.sqrt(np.sum(np.square(X - Z),axis=2))
        pdb.set_trace()
        return dist
    #------------------------------------

    #------------------------------------
    # 5) カーネルの計算
    # x: カーネルを計算する対象の行列（次元 x データ数）
    def kernel(self,x):
        #【self.xの各データ点xiと行列xの各データ点xjと間のカーネル値k(xi,xj)を各要素に持つグラム行列を計算】
        K = np.exp(np.square(self.calcDist(self.x,x)) / -2*np.square(self.kernelParam))
        return K
    #------------------------------------

    #------------------------------------
    # カーネルモデルのパラメータを最適化
    def trainMatKernel(self,lam):
        #self.w = np.zeros([self.x.shape[0],1])
        K = self.kernel(self.x)
        K = np.append(K,np.ones([1,K.shape[1]]),axis=0)
        pdb.set_trace()
        w_1 = np.dot(K,K.T) + lam*np.eye(K.shape[0])    #w_1.shape=[201,201]
        w_2 = np.dot(K,self.y.T)    #w_2.shape=[201,]
        self.w = np.dot(np.linalg.inv(w_1),w_2)    #w.shape=[201,]
    #------------------------------------


    #------------------------------------
    # 3) 予測
    # x: 入力データ（入力次元 x データ数）
    def predict(self,x):
        y = []
        #mainでxを指定
        x = np.append(x,np.ones([1,x.shape[1]]),axis=0)
        y = np.dot(x.T,self.w).T
        return y
    #------------------------------------

    #------------------------------------
    # 4) 二乗損失の計算
    # x: 入力データ（入力次元 x データ数）
    # y: 出力データ（データ数）
    def loss(self,x,y):
        loss = 0.0
        y_p = self.predict(x)
        loss = np.sum(np.square(y - y_p))
        loss = loss / x.shape[1]
        return loss
    #------------------------------------

# クラスの定義終わり
#-------------------

#-------------------
# メインの始まり
if __name__ == "__main__":

    # 1) 学習入力次元が2の場合のデーター生成
    #myData = rg.artificial(200,100, dataType="1D")
    #myData = rg.artificial(200,100, dataType="2D",isNonlinear=False)
    myData = rg.artificial(200,100, dataType="1D",isNonlinear=True)
    #myData = rg.artificial(200,100, dataType="2D",isNonlinear=True)

    # 2) 線形回帰モデル
    #regression = linearRegression(myData.xTrain,myData.yTrain)
    regression = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=1)
    #regression = linearRegression(myData.xTrain,myData.yTrain,kernelType="linear",kernelParam=1)

    # 3) 学習（For文版）
    sTime = time.time()
    regression.train()
    eTime = time.time()
    print("train with for-loop: time={0:.4} sec".format(eTime-sTime))

    # 4) 学習（行列版）
    sTime = time.time()
    regression.trainMat()
    eTime = time.time()
    print("train with matrix: time={0:.4} sec".format(eTime-sTime))


    sTime = time.time()
    regression.trainMatKernel(0.01)
    eTime = time.time()
    print("kernel with matrix: time={0:.4} sec".format(eTime-sTime))

    # 5) 学習したモデルを用いて予測
    print("loss={0:.3}".format(regression.loss(regression.kernel(myData.xTest),myData.yTest)))

    # 6) 学習・評価データおよび予測結果をプロット
    predict = regression.predict(regression.kernel(myData.xTest))
    myData.plot(predict)
    #myData.plot(predict,isTrainPlot=True)

    # 7) カーネルモデル

# メインの終わり
#-------------------
