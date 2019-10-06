# -*- coding: utf-8 -*-

import numpy as np
import regressionData as rg
import time
import pdb

#-------------------
# クラスの定義始まり
class linearRegression():
    # ------------------------------------
    # 1) 学習データおよびモデルパラメータの初期化
    # x: 学習入力データ（入力ベクトルの次元数×データ数のnumpy.array）
    # y: 学習出力データ（データ数のnumpy.array）
    # kernelType: カーネルの種類（文字列：gaussian）
    # kernelParam: カーネルのハイパーパラメータ（スカラー）
    def __init__(self, x, y, kernelType="linear", kernelParam=1.0):
        # 学習データの設定
        # pdb.set_trace()
        self.x = x
        self.y = y
        self.xDim = x.shape[0]
        self.dNum = x.shape[1]

        # カーネルの設定
        self.kernelType = kernelType
        self.kernelParam = kernelParam
    # ------------------------------------

    # ------------------------------------
    # 2) 最小二乗法を用いてモデルパラメータを最適化
    # （分子・分母の計算にFor文を用いた場合）
    def train(self):
        one = np.ones((1,self.dNum))
        x = np.append(self.x,one,axis=0)

        y = self.y
        d = x.shape[0]
        sum1 = np.matmul(x[:, 0].reshape(d, 1), x[:, 0].reshape(1, d))
        sum2 = y[0] * x[:, 0]

        for i in np.arange(1, x.shape[1]):
            sum1 = sum1 + np.matmul(x[:, i].reshape(d, 1), x[:, i].reshape(1, d))
            sum2 = sum2 + y[i] * x[:, i]

        self.w = np.matmul(np.linalg.inv(sum1), sum2)
        print("train:",self.w)

    # ------------------------------------

    def trainMatKernel(self):
        one = np.ones((1,self.dNum))
        k = np.append(self.kernel(self.x),one,axis=0)
        # k = self.kernel(self.x)
        y = self.y
        ramda = 0.01

        sum1 = np.matmul(k, k.T) + ramda * np.identity(self.dNum + 1)
        sum2 = np.matmul(k, y.T)
        pdb.set_trace()
        self.w = (np.matmul(np.linalg.inv(sum1), sum2)).reshape(1,self.dNum+1)
        print("trainMatKernel:", self.w)

    # ------------------------------------
    # 2) 最小二乗法を用いてモデルパラメータを最適化
    # （分子・分母の計算に行列演算を用いた場合）
    def trainMat(self):
        one = np.ones((1,self.dNum))
        x = np.append(self.x,one,axis=0)
        y = self.y
        sum1 = np.matmul(x, x.T)
        sum2 = np.matmul(x, y.T)

        self.w = np.matmul(np.linalg.inv(sum1), sum2)
        print("trainMat:",self.w)
    # ------------------------------------

    # ------------------------------------
    # 3) 予測
    # x: 入力データ（入力次元 x データ数
    def predict(self,x):
        # one = np.ones((1, x.shape[1]))
        # x = np.append(x, one, axis=0)
        # y = np.matmul(self.w.T,x)

        # pdb.set_trace()
        one = np.ones((1, x.shape[1]))
        k = np.append(self.kernel(x), one, axis=0)
        pdb.set_trace()

        y = np.matmul(self.w,k)
        pdb.set_trace()
        return y[0]
    # ------------------------------------

    # ------------------------------------
    # 4) 二乗損失の計算
    # x: 入力データ（入力次元 x データ数）
    # y: 出力データ（データ数）
    def loss(self,x,y):
        f = y - self.predict(x)
        loss = np.sum(f ** 2)/self.dNum

        return loss
    # ------------------------------------


    def kernel(self,x):
        # pdb.set_trace()
        phi = self.kernelParam ** 2

        K = np.exp(self.calcDist(self.x,x)/(-2*phi))

        return K

    def calcDist(self,x,z):
        x1 = x
        x2 = z

        n = x1.shape[1]
        m = x2.shape[1]
        #
        # x1 = x1.reshape(self.xDim,n1)
        # x2 = x2.reshape(self.xDim,m1)
        # pdb.set_trace()
        X1 = np.tile(x1.T[:,np.newaxis],(1,1,1))
        X1 = np.tile(X1, (1, m,1))

        X2 = np.tile(x2.T,(1,1,1))
        X2 = np.tile(X2, (n, 1,1))
        

        Y = (X1 - X2) ** 2
        Y = np.sum(Y,axis=2)

        return Y

# クラスの定義終わり
# ------------------

#-------------------
# メインの始まり
if __name__ == "__main__":

    # 1) 学習入力次元が2の場合のデーター生成
    # myData = rg.artificial(200,100, dataType="1D",isNonlinear=True)
    myData = rg.artificial(200,100, dataType="1D",isNonlinear=True)

    # 2) 線形回帰モデル
    # regression = linearRegression(myData.xTrain,myData.yTrain)
    regression = linearRegression(myData.xTrain,myData.yTrain,kernelType="gaussian",kernelParam=1)

    # pdb.set_trace()

    # # 3) 学習（For文版）
    # sTime = time.time()
    # regression.train()
    # eTime = time.time()
    # print("train with for-loop: time={0:.4} sec".format(eTime-sTime))

    # 4) 学習（行列版）
    # sTime = time.time()
    # regression.trainMat()
    # eTime = time.time()
    # print("train with matrix: time={0:.4} sec".format(eTime-sTime))

    sTime = time.time()
    regression.trainMatKernel()
    eTime = time.time()
    print("train with matrix: time={0:.4} sec".format(eTime-sTime))

    # 5) 学習したモデルを用いて予測
    print("loss={0:.3}".format(regression.loss(myData.xTest,myData.yTest)))

    # 6) 学習・評価データおよび予測結果をプロット
    predict = regression.predict(myData.xTest)
    myData.plot(predict)
    
#メインの終わり
#-------------------
    