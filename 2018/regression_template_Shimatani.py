import numpy as np
import pdb
import time
import regressionData as rg

#-------------------
# クラスの定義始まり
#-------------------
# クラスの定義始まり
class linearRegression():
    #------------------------------------
    # 1) 学習データおよびモデルパラメータの初期化
    # x: 学習入力データ（入力ベクトルの次元数×データ数のnumpy.array）
    # y: 学習出力データ（データ数のnumpy.array）
    # kernelType: カーネルの種類（文字列：gaussian）
    # kernelParam: カーネルのハイパーパラメータ（スカラー）
    def __init__(self, x, y, kernelType="curve", kernelParam=1.0):
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
        N = [np.ones(self.x.shape[1])]
        P = np.zeros([self.xDim+1,self.xDim+1])
        Q = np.zeros(self.xDim+1)
        self.x = np.concatenate([self.x,N],axis=0)
        self.w = np.zeros([self.xDim,1])
        for i in range(self.xDim+1):
            for j in range(self.xDim+1):
                P[i,j] = np.dot(self.x[i,:],self.x[j,:])
            Q[i] = np.dot(self.y,self.x[i,:])
        self.w = np.dot(np.linalg.inv(P) , Q[np.newaxis].T)
            
              
    #------------------------------------

    #------------------------------------
    # 2) 最小二乗法を用いてモデルパラメータを最適化
    # （分子・分母の計算に行列演算を用いた場合）
    def trainMat(self):
        N = [np.ones(self.x.shape[1])]
        self.w = np.zeros([self.xDim,1])
        self.x = np.concatenate([self.x,N],axis=0)
        P = np.dot(self.x,self.x.T)
        Q = np.dot(self.x,self.y[np.newaxis].T)
        self.w = np.dot(np.linalg.inv(P) , Q)
    #------------------------------------
    
    #------------------------------------
    # 3) 予測
    # x: 入力データ（入力次元 x データ数）
    def predict(self,x):   
        y = []
        N = [np.ones(x.shape[1])]
        x = np.concatenate([x,N],axis = 0)
        y = np.dot( self.w.T , x )
        return y[0]
    #------------------------------------

    #------------------------------------
    # 4) 二乗損失の計算
    # x: 入力データ（入力次元 x データ数）
    # y: 出力データ（データ数）
    def loss(self,x,y):
        if self.kernelType == "linear" :
            F = self.predict(x)
        elif self.kernelType == "curve" :
            F = self.kernelpredict(x)
        loss = 0.0
        loss = np.mean(np.power(y-F,2))
        return loss
    
    #------------------------------------
    
    #------------------------------------
    # 5) カーネルの計算
    # x: カーネルを計算する対象の行列（次元 x データ数）
    def kernel(self,x):
        #【self.xの各データ点xiと行列xの各データ点xjと間のカーネル値k(xi,xj)を各要素に持つグラム行列を計算】
        K = self.calcDist(self.x,x)
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                K[i][j] = np.exp(K[i][j] * -1 / (2 * self.kernelParam**2))
        return K
    #------------------------------------

    #------------------------------------
    # 6) 2つのデータ集合間の全ての組み合わせの距離の計算
    # x: 行列（次元 x データ数）
    # z: 行列（次元 x データ数）
    def calcDist(self,x,z):
        #【行列xのデータ点x1, x2, ..., xNと、行列zのデータ点z1, z2, ..., zMとの間のMxN個の距離を計算】
        dist = np.zeros((x.shape[1],z.shape[1]))
        xd = np.resize(x.T,(z.shape[1],x.shape[1],x.shape[0])).transpose(1,0,2)
        zd = np.resize(z.T,(x.shape[1],z.shape[1],z.shape[0]))
        dist = np.sum((xd-zd)**2,axis=2)
        return dist
    #------------------------------------

    def trainMatKernel(self):
        K = self.kernel(self.x)
        P = np.dot(K,K.T) + 0.01*np.eye(len(K))
        Q = np.dot(K,self.y[np.newaxis].T)
        self.w = np.dot(np.linalg.inv(P),Q)
        
    def kernelpredict(self,x):
        #self.x = np.reshape(np.arange(-2,8,0.05),(1,200))
        K = self.kernel(x)
        y = np.dot(self.w.T, K)
        return y[0]
# クラスの定義終わり
#-------------------

#-------------------
# メインの始まり
if __name__ == "__main__":
    # 1) 学習入力次元が2の場合のデーター生成
    myData = rg.artificial(200,100, dataType="1D",isNonlinear=True)
    
    # 2) 線形回帰モデル
    regression = linearRegression(myData.xTrain,myData.yTrain)
    """
    # 3) 学習（For文版）
    sTime = time.time()
    regression.train()
    eTime = time.time()
    print("train with for-loop: time={0:.4} sec".format(eTime-sTime))
    """
    
    # 4) 学習（行列版）
    # 5) 学習したモデルを用いて予測
    # 6) 学習・評価データおよび予測結果をプロット
    if regression.kernelType == "linear" :
        sTime = time.time()
        regression.trainMat()
        eTime = time.time()
        print("train with matrix: time={0:.4} sec".format(eTime-sTime))
        print("loss={0:.3}".format(regression.loss(myData.xTest,myData.yTest)))
        predict = regression.predict(myData.xTest)
        
    elif regression.kernelType == "curve" :
        sTime = time.time()
        regression.trainMatKernel()
        eTime = time.time()
        print("train with matrix: time={0:.4} sec".format(eTime-sTime))
        print("loss={0:.3}".format(regression.loss(myData.xTest,myData.yTest)))
        predict = regression.kernelpredict(myData.xTest)
  
    myData.plot(predict)
#メインの終わり
#-------------------
