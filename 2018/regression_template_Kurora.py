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
	#------------------------------------

	#------------------------------------
	# 2) �ŏ������@���p���ă��f���p�����[�^���œK��
	# �i�����̌v�Z��For�����p�����ꍇ�j
	def train(self):
		self.w = np.zeros([self.xDim,1])
		#pdb.set_trace()
		x_d = np.vstack((self.x, np.array([1]*self.dNum)))

		""" ミス
		#x_d = np.insert(self.x, self.dNum, 1, axis=1)
		"""

		x_d_t = x_d.T
		y_d = self.y[np.newaxis]
		#print(y_d.shape)
		sigma_left = np.zeros((x_d.shape[0], x_d_t.shape[1]))
		sigma_right = np.zeros((y_d.shape[0], x_d_t.shape[1]))
		#"""
		for i in range(self.dNum):
			sigma_left += np.dot(x_d[0:,i][np.newaxis].T,x_d_t[i, 0:][np.newaxis])
			sigma_right += y_d[0:, i]*x_d_t[i,0:]
			#print(sigma_left)
		self.w = np.dot(np.linalg.inv(sigma_left), sigma_right.T)
		#"""
	#------------------------------------

	#------------------------------------
	# 2) �ŏ������@���p���ă��f���p�����[�^���œK���i�s�񉉎Z�ɂ��荂�����j
	def trainMat(self):
		self.w = np.zeros([self.xDim,1])
		x_d = np.vstack((self.x, np.array([1]*self.dNum)))

		""" ミス
		#x_d = np.insert(self.x, self.dNum, 1, axis=1)
		"""

		x_d_t = x_d.T
		#pdb.set_trace()

		""" ミス
		#sigma_left = np.dot(x_d[0:, :self.dNum], x_d_t[:self.dNum,0:])
		#sigma_right = np.dot(self.yn, x_d_t[:self.dNum,0:])
		"""
		sigma_left = np.dot(x_d, x_d_t)
		sigma_right = np.dot(self.y[np.newaxis], x_d_t)
		self.w = np.dot(np.linalg.inv(sigma_left),sigma_right.T)

	#------------------------------------

	#------------------------------------
	# 3) �\��
	# x: ���̓f�[�^�i���͎��� x �f�[�^���j
	def predict(self,x):
		#pdb.set_trace()
		#x_d = np.vstack((x, np.array([1]*x.shape[1])))
		x_d = x
		#pdb.set_trace()
		y = np.matmul(self.w.T, x_d)
		return y
	#------------------------------------

	#------------------------------------
	# 4) ���摹���̌v�Z
	# x: ���̓f�[�^�i���͎��� x �f�[�^���j
	# y: �o�̓f�[�^�i�f�[�^���j
	def loss(self,x,y):
		N = self.dNum
		#pdb.set_trace()
		num = pow(y[np.newaxis] - self.predict(x),2)
		loss = np.sum(y - num) / N
		return loss
	#------------------------------------

	#------------------------------------
	# 6) 2つのデータ集合間全て組み合わせ距離計算
	# x: 行列 (次元 x データ数)
	# y: 行列 (次元 x データ数)
	def calcDist(self,x,z):
		"""
		x行列とz行列に奥行きを追加する
		"""
		xn = x[np.newaxis]
		zn = z[np.newaxis]
		#pdb.set_trace()
		"""
		x行列とz行列をそれぞれ
		　xデータ数 x zデータ数 x 行列
		の行列変換
		"""
		#pdb.set_trace()
		x_d = np.tile(xn, zn.shape[2]).reshape(xn.shape[1], zn.shape[2], xn.shape[2]).transpose(0,2,1)
		z_d = np.tile(zn, xn.shape[2]).reshape(zn.shape[1], xn.shape[2], zn.shape[2])

		"""
		x行列とz行列の距離を計算する
			x行列とz行列の差を計算
			計算結果の行列 dist_d を奥行方向に足す
			奥行き方向に足し算した行列 dist を2乗する
		"""
		dist_d = x_d - z_d
		#pdb.set_trace()
		dist = np.sum(dist_d, axis=0)
		dist = pow(dist,2)

		return dist

		"""  miss
		dist = np.zeros((1,z_d.shape[1],z_d.shape[2]))
		#pdb.set_trace()
		for d in dist_d:
			dist += d
		"""

		"""  miss
		x_d = np.tile(xn,zn.shape[xn.shape[-2]]).reshape(xn.shape[xn.shape[-2]-1],zn.shape[xn.shape[-2]],xn.shape[xn.shape[-2]]).transpose(0,2,1)
		z_d = np.tile(zn,xn.shape[xn.shape[-2]]).reshape(zn.shape[xn.shape[-2]-1],xn.shape[xn.shape[-2]],zn.shape[xn.shape[-2]])
		"""
		"""  miss
		x_d = np.tile(x,z.shape[1]).reshape(x.shape[0],z.shape[1],x.shape[1]).transpose(0,2,1)
		z_d = np.tile(z,x.shape[1]).reshape(z.shape[0],x.shape[1],z.shape[1])
		"""
		"""	 miss
		x_d = np.tile(x,(z.shape[0],1)).T
		z_d = np.tile(z,(x.shape[0],1))
		"""

	def kernel(self,x):
		k_d = np.exp(-1 * self.calcDist(self.x, x) / ( 2*pow(self.kernelParam,2) ))
		#k = np.insert(k_d, k_d.shape[1], 1, axis=1)
		k = np.vstack((k_d, np.array([1]*k_d.shape[1])))
		#pdb.set_trace()
		return k

	def trainMatKernel(self):
		lamda = 0.1
		"""
		sigma0 : 重みwの左側のΣ(逆行列)
		sigma1 : 重みwの右側のΣ
		"""
		sigma0 = np.dot(self.kernel(self.x), self.kernel(self.x).transpose())
		#pdb.set_trace()
		"""
		逆行列求める際に単位行列を足す
		"""
		sigma0 = np.linalg.inv(sigma0 + lamda * np.eye(sigma0.shape[1]))
		sigma1 = np.dot(self.y[np.newaxis], self.kernel(self.x).transpose())
		self.w = np.dot(sigma0,sigma1.transpose())
		#pdb.set_trace()
		#dist = np.linalg.norm(x_d - z_d)
# �N���X�̒��`�I����
#-------------------

#-------------------
# ���C���̎n�܂�
if __name__ == "__main__":

	# 1) �w�K���͎�����2�̏ꍇ�̃f�[�^�[����
	#myData = rg.artificial(200,100, dataType="1D")
	myData = rg.artificial(200,100, dataType="2D",isNonlinear=True)

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


	sTime = time.time()
	regression.trainMatKernel()
	eTime = time.time()

	print("train with matrix: time={0:.4} sec".format(eTime-sTime))
	#pdb.set_trace()
	# 5) �w�K�������f�����p���ė\��
	#print("loss={0:.3}".format(regression.loss(myData.xTest,myData.yTest)))

	print("loss={0:.3}".format(regression.loss(regression.kernel(myData.xTest),myData.yTest)))

	# 6) �w�K�E�]���f�[�^�����ї\�����ʂ��v���b�g
	#predict = regression.predict(myData.xTest)

	predict=regression.predict(regression.kernel(myData.xTest))
	myData.plot(predict[0],isTrainPlot=False)

#���C���̏I����
#-------------------
