"""
@author: LXA
 Created on: 2022 年 4 月 20 日
"""
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib
import platform
import shutil
import time

import DNN_Class_base
import DNN_data
import Load_data2Mat
import saveData
import plotData
import DNN_Log_Print


class MscaleDNN(object):
    def __init__(self, input_dim=4, out_dim=1, hidden_layer=None, Model_name='DNN', name2actIn='relu',
                 name2actHidden='relu', name2actOut='linear', opt2regular_WB='L2', type2numeric='float32',
                 factor2freq=None, sFourier=1.0):
        super(MscaleDNN, self).__init__()
        if 'DNN' == str.upper(Model_name):
            self.DNN = DNN_Class_base.Pure_Dense_Net(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name, actName2in=name2actIn,
                actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric)
        elif 'SCALE_DNN' == str.upper(Model_name) or 'DNN_SCALE' == str.upper(Model_name):
            self.DNN = DNN_Class_base.Dense_ScaleNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name, actName2in=name2actIn,
                actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric)
        elif 'FOURIER_DNN' == str.upper(Model_name) or 'DNN_FOURIERBASE' == str.upper(Model_name):
            self.DNN = DNN_Class_base.Dense_FourierNet(
                indim=input_dim, outdim=out_dim, hidden_units=hidden_layer, name2Model=Model_name, actName2in=name2actIn,
                actName=name2actHidden, actName2out=name2actOut, type2float=type2numeric)

        if type2numeric == 'float32':
            self.float_type = tf.float32
        elif type2numeric == 'float64':
            self.float_type = tf.float64
        elif type2numeric == 'float16':
            self.float_type = tf.float16

        self.input_dim = input_dim
        self.factor2freq = factor2freq
        self.sFourier = sFourier
        self.opt2regular_WB = opt2regular_WB

        if input_dim == 1:
            self.mat2X = tf.constant([[1, 0]], dtype=self.float_type)    # 1 行 2 列
            self.mat2T = tf.constant([[0, 1]], dtype=self.float_type)    # 1 行 2 列
        elif input_dim == 2:
            self.mat2X = tf.constant([[1, 0, 0],
                                      [0, 1, 0]], dtype=self.float_type)    # 2 行 3 列
            self.mat2T = tf.constant([[0, 0, 1]], dtype=self.float_type)    # 1 行 3 列
        elif input_dim == 3:
            self.mat2X = tf.constant([[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0]], dtype=self.float_type)  # 3 行 4 列
            self.mat2T = tf.constant([[0, 0, 0, 1]], dtype=self.float_type)  # 1 行 4 列
        elif input_dim == 4:
            self.mat2X = tf.constant([[1, 0, 0, 0, 0],
                                      [0, 1, 0, 0, 0],
                                      [0, 0, 1, 0, 0],
                                      [0, 0, 0, 0, 1]], dtype=self.float_type)  # 4 行 5 列
            self.mat2T = tf.constant([[0, 0, 0, 0, 1]], dtype=self.float_type)     # 1 行 5 列

    def loss2HeatEq(self, X=None, t=None, fside=None, if_lambda2fside=True, loss_type='l2_loss', alpha=1.0):
        # Heat Equation: Ut = alpha*Uxx+f(x,t)
        assert (X is not None)
        assert (t is not None)
        assert (fside is not None)

        shape2X = X.get_shape().as_list()
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)

        if if_lambda2fside:
            if self.input_dim == 1:
                assert (shape2X[-1] == 1)
                force_side = fside(X, t)
            elif self.input_dim == 2:
                assert (shape2X[-1] == 2)
                X1 = tf.reshape(X[:, 0], shape=[-1, 1])
                X2 = tf.reshape(X[:, 1], shape=[-1, 1])
                force_side = fside(X1, X2, t)
            elif self.input_dim == 3:
                assert (shape2X[-1] == 3)
                X1 = tf.reshape(X[:, 0], shape=[-1, 1])
                X2 = tf.reshape(X[:, 1], shape=[-1, 1])
                X3 = tf.reshape(X[:, 2], shape=[-1, 1])
                force_side = fside(X1, X2, X3, t)
            elif self.input_dim == 4:
                assert (shape2X[-1] == 4)
                X1 = tf.reshape(X[:, 0], shape=[-1, 1])
                X2 = tf.reshape(X[:, 1], shape=[-1, 1])
                X3 = tf.reshape(X[:, 2], shape=[-1, 1])
                X4 = tf.reshape(X[:, 3], shape=[-1, 1])
                force_side = fside(X1, X2, X3, X4, t)
            elif self.input_dim == 5:
                assert (shape2X[-1] == 5)
                X1 = tf.reshape(X[:, 0], shape=[-1, 1])
                X2 = tf.reshape(X[:, 1], shape=[-1, 1])
                X3 = tf.reshape(X[:, 2], shape=[-1, 1])
                X4 = tf.reshape(X[:, 3], shape=[-1, 1])
                X5 = tf.reshape(X[:, 4], shape=[-1, 1])
                force_side = fside(X1, X2, X3, X4, X5, t)
        else:
            force_side = fside

        XT = tf.matmul(X, self.mat2X) + tf.matmul(t, self.mat2T)

        UNN = self.DNN(XT, scale=self.factor2freq, sFourier=self.sFourier)
        dUNN2X = tf.gradients(UNN, X)[0]
        dUNN2T = tf.gradients(UNN, t)[0]

        if str.lower(loss_type) == 'l2_loss':
            # -Laplace U=f --> -Laplace U - f --> -(Laplace U + f)
            if self.input_dim == 1:
                dUNNxx = tf.gradients(dUNN2X, X)[0]
                loss_it_L2 = dUNN2T - dUNNxx - tf.reshape(force_side, shape=[-1, 1])
            elif self.input_dim == 2:
                dUNN_x1 = tf.gather(dUNN2X, [0], axis=-1)
                dUNN_x2 = tf.gather(dUNN2X, [1], axis=-1)
                dUNNx1x1x2 = tf.gradients(dUNN_x1, X)[0]
                dUNNx2x1x2 = tf.gradients(dUNN_x2, X)[0]
                dUNNx1x1 = tf.gather(dUNNx1x1x2, [0], axis=-1)
                dUNNx2x2 = tf.gather(dUNNx2x1x2, [1], axis=-1)
                loss_it_L2 = dUNN2T - tf.add(dUNNx1x1, dUNNx2x2) - tf.reshape(force_side, shape=[-1, 1])
            elif self.input_dim == 3:
                dUNN_x1 = tf.gather(dUNN2X, [0], axis=-1)
                dUNN_x2 = tf.gather(dUNN2X, [1], axis=-1)
                dUNN_x3 = tf.gather(dUNN2X, [2], axis=-1)
                dUNNx1x1x2x3 = tf.gradients(dUNN_x1, X)[0]
                dUNNx2x1x2x3 = tf.gradients(dUNN_x2, X)[0]
                dUNNx3x1x2x3 = tf.gradients(dUNN_x3, X)[0]
                dUNNx1x1 = tf.gather(dUNNx1x1x2x3, [0], axis=-1)
                dUNNx2x2 = tf.gather(dUNNx2x1x2x3, [1], axis=-1)
                dUNNx3x3 = tf.gather(dUNNx3x1x2x3, [2], axis=-1)
                loss_it_L2 = dUNN2T - (dUNNx1x1 + dUNNx2x2 + dUNNx3x3) - tf.reshape(force_side, shape=[-1, 1])
            elif self.input_dim == 4:
                dUNN_x1 = tf.gather(dUNN2X, [0], axis=-1)
                dUNN_x2 = tf.gather(dUNN2X, [1], axis=-1)
                dUNN_x3 = tf.gather(dUNN2X, [2], axis=-1)
                dUNN_x4 = tf.gather(dUNN2X, [3], axis=-1)
                dUNNx1x1x2x3x4 = tf.gradients(dUNN_x1, X)[0]
                dUNNx2x1x2x3x4 = tf.gradients(dUNN_x2, X)[0]
                dUNNx3x1x2x3x4 = tf.gradients(dUNN_x3, X)[0]
                dUNNx4x1x2x3x4 = tf.gradients(dUNN_x4, X)[0]
                dUNNx1x1 = tf.gather(dUNNx1x1x2x3x4, [0], axis=-1)
                dUNNx2x2 = tf.gather(dUNNx2x1x2x3x4, [1], axis=-1)
                dUNNx3x3 = tf.gather(dUNNx3x1x2x3x4, [2], axis=-1)
                dUNNx4x4 = tf.gather(dUNNx4x1x2x3x4, [3], axis=-1)
                loss_it_L2 = dUNN2T - (dUNNx1x1 + dUNNx2x2 + dUNNx3x3 + dUNNx4x4) - tf.reshape(force_side, shape=[-1, 1])
            elif self.input_dim == 5:
                dUNN_x1 = tf.gather(dUNN2X, [0], axis=-1)
                dUNN_x2 = tf.gather(dUNN2X, [1], axis=-1)
                dUNN_x3 = tf.gather(dUNN2X, [2], axis=-1)
                dUNN_x4 = tf.gather(dUNN2X, [3], axis=-1)
                dUNN_x5 = tf.gather(dUNN2X, [4], axis=-1)
                dUNNx1x1x2x3x4x5 = tf.gradients(dUNN_x1, X)[0]
                dUNNx2x1x2x3x4x5 = tf.gradients(dUNN_x2, X)[0]
                dUNNx3x1x2x3x4x5 = tf.gradients(dUNN_x3, X)[0]
                dUNNx4x1x2x3x4x5 = tf.gradients(dUNN_x4, X)[0]
                dUNNx5x1x2x3x4x5 = tf.gradients(dUNN_x5, X)[0]
                dUNNx1x1 = tf.gather(dUNNx1x1x2x3x4x5, [0], axis=-1)
                dUNNx2x2 = tf.gather(dUNNx2x1x2x3x4x5, [1], axis=-1)
                dUNNx3x3 = tf.gather(dUNNx3x1x2x3x4x5, [2], axis=-1)
                dUNNx4x4 = tf.gather(dUNNx4x1x2x3x4x5, [3], axis=-1)
                dUNNx5x5 = tf.gather(dUNNx5x1x2x3x4x5, [4], axis=-1)
                loss_it_L2 = dUNN2T - (dUNNx1x1 + dUNNx2x2 + dUNNx3x3 + dUNNx4x4 + dUNNx5x5) - tf.reshape(force_side, shape=[-1, 1])
            square_loss_it = tf.square(loss_it_L2)
            loss_it = tf.reduce_mean(square_loss_it)
        return UNN, loss_it

    def loss2Navier_Stokes(self, X=None, t=None, Aeps=None, if_lambda2Aeps=True, fside=None, if_lambda2fside=True,
                           loss_type='l2_loss', p_index=2):
        assert (X is not None)
        assert (t is not None)
        assert (fside is not None)
        assert (Aeps is not None)

        shape2X = X.get_shape().as_list()
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)

        if self.input_dim == 1:
            assert (shape2X[-1] == 1)

            if if_lambda2Aeps:
                a_eps = Aeps(X)  # * 行 1 列
            else:
                a_eps = Aeps

            if if_lambda2fside:
                force_side = fside(X)
            else:
                force_side = fside
        elif self.input_dim == 2:
            assert (shape2X[-1] == 2)
            X1 = tf.reshape(X[:, 0], shape=[-1, 1])
            X2 = tf.reshape(X[:, 1], shape=[-1, 1])

            if if_lambda2Aeps:
                a_eps = Aeps(X1, X2)  # * 行 1 列
            else:
                a_eps = Aeps

            if if_lambda2fside:
                force_side = fside(X1, X2)
            else:
                force_side = fside
        elif self.input_dim == 3:
            assert (shape2X[-1] == 3)
            X1 = tf.reshape(X[:, 0], shape=[-1, 1])
            X2 = tf.reshape(X[:, 1], shape=[-1, 1])
            X3 = tf.reshape(X[:, 2], shape=[-1, 1])

            if if_lambda2Aeps:
                a_eps = Aeps(X1, X2, X3)  # * 行 1 列
            else:
                a_eps = Aeps
            if if_lambda2fside:
                force_side = fside(X1, X2, X3)
            else:
                force_side = fside
        elif self.input_dim == 4:
            assert (shape2X[-1] == 4)
            X1 = tf.reshape(X[:, 0], shape=[-1, 1])
            X2 = tf.reshape(X[:, 1], shape=[-1, 1])
            X3 = tf.reshape(X[:, 2], shape=[-1, 1])
            X4 = tf.reshape(X[:, 3], shape=[-1, 1])

            if if_lambda2Aeps:
                a_eps = Aeps(X1, X2, X3, X4)  # * 行 1 列
            else:
                a_eps = Aeps

            if if_lambda2fside:
                force_side = fside(X1, X2, X3, X4)
            else:
                force_side = fside
        elif self.input_dim == 5:
            assert (shape2X[-1] == 5)
            X1 = tf.reshape(X[:, 0], shape=[-1, 1])
            X2 = tf.reshape(X[:, 1], shape=[-1, 1])
            X3 = tf.reshape(X[:, 2], shape=[-1, 1])
            X4 = tf.reshape(X[:, 3], shape=[-1, 1])
            X5 = tf.reshape(X[:, 4], shape=[-1, 1])

            if if_lambda2Aeps:
                a_eps = Aeps(X1, X2, X3, X4, X5)  # * 行 1 列
            else:
                a_eps = Aeps
            if if_lambda2fside:
                force_side = fside(X1, X2, X3, X4, X5)
            else:
                force_side = fside

        UNN = self.DNN(X, scale=self.factor2freq, sFourier=self.sFourier)
        dUNN = tf.gradients(UNN, X)[0]
        # 变分形式的loss of interior，训练得到的 UNN 是 * 行 1 列
        if str.lower(loss_type) == 'ritz_loss' or str.lower(loss_type) == 'variational_loss':
            dUNN_Norm = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(dUNN), axis=-1)), shape=[-1, 1])  # 按行求和
            AdUNN_pNorm = tf.multiply(a_eps, tf.pow(dUNN_Norm, p_index))
            loss_it_ritz = (1.0/p_index)*AdUNN_pNorm-tf.multiply(tf.reshape(force_side, shape=[-1, 1]), UNN)
            loss_it = tf.reduce_mean(loss_it_ritz)
        return UNN, loss_it

    # Dirichlet 边界条件
    def loss_bd2Dirichlet(self, X_bd=None, Ubd_exact=None, if_lambda2Ubd=True):
        assert (X_bd is not None)
        assert (Ubd_exact is not None)

        shape2X = X_bd.get_shape().as_list()
        lenght2X_shape = len(shape2X)
        assert (lenght2X_shape == 2)

        if if_lambda2Ubd:
            if self.input_dim == 1:
                Ubd = Ubd_exact(X_bd)
            elif self.input_dim == 2:
                X1_bd = tf.reshape(X_bd[:, 0], shape=[-1, 1])
                X2_bd = tf.reshape(X_bd[:, 1], shape=[-1, 1])
                Ubd = Ubd_exact(X1_bd, X2_bd)
            elif self.input_dim == 3:
                X1_bd = tf.reshape(X_bd[:, 0], shape=[-1, 1])
                X2_bd = tf.reshape(X_bd[:, 1], shape=[-1, 1])
                X3_bd = tf.reshape(X_bd[:, 2], shape=[-1, 1])
                Ubd = Ubd_exact(X1_bd, X2_bd, X3_bd)
            elif self.input_dim == 4:
                X1_bd = tf.reshape(X_bd[:, 0], shape=[-1, 1])
                X2_bd = tf.reshape(X_bd[:, 1], shape=[-1, 1])
                X3_bd = tf.reshape(X_bd[:, 2], shape=[-1, 1])
                X4_bd = tf.reshape(X_bd[:, 3], shape=[-1, 1])
                Ubd = Ubd_exact(X1_bd, X2_bd, X3_bd, X4_bd)
            elif self.input_dim == 5:
                X1_bd = tf.reshape(X_bd[:, 0], shape=[-1, 1])
                X2_bd = tf.reshape(X_bd[:, 1], shape=[-1, 1])
                X3_bd = tf.reshape(X_bd[:, 2], shape=[-1, 1])
                X4_bd = tf.reshape(X_bd[:, 3], shape=[-1, 1])
                X5_bd = tf.reshape(X_bd[:, 4], shape=[-1, 1])
                Ubd = Ubd_exact(X1_bd, X2_bd, X3_bd, X4_bd, X5_bd)
        else:
            Ubd = Ubd_exact

        UNN_bd = self.DNN(X_bd, scale=self.factor2freq, sFourier=self.sFourier)
        loss_bd_square = tf.square(UNN_bd - Ubd)
        loss_bd = tf.reduce_mean(loss_bd_square)
        return loss_bd

    def get_regularSum2WB(self):
        sum2WB = self.DNN.get_regular_sum2WB(self.opt2regular_WB)
        return sum2WB

    def evalue_MscaleDNN(self, X_points=None):
        UNN = self.DNN(X_points, scale=self.factor2freq, sFourier=self.sFourier)
        return UNN
