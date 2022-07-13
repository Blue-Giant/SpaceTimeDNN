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

        self.factor2freq = factor2freq
        self.opt2regular_WB = opt2regular_WB
        self.sFourier = sFourier

        self.mat2XY = tf.constant([[1, 0, 0],
                                  [0, 1, 0]], dtype=self.float_type)  # 2 行 3 列
        self.mat2T = tf.constant([[0, 0, 1]], dtype=self.float_type)  # 1 行 3 列

    def loss2HeatEq(self, XY=None, t=None, fside=None, if_lambda2fside=True, loss_type='l2_loss', alpha=1.0):
        # Heat Equation: Ut - alpha*LaplaceU = f(x,t)
        assert (XY is not None)
        assert (t is not None)
        assert (fside is not None)

        shape2XY = XY.get_shape().as_list()
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)
        X = tf.reshape(XY[:, 0], shape=[-1, 1])
        Y = tf.reshape(XY[:, 1], shape=[-1, 1])

        if if_lambda2fside:
            force_side = fside(X, Y, t)
        else:
            force_side = fside

        XYT = tf.matmul(XY, self.mat2XY) + tf.matmul(t, self.mat2T)
        UNN = self.DNN(XYT, scale=self.factor2freq, sFourier=self.sFourier)
        dUNN2XY = tf.gradients(UNN, XY)[0]                                       # * 行 2 列
        dUNN2t = tf.gradients(UNN, t)[0]                                         # * 行 2 列

        if str.lower(loss_type) == 'l2_loss':
            dUNN_x = tf.gather(dUNN2XY, [0], axis=-1)
            dUNN_y = tf.gather(dUNN2XY, [1], axis=-1)
            dUNNxxy = tf.gradients(dUNN_x, XY)[0]
            dUNNyxy = tf.gradients(dUNN_y, XY)[0]
            dUNNxx = tf.gather(dUNNxxy, [0], axis=-1)
            dUNNyy = tf.gather(dUNNyxy, [1], axis=-1)
            # Ut - alpha*Laplace U = f
            loss_it_L2 = dUNN2t - alpha*tf.add(dUNNxx, dUNNyy) - tf.reshape(force_side, shape=[-1, 1])
            square_loss_it = tf.square(loss_it_L2)
            loss_it = tf.reduce_mean(square_loss_it)
        return UNN, loss_it

    def loss2WaveEq(self, XY=None, t=None, fside=None, if_lambda2fside=True, loss_type='l2_loss', alpha=1.0):
        # Heat Equation: Utt - alpha*Laplace U = f(x,t)
        assert (XY is not None)
        assert (t is not None)
        assert (fside is not None)

        shape2XY = XY.get_shape().as_list()
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)

        X = tf.reshape(XY[:, 0], shape=[-1, 1])
        Y = tf.reshape(XY[:, 1], shape=[-1, 1])

        if if_lambda2fside:
            force_side = fside(X, Y,  t)
        else:
            force_side = fside

        XYT = tf.matmul(XY, self.mat2XY) + tf.matmul(t, self.mat2T)

        UNN = self.DNN(XYT, scale=self.factor2freq, sFourier=self.sFourier)
        dUNN2XY = tf.gradients(UNN, X)[0]
        dUNN2t = tf.gradients(UNN, t)[0]

        if str.lower(loss_type) == 'l2_loss':
            dUNN_x = tf.gather(dUNN2XY, [0], axis=-1)
            dUNN_y = tf.gather(dUNN2XY, [1], axis=-1)
            dUNNxxy = tf.gradients(dUNN_x, XY)[0]
            dUNNyxy = tf.gradients(dUNN_y, XY)[0]
            dUNNxx = tf.gather(dUNNxxy, [0], axis=-1)
            dUNNyy = tf.gather(dUNNyxy, [1], axis=-1)
            dUNNtt = tf.gradients(dUNN2t, t)[0]
            loss_it_L2 = dUNNtt - alpha * (dUNNxx + dUNNyy) - tf.reshape(force_side, shape=[-1, 1])
            square_loss_it = tf.square(loss_it_L2)
            loss_it = tf.reduce_mean(square_loss_it)
        return UNN, loss_it

    def loss2Burges(self, XY=None, t=None, loss_type='l2_loss', nu_coef=None, if_lambda2nu=False):
        # Ut + U*(Ux + Uy) = V Laplace U   在 Omega x T 上, Omega=[-1,1], t>0
        assert (XY is not None)
        assert (t is not None)

        shape2XY = XY.get_shape().as_list()
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)

        X = tf.reshape(XY[:, 0], shape=[-1, 1])
        Y = tf.reshape(XY[:, 1], shape=[-1, 1])

        if if_lambda2nu:
            nu0 = nu_coef(X, Y, t)
        else:
            nu0 = nu_coef

        XYT = tf.matmul(XY, self.mat2XY) + tf.matmul(t, self.mat2T)

        UNN = self.DNN(XYT, scale=self.factor2freq, sFourier=self.sFourier)
        dUNN2XY = tf.gradients(UNN, X)[0]
        dUNN2t = tf.gradients(UNN, t)[0]

        dUNN_x = tf.gather(dUNN2XY, [0], axis=-1)
        dUNN_y = tf.gather(dUNN2XY, [1], axis=-1)

        dUxAdddUy = dUNN_x+dUNN_y

        if str.lower(loss_type) == 'l2_loss':
            dUNNxxy = tf.gradients(dUNN_x, XY)[0]
            dUNNyxy = tf.gradients(dUNN_y, XY)[0]
            dUNNxx = tf.gather(dUNNxxy, [0], axis=-1)
            dUNNyy = tf.gather(dUNNyxy, [1], axis=-1)
            UdUxAdddUy = tf.multiply(UNN, dUxAdddUy)
            LaplaceU = dUNNxx + dUNNyy
            nuLaplaceU = tf.multiply(nu0, LaplaceU)
            # Ut + U*(Ux + Uy) = v(Uxx Uyy)
            loss_it_L2 = dUNN2t + UdUxAdddUy - nuLaplaceU
            square_loss_it = tf.square(loss_it_L2)
            loss_it = tf.reduce_mean(square_loss_it)
        return UNN, loss_it

    def loss2Convection(self, XY=None, t=None, fside=None, if_lambda2fside=True, p_coef=None, if_lambda2p=True,
                        loss_type='l2_loss', alpha=1.0):
        # Convection Equation: Ut + P(x,t)*(Ux+Uy) = v*(Uxx + Uyy) + f(x,t)
        assert (XY is not None)
        assert (t is not None)
        assert (fside is not None)
        assert (p_coef is not None)

        shape2XY = XY.get_shape().as_list()
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)

        X = tf.reshape(XY[:, 0], shape=[-1, 1])
        Y = tf.reshape(XY[:, 1], shape=[-1, 1])

        if if_lambda2fside:
            force_side = fside(X, Y, t)
        else:
            force_side = fside

        if if_lambda2p:
            coef2p = p_coef(X, Y, t)
        else:
            coef2p = p_coef

        XYT = tf.matmul(XY, self.mat2XY) + tf.matmul(t, self.mat2T)

        UNN = self.DNN(XYT, scale=self.factor2freq, sFourier=self.sFourier)
        dUNN2XY = tf.gradients(UNN, X)[0]
        dUNN2t = tf.gradients(UNN, t)[0]

        dUNN_x = tf.gather(dUNN2XY, [0], axis=-1)
        dUNN_y = tf.gather(dUNN2XY, [1], axis=-1)

        dUxAdddUy = dUNN_x + dUNN_y

        if str.lower(loss_type) == 'l2_loss':
            dUNNxxy = tf.gradients(dUNN_x, XY)[0]
            dUNNyxy = tf.gradients(dUNN_y, XY)[0]
            dUNNxx = tf.gather(dUNNxxy, [0], axis=-1)
            dUNNyy = tf.gather(dUNNyxy, [1], axis=-1)
            LaplaceU = dUNNxx + dUNNyy
            pdUxAdddUy = tf.multiply(coef2p, dUxAdddUy)
            loss_it_L2 = dUNN2t + pdUxAdddUy - alpha*LaplaceU - tf.reshape(force_side, shape=[-1, 1])
            square_loss_it = tf.square(loss_it_L2)
            loss_it = tf.reduce_mean(square_loss_it)
        return UNN, loss_it

    # 0 阶导数边界条件(Dirichlet 边界)
    def loss_bd2dirichlet(self, XY_bd=None, t=None, Ubd_exact=None, if_lambda2Ubd=True):
        assert (XY_bd is not None)
        assert (Ubd_exact is not None)

        shape2XY = np.shape(XY_bd)
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)
        X_bd = tf.reshape(XY_bd[:, 0], shape=[-1, 1])
        Y_bd = tf.reshape(XY_bd[:, 1], shape=[-1, 1])

        if if_lambda2Ubd:
            Ubd = Ubd_exact(X_bd, Y_bd, t)
        else:
            Ubd = Ubd_exact

        XYT = tf.matmul(XY_bd, self.mat2XY) + tf.matmul(t, self.mat2T)
        UNN_bd = self.DNN(XYT, scale=self.factor2freq, sFourier=self.sFourier)
        loss_bd_square = tf.square(UNN_bd - Ubd)
        loss_bd = tf.reduce_mean(loss_bd_square)
        return loss_bd

    def loss2Init(self, XY=None, tinit=None, Uinit_exact=None, if_lambda2Uinit=True):
        assert (XY is not None)
        assert (tinit is not None)
        assert (Uinit_exact is not None)

        shape2XY = XY.get_shape().as_list()
        lenght2XY_shape = len(shape2XY)
        assert (lenght2XY_shape == 2)
        assert (shape2XY[-1] == 2)

        X = tf.reshape(XY[:, 0], shape=[-1, 1])
        Y = tf.reshape(XY[:, 1], shape=[-1, 1])

        if if_lambda2Uinit:
            Uinit = Uinit_exact(X, Y, tinit)
        else:
            Uinit = Uinit_exact

        XYT = tf.matmul(XY, self.mat2XY) + tf.matmul(tinit, self.mat2T)
        UNN_init = self.DNN(XYT, scale=self.factor2freq, sFourier=self.sFourier)
        loss_init_square = tf.square(UNN_init - Uinit)
        loss_init = tf.reduce_mean(loss_init_square)
        return loss_init

    def get_regularSum2WB(self):
        sum2WB = self.DNN.get_regular_sum2WB(self.opt2regular_WB)
        return sum2WB

    def evalue_MscaleDNN(self, XY_points=None, t_points=None):
        assert (XY_points is not None)
        assert (t_points is not None)
        XYT = tf.matmul(XY_points, self.mat2XY) + tf.matmul(t_points, self.mat2T)
        UNN = self.DNN(XYT, scale=self.factor2freq, sFourier=self.sFourier)
        return UNN


def solve_PDE(R):
    log_out_path = R['FolderName']        # 将路径从字典 R 中提取出来
    if not os.path.exists(log_out_path):  # 判断路径是否已经存在
        os.mkdir(log_out_path)            # 无 log_out_path 路径，创建一个 log_out_path 路径
    logfile_name = '%s_%s.txt' % ('log2train', R['name2act_hidden'])
    log_fileout = open(os.path.join(log_out_path, logfile_name), 'w')  # 在这个路径下创建并打开一个可写的 log_train.txt文件
    DNN_Log_Print.dictionary2space_time(R, log_fileout)

    # 问题需要的设置
    batchsize_it = R['batch_size2interior']
    batchsize_bd = R['batch_size2boundary']
    batchsize_init = R['batch_size2init']
    batchsize_test = R['batch_size2test']

    bd_penalty_init = R['init_boundary_penalty']           # Regularization parameter for boundary conditions
    init_penalty_init = R['init_init_penalty']  # Regularization parameter for init conditions
    penalty2WB = R['penalty2weight_biases']                # Regularization parameter for weights and biases
    lr_decay = R['learning_rate_decay']
    learning_rate = R['learning_rate']
    act_func = R['name2act_hidden']

    input_dim = R['input_dim']
    out_dim = R['output_dim']

    mesh_number = 2
    region_lb = 0.0
    region_rt = 1.0
    init_time = 0.0
    end_time = 1.0
    if R['PDE_type'] == 'HeatEq':
        # Ut-v(Uxx+Uyy) = f    在 Omega x T 上
        # U(t, x) = g   在边界上, x 为边界点
        # U(t0, x) = h  在初始点, t0为初始时刻
        if R['equa_name'] == 'Heat1':
            region_lb = 0.0
            region_rt = 1.0
            init_time = 0.0
            end_time = 1.0
            nu = 1.0
            vareps = 0.1
            u_true = lambda x, y, t: tf.exp(-vareps * np.pi * np.pi * t) * tf.sin(np.pi * x) * tf.sin(np.pi * y)
            u_left = lambda x, y, t: tf.exp(-vareps * np.pi * np.pi * t) * tf.sin(np.pi * region_lb) * tf.sin(np.pi * y)
            u_right = lambda x, y, t: tf.exp(-vareps * np.pi * np.pi * t) * tf.sin(np.pi * region_rt) * tf.sin(
                np.pi * y)
            u_bottom = lambda x, y, t: tf.exp(-vareps * np.pi * np.pi * t) * tf.sin(np.pi * x) * tf.sin(
                np.pi * region_lb)
            u_top = lambda x, y, t: tf.exp(-vareps * np.pi * np.pi * t) * tf.sin(np.pi * x) * tf.sin(np.pi * region_rt)
            u_init = lambda x, y, t: tf.exp(-vareps * np.pi * np.pi * init_time) * tf.sin(np.pi * x) * tf.sin(np.pi * y)
            f = lambda x, y, t: (2.0 - vareps) * np.pi * np.pi * tf.exp(-vareps * np.pi * np.pi * t) * tf.sin(
                np.pi * x) * tf.sin(np.pi * y)
        elif R['equa_name'] == 'Heat2':
            region_lb = 0.0
            region_rt = 1.0
            init_time = 0.0
            end_time = 1.0
            nu = 1.0
            u_true = lambda x, y, t: tf.exp(0.5*x+0.5*y-t)
            u_left = lambda x, y, t: tf.exp(0.5*region_lb+0.5*y-t)
            u_right = lambda x, y, t: tf.exp(0.5*region_rt+0.5*y-t)
            u_bottom = lambda x, y, t: tf.exp(0.5*x+0.5*region_lb-t)
            u_top = lambda x, y, t: tf.exp(0.5*x+0.5*region_rt-t)
            u_init = lambda x, y, t: tf.exp(0.5*x+0.5*region_rt-init_time)
            f = lambda x, y, t: -1.5*tf.exp(0.5*x+0.5*y-t)
    elif R['PDE_type'] == 'Burgers':
        # Ut + U*Ux = VUxx + f  在 Omega x T 上, Omega=[-1,1], t>0
        # U(t, x) = g        在边界上, x 为边界点
        # U(t0, x) = h       在初始点, t0为初始时刻
        region_lb = 0.0
        region_rt = 1.0
        init_time = 0.0
        end_time = 1.0
        nu = 0.5
        # nu = 0.1
        # nu = 0.01
        u_true = lambda x, y, t: 1.0/(1+tf.exp(0.5*(x+y-t)/nu))
        u_left = lambda x, y, t: 1.0/(1+tf.exp(0.5*(region_lb+y-t)/nu))
        u_right = lambda x, y, t: 1.0/(1+tf.exp(0.5*(region_rt+y-t)/nu))
        u_bottom = lambda x, y, t: 1.0 / (1 + tf.exp(0.5 * (x + region_lb - t) / nu))
        u_top = lambda x, y, t: 1.0/(1+tf.exp(0.5*(x+region_rt-t)/nu))
        u_init = lambda x, y, t: 1.0/(1+tf.exp(0.5*(x+y-init_time)/nu))
        f= lambda x, y, t: 0.0*tf.zeros_like(t)
    elif R['PDE_type'] == 'WaveEq':
        # Utt-a(Uxx+Uyy) = f    在 Omega x T 上, f一般选为0
        # U(t, x) = g   在边界上, x 为边界点
        # U(t0, x) = h  在初始点, t0为初始时刻
        region_lb = 0.0
        region_rt = 1.0
        init_time = 0.0
        end_time = 1.0
        nu = 1.0
        u_true = lambda x, y, t: tf.sin(np.pi * t)*tf.sin(np.pi * x) * tf.sin(np.pi * y)
        u_left = lambda x, y, t: tf.sin(np.pi * t)*tf.sin(np.pi * region_lb) * tf.sin(np.pi * y)
        u_right = lambda x, y, t: tf.sin(np.pi * t)*tf.sin(np.pi * region_rt) * tf.sin(np.pi * y)
        u_bottom = lambda x, y, t: tf.sin(np.pi * t) * tf.sin(np.pi * x) * tf.sin(np.pi * region_lb)
        u_top = lambda x, y, t: tf.sin(np.pi * t) * tf.sin(np.pi * x) * tf.sin(np.pi * region_rt)
        u_init = lambda x, y, t: tf.sin(np.pi * init_time) * tf.sin(np.pi * x) * tf.sin(np.pi * region_rt)
        f = lambda x, y, t: -1.0*(np.pi*np.pi)*tf.sin(np.pi * t)*tf.sin(np.pi * x) * tf.sin(np.pi * y)

    mscalednn = MscaleDNN(input_dim=R['input_dim']+1, out_dim=R['output_dim'], hidden_layer=R['hidden_layers'],
                          Model_name=R['model2NN'], name2actIn=R['name2act_in'], name2actHidden=R['name2act_hidden'],
                          name2actOut=R['name2act_out'], opt2regular_WB='L0', type2numeric='float32',
                          factor2freq=R['freq'], sFourier=R['sfourier'])
    global_steps = tf.compat.v1.Variable(0, trainable=False)
    with tf.device('/gpu:%s' % (R['gpuNo'])):
        with tf.compat.v1.variable_scope('vscope', reuse=tf.compat.v1.AUTO_REUSE):
            XY_it = tf.compat.v1.placeholder(tf.float32, name='XY_it', shape=[batchsize_it, input_dim])     # * 行 2 列
            t_it = tf.compat.v1.placeholder(tf.float32, name='t_it', shape=[batchsize_it, 1])              # * 行 1 列

            XY_left = tf.compat.v1.placeholder(tf.float32, name='left_bd', shape=[batchsize_bd, input_dim])   # * 行 2 列
            XY_right = tf.compat.v1.placeholder(tf.float32, name='right_bd', shape=[batchsize_bd, input_dim]) # * 行 2 列
            XY_bottom = tf.compat.v1.placeholder(tf.float32, name='bottom_bd', shape=[batchsize_bd, input_dim])         # * 行 2 列
            XY_top = tf.compat.v1.placeholder(tf.float32, name='top_bd', shape=[batchsize_bd, input_dim])     # * 行 2 列
            t_bd = tf.compat.v1.placeholder(tf.float32, name='t_bd', shape=[batchsize_bd, 1])

            XYinit = tf.compat.v1.placeholder(tf.float32, name='XYinit', shape=[batchsize_init, input_dim])  # * 行 2 列
            tinit = tf.compat.v1.placeholder(tf.float32, name='tinit', shape=[batchsize_init, 1])

            boundary_penalty = tf.compat.v1.placeholder_with_default(input=1e3, shape=[], name='bd_p')
            init_penalty = tf.compat.v1.placeholder_with_default(input=1e2, shape=[], name='init_p')
            in_learning_rate = tf.compat.v1.placeholder_with_default(input=1e-5, shape=[], name='lr')

            XY_it2test = tf.compat.v1.placeholder(tf.float32, name='XY_it2test', shape=[batchsize_test, input_dim])
            t_it2test = tf.compat.v1.placeholder(tf.float32, name='t_it2test', shape=[batchsize_test, 1])

            X_it = tf.reshape(XY_it[:, 0], shape=[-1, 1])
            Y_it = tf.reshape(XY_it[:, 1], shape=[-1, 1])

            if R['PDE_type'] == 'HeatEq':
                UNN2train, loss_it = mscalednn.loss2HeatEq(XY=XY_it, t=t_it, fside=f, loss_type=R['loss_type'], alpha=nu)
            elif R['PDE_type'] == 'WaveEq':
                UNN2train, loss_it = mscalednn.loss2WaveEq(XY=XY_it, t=t_it, fside=f, loss_type=R['loss_type'], alpha=nu)
            elif R['PDE_type'] == 'Burgers':
                UNN2train, loss_it = mscalednn.loss2Burges(XY=XY_it, t=t_it, nu_coef=nu, loss_type=R['loss_type'],
                                                           if_lambda2nu=False)

            loss_bd2left = mscalednn.loss_bd2dirichlet(XY_bd=XY_left, t=t_bd, Ubd_exact=u_left)
            loss_bd2right = mscalednn.loss_bd2dirichlet(XY_bd=XY_right, t=t_bd, Ubd_exact=u_right)
            loss_bd2bottom = mscalednn.loss_bd2dirichlet(XY_bd=XY_bottom, t=t_bd, Ubd_exact=u_bottom)
            loss_bd2top = mscalednn.loss_bd2dirichlet(XY_bd=XY_top, t=t_bd, Ubd_exact=u_top)
            loss_bd = loss_bd2left + loss_bd2right + loss_bd2bottom + loss_bd2top

            loss_init = mscalednn.loss2Init(XY=XYinit, tinit=tinit, Uinit_exact=u_init)

            regularSum2WB = mscalednn.get_regularSum2WB()
            PWB = penalty2WB * regularSum2WB

            loss = loss_it + boundary_penalty * loss_bd + init_penalty*loss_init + PWB       # 要优化的loss function

            my_optimizer = tf.compat.v1.train.AdamOptimizer(in_learning_rate)
            if R['train_model'] == 'union_training':
                train_my_loss = my_optimizer.minimize(loss, global_step=global_steps)
            elif R['train_model'] == 'group2_training':
                train_op0 = my_optimizer.minimize(loss, global_step=global_steps)
                train_op1 = my_optimizer.minimize(loss_it, global_step=global_steps)
                train_my_loss = tf.group(train_op0, train_op1)
            elif R['train_model'] == 'group3_training':
                train_op0 = my_optimizer.minimize(loss, global_step=global_steps)
                train_op1 = my_optimizer.minimize(loss_it, global_step=global_steps)
                train_op2 = my_optimizer.minimize(loss_bd + loss_init, global_step=global_steps)
                train_my_loss = tf.group(train_op0, train_op1, train_op2)
            elif R['train_model'] == 'group4_training':
                train_op0 = my_optimizer.minimize(loss, global_step=global_steps)
                train_op1 = my_optimizer.minimize(loss_it, global_step=global_steps)
                train_op2 = my_optimizer.minimize(loss_bd, global_step=global_steps)
                train_op3 = my_optimizer.minimize(loss_init, global_step=global_steps)
                train_my_loss = tf.group(train_op0, train_op1, train_op2, train_op3)

            # 训练上的真解值和训练结果的误差
            Utrue2train = u_true(X_it, Y_it, t_it)
            mean_square_error = tf.reduce_mean(tf.square(Utrue2train - UNN2train))
            residual_error = mean_square_error / tf.reduce_mean(tf.square(Utrue2train))

            X_it2test = tf.reshape(XY_it2test[:, 0], shape=[-1, 1])
            Y_it2test = tf.reshape(XY_it2test[:, 1], shape=[-1, 1])
            Utrue2test = u_true(X_it2test, Y_it2test, t_it2test)
            UNN2test = mscalednn.evalue_MscaleDNN(XY_points=XY_it2test, t_points=t_it2test)

    t0 = time.time()
    loss_it_all, loss_bd_all, loss_all, train_mse_all, train_rel_all = [], [], [], [], []  # 空列表, 使用 append() 添加元素
    test_mse_all, test_rel_all, loss_init_all = [], [], []
    test_epoch = []

    if R['testData_model'] == 'random_generate':
        xy_it2test_batch = DNN_data.rand_it(batchsize_test, input_dim, region_a=region_lb, region_b=region_rt)
        # t_it2test_batch = DNN_data.rand_it(batchsize_test, input_dim, region_a=init_time, region_b=end_time)
        t_it2test_batch = np.ones(shape=[batchsize_test, 1], dtype=np.float32) * 0.5
    else:
        xy_it2test_batch = DNN_data.rand_it(batchsize_test, input_dim, region_a=region_lb, region_b=region_rt)
        # t_it2test_batch = DNN_data.rand_it(batchsize_test, input_dim, region_a=init_time, region_b=end_time)
        t_it2test_batch = np.ones(shape=[batchsize_test, 1], dtype=np.float32) * 0.5

    # ConfigProto 加上allow_soft_placement=True就可以使用 gpu 了
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)  # 创建sess的时候对sess进行参数配置
    config.gpu_options.allow_growth = True              # True是让TensorFlow在运行过程中动态申请显存，避免过多的显存占用。
    config.allow_soft_placement = True                  # 当指定的设备不存在时，允许选择一个存在的设备运行。比如gpu不存在，自动降到cpu上运行
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tmp_lr = learning_rate

        for i_epoch in range(R['max_epoch'] + 1):
            xy_it_batch = DNN_data.rand_it(batchsize_it, input_dim, region_a=region_lb, region_b=region_rt)
            t_it_batch = DNN_data.rand_it(batchsize_it, 1, region_a=init_time, region_b=end_time)

            xl_bd_batch, xr_bd_batch, yb_bd_batch, yt_bd_batch = DNN_data.rand_bd_2D(
                batchsize_bd, input_dim, region_a=region_lb, region_b=region_rt)
            t_bd_batch = DNN_data.rand_it(batchsize_bd, 1, region_a=init_time, region_b=end_time)

            xy_init_batch = DNN_data.rand_it(batchsize_init, input_dim, region_a=region_lb, region_b=region_rt)
            t_init_batch = np.ones(shape=[batchsize_init, 1], dtype=np.float32) * init_time
            tmp_lr = tmp_lr * (1 - lr_decay)
            if R['activate_penalty2bd_increase'] == 1:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_bd = bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_bd = 10 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_bd = 50 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_bd = 100 * bd_penalty_init
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_bd = 200 * bd_penalty_init
                else:
                    temp_penalty_bd = 500 * bd_penalty_init
            else:
                temp_penalty_bd = bd_penalty_init

            if R['activate_penalty2init_increase'] == 1:
                if i_epoch < int(R['max_epoch'] / 10):
                    temp_penalty_init = bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 5):
                    temp_penalty_init = 10 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 4):
                    temp_penalty_init = 50 * bd_penalty_init
                elif i_epoch < int(R['max_epoch'] / 2):
                    temp_penalty_init = 100 * bd_penalty_init
                elif i_epoch < int(3 * R['max_epoch'] / 4):
                    temp_penalty_init = 200 * bd_penalty_init
                else:
                    temp_penalty_init = 500 * bd_penalty_init
            else:
                temp_penalty_init = init_penalty_init

            _, loss_it_tmp, loss_bd_tmp, loss_init_tmp, loss_tmp, train_mse_tmp, train_rel_tmp, pwb = sess.run(
                [train_my_loss, loss_it, loss_bd, loss_init, loss, mean_square_error, residual_error, PWB],
                feed_dict={XY_it: xy_it_batch, t_it: t_it_batch, XY_left: xl_bd_batch, XY_right: xr_bd_batch,
                           XY_bottom: yb_bd_batch, XY_top: yt_bd_batch, t_bd: t_bd_batch, XYinit: xy_init_batch,
                           tinit: t_init_batch, in_learning_rate: tmp_lr, boundary_penalty: temp_penalty_bd,
                           init_penalty: temp_penalty_init})

            loss_it_all.append(loss_it_tmp)
            loss_bd_all.append(loss_bd_tmp)
            loss_init_all.append(loss_init_tmp)
            loss_all.append(loss_tmp)
            train_mse_all.append(train_mse_tmp)
            train_rel_all.append(train_rel_tmp)

            if i_epoch % 1000 == 0:
                run_times = time.time() - t0
                DNN_Log_Print.print_and_log_train_one_epoch2space_time(
                    i_epoch, run_times, tmp_lr, temp_penalty_bd, pwb, loss_it_tmp, loss_bd_tmp, loss_init_tmp, loss_tmp,
                    train_mse_tmp, train_rel_tmp, log_out=log_fileout)

                # ---------------------------   test network ----------------------------------------------
                test_epoch.append(i_epoch / 1000)
                u_true2test, unn2test = sess.run([Utrue2test, UNN2test],
                                                 feed_dict={XY_it2test: xy_it2test_batch, t_it2test: t_it2test_batch})

                point_square_error = np.square(u_true2test - unn2test)
                mse2test = np.mean(point_square_error)
                test_mse_all.append(mse2test)
                res2test = mse2test / np.mean(np.square(u_true2test))
                test_rel_all.append(res2test)

                DNN_Log_Print.print_and_log_test_one_epoch(mse2test, res2test, log_out=log_fileout)

    # ------------------- save the testing results into mat file and plot them -------------------------
    saveData.save_trainLoss2mat_1actFunc(loss_it_all, loss_bd_all, loss_all, actName=act_func,
                                         outPath=R['FolderName'])
    saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=act_func, outPath=R['FolderName'])

    plotData.plotTrain_loss_1act_func(loss_it_all, lossType='loss_it', seedNo=R['seed'], outPath=R['FolderName'])
    plotData.plotTrain_loss_1act_func(loss_bd_all, lossType='loss_bd', seedNo=R['seed'], outPath=R['FolderName'],
                                      yaxis_scale=True)
    plotData.plotTrain_loss_1act_func(loss_all, lossType='loss', seedNo=R['seed'], outPath=R['FolderName'])

    saveData.save_train_MSE_REL2mat(train_mse_all, train_rel_all, actName=act_func, outPath=R['FolderName'])
    plotData.plotTrain_MSE_REL_1act_func(train_mse_all, train_rel_all, actName=act_func, seedNo=R['seed'],
                                         outPath=R['FolderName'], yaxis_scale=True)

    # ----------------------  save testing results to mat files, then plot them --------------------------------
    saveData.save_2testSolus2mat(u_true2test, unn2test, actName='utrue', actName1=act_func,
                                 outPath=R['FolderName'])

    # plotData.plot_Hot_solution2test(u_true2test, size_vec2mat=batchsize_test, actName='Utrue', seedNo=R['seed'],
    #                                 outPath=R['FolderName'])
    # plotData.plot_Hot_solution2test(unn2test, size_vec2mat=batchsize_test, actName=act_func, seedNo=R['seed'],
    #                                 outPath=R['FolderName'])

    saveData.save_testMSE_REL2mat(test_mse_all, test_rel_all, actName=act_func, outPath=R['FolderName'])
    plotData.plotTest_MSE_REL(test_mse_all, test_rel_all, test_epoch, actName=act_func,
                              seedNo=R['seed'], outPath=R['FolderName'], yaxis_scale=True)

    saveData.save_test_point_wise_err2mat(point_square_error, actName=act_func, outPath=R['FolderName'])

    # plotData.plot_Hot_point_wise_err(point_square_error, size_vec2mat=batchsize_test, actName=act_func,
    #                                  seedNo=R['seed'], outPath=R['FolderName'])


if __name__ == "__main__":
    R={}
    R['gpuNo'] = 0
    if platform.system() == 'Windows':
        os.environ["CDUA_VISIBLE_DEVICES"] = "%s" % (R['gpuNo'])
    else:
        print('-------------------------------------- linux -----------------------------------------------')
        # Linux终端没有GUI, 需要添加如下代码，而且必须添加在 import matplotlib.pyplot 之前，否则无效。
        matplotlib.use('Agg')

        if tf.test.is_gpu_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # 设置当前使用的GPU设备仅为第 0,1,2,3 块GPU, 设备名称为'/gpu:0'
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # 文件保存路径设置
    store_file = 'HeatEq'
    # store_file = 'WaveEq'
    # store_file = 'Burgers'
    # store_file = 'Convection'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    OUT_DIR = os.path.join(BASE_DIR, store_file)
    if not os.path.exists(OUT_DIR):
        print('---------------------- OUT_DIR ---------------------:', OUT_DIR)
        os.mkdir(OUT_DIR)

    R['seed'] = np.random.randint(1e5)
    seed_str = str(R['seed'])  # int 型转为字符串型
    FolderName = os.path.join(OUT_DIR, seed_str)  # 路径连接
    R['FolderName'] = FolderName
    if not os.path.exists(FolderName):
        print('--------------------- FolderName -----------------:', FolderName)
        os.mkdir(FolderName)

    # ----------------------------------------  复制并保存当前文件 -----------------------------------------
    if platform.system() == 'Windows':
        tf.compat.v1.reset_default_graph()
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))
    else:
        shutil.copy(__file__, '%s/%s' % (FolderName, os.path.basename(__file__)))

    # ---------------------------- Setup of laplace equation ------------------------------
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    # step_stop_flag = input('please input an  integer number to activate step-stop----0:no---!0:yes--:')
    # R['activate_stop'] = int(step_stop_flag)
    R['activate_stop'] = 1
    # if the value of step_stop_flag is not 0, it will activate stop condition of step to kill program
    # R['max_epoch'] = 10000
    R['max_epoch'] = 50000
    # if 0 != R['activate_stop']:
    #     epoch_stop = input('please input a stop epoch:')
    #     R['max_epoch'] = int(epoch_stop)

    if store_file == 'HeatEq':
        R['PDE_type'] = 'HeatEq'
        # R['equa_name'] = 'Heat1'
        R['equa_name'] = 'Heat2'
    elif store_file == 'Burgers':
        R['PDE_type'] = 'Burgers'
        # R['equa_name'] = 'Burgers1'
        R['equa_name'] = 'Burgers2'
    elif store_file == 'Convection':
        R['PDE_type'] = 'ConvectionEq'
        # R['equa_name'] = 'Convection1'
        R['equa_name'] = 'Convection2'

        if R['equa_name'] == 'Convection1':
            R['max_epoch'] = 50000

    R['epsilon'] = 0.1
    R['order2pLaplace_operator'] = 2

    R['input_dim'] = 2  # 输入维数，即问题的维数(几元问题)
    R['output_dim'] = 1  # 输出维数
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Setup of DNN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # 训练集的设置(内部和边界)
    R['batch_size2interior'] = 5000  # 内部训练数据的批大小
    R['batch_size2boundary'] = 1000  # 边界训练数据大小
    R['batch_size2init'] = 3000
    R['batch_size2test'] = 3600

    # 装载测试数据模式
    # R['testData_model'] = 'loadData'
    R['testData_model'] = 'random_generate'

    R['loss_type'] = 'L2_loss'                          # loss类型:L2 loss

    R['optimizer_name'] = 'Adam'                          # 优化器
    R['learning_rate'] = 2e-4                             # 学习率
    R['learning_rate_decay'] = 5e-5                       # 学习率 decay
    R['train_model'] = 'union_training'
    # R['train_model'] = 'group2_training'
    # R['train_model'] = 'group3_training'

    # 正则化权重和偏置的模式
    R['regular_wb_model'] = 'L0'
    # R['regular_wb_model'] = 'L1'
    # R['regular_wb_model'] = 'L2'
    R['penalty2weight_biases'] = 0.000                    # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.001                  # Regularization parameter for weights
    # R['penalty2weight_biases'] = 0.0025                 # Regularization parameter for weights

    # 边界的惩罚处理方式,以及边界的惩罚因子
    R['activate_penalty2bd_increase'] = 0
    # R['activate_penalty2bd_increase'] = 1

    if R['activate_penalty2bd_increase'] == 0:
        R['init_boundary_penalty'] = 1000                 # Regularization parameter for boundary conditions
    else:
        R['init_boundary_penalty'] = 10                   # Regularization parameter for boundary conditions

    R['activate_penalty2init_increase'] = 0
    # R['activate_penalty2init_increase'] = 1

    if R['activate_penalty2init_increase'] == 0:
        R['init_init_penalty'] = 1000
    else:
        R['init_init_penalty'] = 10

    # 网络的频率范围设置
    R['freq'] = np.arange(1, 31)
    # R['freq'] = np.random.normal(1, 120, 100)

    # &&&&&&&&&&&&&&&&&&& 使用的网络模型 &&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['model2NN'] = 'DNN'
    # R['model2NN'] = 'Scale_DNN'
    # R['model2NN'] = 'Adapt_scale_DNN'
    R['model2NN'] = 'Fourier_DNN'
    # R['model2NN'] = 'Wavelet_DNN'

    # &&&&&&&&&&&&&&&&&&&&&& 隐藏层的层数和每层神经元数目 &&&&&&&&&&&&&&&&&&&&&&&&&&&&
    if R['model2NN'] == 'Fourier_DNN':
        R['hidden_layers'] = (125, 200, 100, 100, 80)  # 1*125+250*200+200*200+200*100+100*100+100*50+50*1=128205
    else:
        # R['hidden_layers'] = (100, 80, 80, 60, 40, 40)
        # R['hidden_layers'] = (200, 100, 80, 50, 30)
        R['hidden_layers'] = (250, 200, 100, 100, 80)  # 1*250+250*200+200*200+200*100+100*100+100*50+50*1=128330
        # R['hidden_layers'] = (500, 400, 300, 200, 100)
        # R['hidden_layers'] = (500, 400, 300, 300, 200, 100)

    # &&&&&&&&&&&&&&&&&&& 激活函数的选择 &&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    # R['name2act_in'] = 'tanh'
    # R['name2act_in'] = 's2relu'
    # R['name2act_in'] = 'sin'
    # R['name2act_in'] = 'sinADDcos'
    R['name2act_in'] = 'gelu'

    # R['name2act_hidden'] = 'relu'
    # R['name2act_hidden'] = 'tanh'
    # R['name2act_hidden'] = 'srelu'
    # R['name2act_hidden'] = 's2relu'
    # R['name2act_hidden'] = 'sin'
    # R['name2act_hidden'] = 'sinADDcos'
    # R['name2act_hidden'] = 'elu'
    R['name2act_hidden'] = 'gelu'
    # R['name2act_hidden'] = 'phi'

    R['name2act_out'] = 'linear'

    R['sfourier'] = 1.0
    if R['model2NN'] == 'Fourier_DNN' and R['name2act_hidden'] == 'tanh':
        R['sfourier'] = 1.0
    elif R['model2NN'] == 'Fourier_DNN' and R['name2act_hidden'] == 's2relu':
        # R['sfourier'] = 0.5
        R['sfourier'] = 1.0
    elif R['model2NN'] == 'Fourier_DNN' and R['name2act_hidden'] == 'sinADDcos':
        # R['sfourier'] = 0.5
        R['sfourier'] = 1.0
    else:
        R['sfourier'] = 1.0

    if R['model2NN'] == 'Wavelet_DNN':
        # R['freq'] = np.concatenate(([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], np.arange(1, 100 - 9)), axis=0)
        # R['freq'] = np.concatenate(([0.25, 0.5, 0.6, 0.7, 0.8, 0.9], np.arange(1, 100 - 6)), axis=0)
        # R['freq'] = np.concatenate(([0.5, 0.6, 0.7, 0.8, 0.9], np.arange(1, 100 - 5)), axis=0)
        # R['freq'] = np.concatenate(([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], np.arange(1, 30-9)), axis=0)
        R['freq'] = np.concatenate(([0.25, 0.5, 0.6, 0.7, 0.8, 0.9], np.arange(1, 100 - 6)), axis=0)
        # R['freq'] = np.arange(1, 100)

    solve_PDE(R)

