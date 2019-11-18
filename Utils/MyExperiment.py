# -*- coding:utf-8 -*-

from keras.optimizers import Adam,SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from keras import metrics
from keras.utils.vis_utils import plot_model

from .MyModel import STNN
from . import MyMetrics as metrics2
from .DataPrepare import read_cache,cache,DataLoad

import time
import pickle
import os


import warnings
warnings.filterwarnings('ignore')

# np.random.seed(1337)  # for reproducibility

class Exp_STNN():
    def __init__(self):
        # base conv kernel
        self.base_filter = 3
        # temporal scale
        self._interval = 10
        # spatial size
        self.map_height = 192
        self.map_width = 192
        # type of prediction
        self.dic_PreType = {'up': 0, 'down': 1, 'flow': 2}
        self.pre_type = self.dic_PreType['flow']  # use : up down flow # Y
        self.nb_flow = 1  # numbers of types to predict simultaneously  # N
        # training parameters
        self.days_test = 9
        self.T = int(1440 / self._interval)
        self.len_test = self.T * self.days_test
        self.lr = 0.0005  # learning rate
        self.nb_epoch = 150  # number of epoch at training stage
        self.batch_size = 4  # batch size
        # program's macro define
        self.MAKECACHE = True
        self.rowDataFileList = ['D:\Code_win_linux\pywork\Multi-Method-Compare-4week'
                                '\data\WuHanTaxi_10min\SanHuan_real_10min_fe_192X192.h5']

        # model_name
        self.model_name = 'STNN'
        # temporal length
        self.len_recent = 3  # length of closeness dependent sequence # C
        self.len_daily = 3  # length of peroid dependent sequence   # P
        self.len_weekly = 3  # length of trend dependent sequence # T
        # Dense Layer setting
        self.nb_dense_block = 0  # D
        self.nb_layers = 8  # L
        self.growth_rate = 4  # G
        self.nb_filter = 16  # N
        self.st_lstm_filter = 8  # T
        # hyperparams_name
        self.hyperparams_name = self.load_hyperparams_name()
        self.fname_param=self.load_fname_param()

        self.fname = os.path.join('Data//CACHE', self.model_name + '_WuHanTaxi_{}min_{}x{}_R{}_D{}_W{}_N{}_Y{}_2D.h5'
                                  .format(self._interval, self.map_height, self.map_width, self.len_recent,
                                          self.len_daily, self.len_weekly, self.nb_flow, self.pre_type))
        self.preprocess_name = os.path.join('Result//MODEL',
                                            self.model_name + '_R{}_D{}_W{}_N{}_Y{}_preprocessing_2D_{}min.pkl'
                                            .format(self.len_recent, self.len_daily, self.len_weekly,
                                                    self.nb_flow, self.pre_type, self._interval))

    def load_fname_param(self,history_lr=None):
        if history_lr==None:
            fname_param = os.path.join('Result//MODEL', '{}.best.h5'.format(self.hyperparams_name))
        else:
            fname_param = os.path.join('Result//MODEL', '{}.best.h5'.format(self.load_hyperparams_name(history_lr)))

        return fname_param

    def load_hyperparams_name(self,set_lr=None):
        if set_lr==None:
            set_lr=self.lr

        hyperparams_name = self.model_name + '_min{}_{}x{}_r{}.d{}.w{}.y{}_B{}.L{}.G{}.F{}.lr{}_2D'.format(
            self._interval, self.map_height, self.map_width, self.len_recent, self.len_daily, self.len_weekly,
            self.pre_type, self.nb_dense_block,self.nb_layers, self.growth_rate, self.nb_filter, set_lr)

        return hyperparams_name

    def load_data(self,meta_data=True):
        # load data
        print("loading data...")
        ts = time.time()

        if os.path.exists(self.fname) and self.MAKECACHE:
            X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = read_cache(
                self.fname, preprocess_name=self.preprocess_name)
            print("load %s successfully" % self.fname)
        else:
            op_load = DataLoad(meta_data=meta_data)
            X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = op_load.load(
                fname_list=self.rowDataFileList, T=self.T, len_recent=self.len_recent,
                len_daily=self.len_daily,len_weekly=self.len_weekly, len_test=self.len_test,
                preprocess_name=self.preprocess_name,pre_type=self.pre_type, _interval=self._interval)
            if self.MAKECACHE:
                cache(self.fname, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test)

        for _X in X_train:
            print(_X.shape, )
        print('\n')
        for _X in X_test:
            print(_X.shape,)
        print('\n')

        print("\n days (test): ", [v[:8] for v in timestamp_test[0::self.T]])
        print("\nelapsed time (loading data): %.3f seconds\n" % (time.time() - ts))

        print('=' * 10)
        print("compiling model...")
        print(
            "**at the first time, it takes a few minites to compile if you use [Theano] as the backend**")

        return X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test

    def build_model(self,external_dim,set_lr):
        # three kinds of input
        c_conf = (self.len_recent, self.nb_flow, self.map_height,
                  self.map_width) if self.len_recent > 0 else None
        p_conf = (self.len_daily, self.nb_flow, self.map_height,
                  self.map_width) if self.len_daily > 0 else None
        t_conf = (self.len_weekly, self.nb_flow, self.map_height,
                  self.map_width) if self.len_weekly > 0 else None

        model = STNN(c_conf=c_conf, p_conf=p_conf, t_conf=t_conf, external_dim=external_dim,
                     st_lstm_filter=self.st_lstm_filter,
                     nb_dense_block=self.nb_dense_block, nb_layers=self.nb_layers,
                     growth_rate=self.growth_rate, nb_filter=self.nb_filter).build()

        adam = Adam(lr=set_lr)
        # sgd = SGD(lr=set_lr, momentum=0.9)
        model.compile(loss='mse', optimizer=adam,
                      metrics=[metrics2.rmse, metrics.mse, metrics.mae])
        # model.summary()

        plot_model(model,
                   to_file='Result//MODEL//' + self.model_name + '_B{}_L{}_G{}_F{}_2D.png'
                   .format(self.nb_dense_block, self.nb_layers, self.growth_rate,
                           self.nb_filter),show_shapes=True)

        print("plot finished")
        return model

    def train(self):

        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = self.load_data()

        ts = time.time()
        model = self.build_model(external_dim,set_lr=self.lr)

        # initial check -- delete redundancy
        # fname_param_1 = os.path.join('Result//MODEL', '{}.best.h5'.format(self.hyperparams_name))
        # if os.path.exists(fname_param_1):
        #     os.remove(fname_param_1)
        # fname_param_2 = os.path.join('Result//MODEL', '{}.h5'.format(self.hyperparams_name))
        # if os.path.exists(fname_param_2):
        #     os.remove(fname_param_2)
        fname_param = os.path.join('Result//MODEL', '{}.best.h5'.format(self.hyperparams_name))

        # training setting
        model_checkpoint = ModelCheckpoint(fname_param, monitor='val_rmse', verbose=1,
                                           save_best_only=True, mode='min')
        ReduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=2, verbose=1,
                                     mode='auto', epsilon=0.00005, cooldown=0, min_lr=0)
        print("\nelapsed time (compiling model): %.3f seconds\n" % (time.time() - ts))

        # model training
        print('=' * 10)
        print("training model...")
        ts = time.time()
        history = model.fit(X_train, Y_train,
                            nb_epoch=self.nb_epoch,
                            batch_size=self.batch_size,
                            validation_split=0.1,
                            callbacks=[model_checkpoint, ReduceLR],
                            verbose=1)
        model.save_weights(os.path.join(
            'Result//MODEL', '{}.h5'.format(self.hyperparams_name)), overwrite=True)
        pickle.dump((history.history), open(os.path.join(
            'Result//RET', '{}.history.pkl'.format(self.hyperparams_name)), 'wb'))
        print("\nelapsed time (training): %.3f seconds\n" % (time.time() - ts))

        # model evaluating
        print('=' * 10)
        print('evaluating using the model that has the best loss on the valid set')
        ts = time.time()
        model.load_weights(fname_param)

        score = model.evaluate(X_train, Y_train, batch_size=self.batch_size, verbose=0)
        print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
              (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
        score = model.evaluate(
            X_test, Y_test, batch_size=self.batch_size, verbose=0)
        print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
              (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
        print('rmse (norm): %.6f rmse (real): %.6f' %
              (score[1], score[1] * (mmn._max - mmn._min) / 2.))
        print('mae (norm): %.6f mae (real): %.6f' %
              (score[3], score[3] * (mmn._max - mmn._min) / 2.))

        print("\nelapsed time (eval): %.3f seconds\n" % (time.time() - ts))

    def predict(self,history_lr=None):
        print('=' * 10)
        print('predicting...')
        X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test = self.load_data()
        # load fname_param
        if history_lr == None:
            fname_param = self.fname_param
        else:
            fname_param = self.load_fname_param(history_lr)
        # build model
        model = self.build_model(external_dim,set_lr=history_lr)
        model.load_weights(fname_param)

        # predict
        score = model.evaluate(X_train, Y_train, batch_size=self.batch_size, verbose=0)
        print('Train score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
              (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))

        score = model.evaluate(
            X_test, Y_test, batch_size=self.batch_size, verbose=0)
        print('Test score: %.6f rmse (norm): %.6f rmse (real): %.6f' %
              (score[0], score[1], score[1] * (mmn._max - mmn._min) / 2.))
        print('rmse (norm): %.6f rmse (real): %.6f' %
              (score[1], score[1] * (mmn._max - mmn._min) / 2.))
        print('mae (norm): %.6f mae (real): %.6f' %
              (score[3], score[3] * (mmn._max - mmn._min) / 2.))

