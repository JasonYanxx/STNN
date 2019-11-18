# -*- coding:utf-8 -*-

import h5py
import pickle
import time
import numpy as np
import pandas as pd
from datetime import datetime

def read_cache(fname,preprocess_name):
    mmn = pickle.load(open(preprocess_name, 'rb'))

    f = h5py.File(fname, 'r')
    num = int(f['num'].value)
    X_train, Y_train, X_test, Y_test = [], [], [], []
    for i in range(num):
        X_train.append(f['X_train_%i' % i].value)
        X_test.append(f['X_test_%i' % i].value)
    Y_train = f['Y_train'].value
    Y_test = f['Y_test'].value
    external_dim = f['external_dim'].value
    timestamp_train = f['T_train'].value
    timestamp_test = f['T_test'].value
    f.close()

    return X_train, Y_train, X_test, Y_test, mmn, external_dim, timestamp_train, timestamp_test

def cache(fname, X_train, Y_train, X_test, Y_test, external_dim, timestamp_train, timestamp_test):
    h5 = h5py.File(fname, 'w')
    h5.create_dataset('num', data=len(X_train))

    for i, data in enumerate(X_train):
        h5.create_dataset('X_train_%i' % i, data=data)
    # for i, data in enumerate(Y_train):
    for i, data in enumerate(X_test):
        h5.create_dataset('X_test_%i' % i, data=data)
    h5.create_dataset('Y_train', data=Y_train)
    h5.create_dataset('Y_test', data=Y_test)
    external_dim = -1 if external_dim is None else int(external_dim)
    h5.create_dataset('external_dim', data=external_dim)
    h5.create_dataset('T_train', data=timestamp_train)
    h5.create_dataset('T_test', data=timestamp_test)
    h5.close()

def string2timestamp(strings, T=48):
    timestamps = []

    time_per_slot = 24.0 / T
    num_per_T = T // 24
    for t in strings:
        year, month, day, slot = int(t[:4]), int(t[4:6]), int(t[6:8]), int(t[8:])-1
        timestamps.append(pd.Timestamp(datetime(year, month, day, hour=int(slot * time_per_slot), minute=(slot % num_per_T) * int(60.0 * time_per_slot))))

    return timestamps

class MinMaxNormalization(object):
    def __init__(self):
        pass

    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print("min:", self._min, "max:", self._max)

    def transform(self, X):
        X = 1. * (X - self._min) / (self._max - self._min)
        X = X * 2. - 1.
        return X

    def transform_fake(self, X):
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = (X + 1.) / 2.
        X = 1. * X * (self._max - self._min) + self._min
        return X

class STMatrix(object):
    def __init__(self, data, timestamps, T=144, CheckComplete=True):
        super(STMatrix, self).__init__()
        assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = timestamps
        self.T = T
        self.pd_timestamps = string2timestamp(timestamps, T=self.T)
        if CheckComplete:
            self.check_complete()
        # index
        self.make_index()

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def check_complete(self):
        missing_timestamps = []
        offset = pd.DateOffset(minutes=24 * 60 // self.T)
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            if pd_timestamps[i-1] + offset != pd_timestamps[i]:
                missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i-1], pd_timestamps[i]))
            i += 1
        for v in missing_timestamps:
            print(v)
        assert len(missing_timestamps) == 0

    def get_matrix(self, timestamp):
        return self.data[self.get_index[timestamp]]

    def save(self, fname):
        pass

    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def create_dataset_STNN(self, len_recent=3, len_weekly=3, TrendInterval=7, len_daily=3, PeriodInterval=1):

        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        XC = []
        XP = []
        XT = []
        Y = []
        timestamps_Y = []

        depends = [range(1, len_recent+1),
                   [PeriodInterval * self.T + j for j in range(int(int(1-len_daily)/2),int(int(len_daily+1)/2))],
                   [TrendInterval * self.T + j for j in range(int(int(1-len_weekly)/2),int(int(len_weekly+1)/2))]]

        i = max(max(depends[0]),max(depends[1]),max(depends[2]))
        while i < len(self.pd_timestamps):
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                Flag = self.check_it([self.pd_timestamps[i] - j * offset_frame for j in depend])

            if Flag is False:
                i += 1
                continue

            x_c = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[0]]
            x_p = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[1]]
            x_t = [self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in depends[2]]
            y = self.get_matrix(self.pd_timestamps[i])
            # for LSTM use
            x_c.reverse()


            if len_recent > 0:
                XC.append(np.vstack(x_c))
            if len_daily > 0:
                XP.append(np.vstack(x_p))
            if len_weekly > 0:
                XT.append(np.vstack(x_t))
            Y.append(y)
            timestamps_Y.append(self.timestamps[i])
            i += 1

        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        Y = np.asarray(Y)

        print("XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "Y shape:", Y.shape)
        return XC, XP, XT, Y, timestamps_Y

class DataLoad():
    def __init__(self,meta_data=True):
        self.meta_data = meta_data

    def load_stdata(self,fname):
        f = h5py.File(fname, 'r')
        data = f['data'].value
        timestamps = f['date'].value
        f.close()
        return data, timestamps

    def stat(self,fname):

        def get_nb_timeslot(f):
            s = f['date'][0]
            e = f['date'][-1]
            year, month, day = map(int, [s[:4], s[4:6], s[6:8]])
            ts = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
            year, month, day = map(int, [e[:4], e[4:6], e[6:8]])
            te = time.strptime("%04i-%02i-%02i" % (year, month, day), "%Y-%m-%d")
            nb_timeslot = (time.mktime(te) - time.mktime(ts)) / (0.5 * 3600) + 48
            ts_str, te_str = time.strftime("%Y-%m-%d", ts), time.strftime("%Y-%m-%d", te)
            return nb_timeslot, ts_str, te_str

        with h5py.File(fname) as f:
            nb_timeslot, ts_str, te_str = get_nb_timeslot(f)
            nb_day = int(nb_timeslot / 48)
            mmax = f['data'].value.max()
            mmin = f['data'].value.min()
            stat = '=' * 5 + 'stat' + '=' * 5 + '\n' + \
                   'data shape: %s\n' % str(f['data'].shape) + \
                   '# of days: %i, from %s to %s\n' % (nb_day, ts_str, te_str) + \
                   '# of timeslots: %i\n' % int(nb_timeslot) + \
                   '# of timeslots (available): %i\n' % f['date'].shape[0] + \
                   'missing ratio of timeslots: %.1f%%\n' % ((1. - float(f['date'].shape[0] / nb_timeslot)) * 100) + \
                   'max: %.3f, min: %.3f\n' % (mmax, mmin) + \
                   '=' * 5 + 'stat' + '=' * 5
            print(stat)

    def remove_incomplete_days(self,data, timestamps, T=144):
        # remove a certain day which has not 48 timestamps
        days = []  # available days: some day only contain some seqs
        days_incomplete = []
        i = 0
        while i < len(timestamps):
            if int(timestamps[i][8:]) != 1:
                i += 1
            elif i + T - 1 < len(timestamps) and int(timestamps[i + T - 1][8:]) == T:
                days.append(timestamps[i][:8])
                i += T
            else:
                days_incomplete.append(timestamps[i][:8])
                i += 1
        print("incomplete days: ", days_incomplete)
        days = set(days)
        idx = []

        for i, t in enumerate(timestamps):
            if t[:8] in days:
                idx.append(i)

        data = data[idx]
        timestamps = [timestamps[i] for i in idx]

        return data, timestamps

    def timestamp2vec(self,timestamps):
        # tm_wday range [0, 6], Monday is 0
        vec = [time.strptime(t[:8], '%Y%m%d').tm_wday for t in timestamps]
        ret = []
        for i in vec:
            v = [0 for _ in range(7)]
            v[i] = 1
            if i >= 5:
                v.append(0)  # weekend
            else:
                v.append(1)  # weekday
            ret.append(v)
        return np.asarray(ret)

    def load(self,fname_list, T=144,
             len_recent=None, len_daily=None, len_weekly=None,
             len_test=None, preprocess_name='preprocessing.pkl',
             pre_type=0,  _interval=10):

        assert (len_recent + len_daily + len_weekly > 0)
        # load data
        data_all = []
        timestamps_all = list()
        for fname in fname_list:
            print("file name: ", fname)
            self.stat(fname)
            data, timestamps = self.load_stdata(fname)
            # remove a certain day which does not have 48 timestamps
            data, timestamps = self.remove_incomplete_days(data, timestamps, T)
            data = data[:, pre_type:(pre_type + 1)]
            data[data < 0] = 0.
            data_all.append(data)
            timestamps_all.append(timestamps)
            print("\n")

        # minmax_scale
        data_train = np.vstack(np.copy(data_all))[:-len_test]
        print('train_data shape: ', data_train.shape)
        mmn = MinMaxNormalization()
        mmn.fit(data_train)
        # minmax normalization
        data_all_mmn = [mmn.transform(d) for d in data_all]

        # save min_max_scale
        fpkl = open(preprocess_name, 'wb')
        for obj in [mmn]:
            pickle.dump(obj, fpkl)
        fpkl.close()

        XC, XP, XT = [], [], []
        Y = []
        timestamps_Y = []
        for data, timestamps in zip(data_all_mmn, timestamps_all):
            st = STMatrix(data, timestamps, T, CheckComplete=False)
            _XC, _XP, _XT, _Y, _timestamps_Y = st.create_dataset_STNN(
                len_recent=len_recent, len_daily=len_daily, len_weekly=len_weekly)

            XC.append(_XC)
            XP.append(_XP)
            XT.append(_XT)
            Y.append(_Y)
            timestamps_Y += _timestamps_Y

        meta_feature = []
        if self.meta_data:
            # load time feature
            time_feature = self.timestamp2vec(timestamps_Y)
            meta_feature.append(time_feature)

        meta_feature = np.hstack(meta_feature) if len(
            meta_feature) > 0 else np.asarray(meta_feature)
        metadata_dim = meta_feature.shape[1] if len(
            meta_feature.shape) > 1 else None
        if metadata_dim < 1:
            metadata_dim = None
        if self.meta_data :
            print('time feature:', time_feature.shape, 'mete feature: ', meta_feature.shape)

        if XC != None: XC = np.vstack(XC)
        if XP != None: XP = np.vstack(XP)
        if XT != None: XT = np.vstack(XT)
        Y = np.vstack(Y)

        print("XC shape: ", XC.shape, "XP shape: ", XP.shape,
              "XT shape: ", XT.shape, "Y shape:", Y.shape)

        print("add new axis")
        XC = XC[:, :, np.newaxis, :, :]
        print("XC shape: ", XC.shape, "XP shape: ", XP.shape,
              "XT shape: ", XT.shape, "Y shape:", Y.shape)

        XC_train, XP_train, XT_train, Y_train = XC[
                                                :-len_test], XP[:-len_test], XT[:-len_test], Y[:-len_test]
        XC_test, XP_test, XT_test, Y_test = XC[
                                            -len_test:], XP[-len_test:], XT[-len_test:], Y[-len_test:]
        timestamp_train, timestamp_test = timestamps_Y[
                                          :-len_test], timestamps_Y[-len_test:]

        X_train = []
        X_test = []
        for l, X_ in zip([len_recent, len_daily, len_weekly], [XC_train, XP_train, XT_train]):
            if l > 0:
                X_train.append(X_)

        for l, X_ in zip([len_recent, len_daily, len_weekly], [XC_test, XP_test, XT_test]):
            if l > 0:
                X_test.append(X_)

        print('Y train shape:', Y_train.shape,
              'Y test shape: ', Y_test.shape)

        if metadata_dim is not None:
            meta_feature_train, meta_feature_test = meta_feature[
                                                    :-len_test], meta_feature[-len_test:]
            X_train.append(meta_feature_train)
            X_test.append(meta_feature_test)
        for _X in X_train:
            print(_X.shape, )
        print()
        for _X in X_test:
            print(_X.shape, )
        print()

        return X_train, Y_train, X_test, Y_test, mmn, metadata_dim, timestamp_train, timestamp_test
