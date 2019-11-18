# -*- coding:utf-8 -*-

# keras
from keras.models import Model
from keras.layers.core import Dense,Activation,Reshape
from keras.layers.convolutional import Convolution2D
from keras.layers import Input,merge
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers import ConvLSTM2D

# self define method
from .MyNet import mDense2D,myLayer


class STNN():
    def __init__(self,c_conf=(3, 1, 32, 32), p_conf=(3, 1, 32, 32), t_conf=(3, 1, 32, 32),
                 external_dim=8,st_lstm_filter=32,nb_dense_block=3, nb_layers=8, growth_rate=4, nb_filter=16):

        # default variable
        self.kernel_size = 3
        self.base_filter = 3
        self.output_ch = 1
        self.external_dim=external_dim

        # default variable
        self.use_dense_c = True
        self.use_dense_pt = True
        # temporal variable
        self.c_conf = c_conf
        self.p_conf = p_conf
        self.t_conf = t_conf
        # STNN param
        self.st_lstm_filter = st_lstm_filter
        self.nb_dense_block = nb_dense_block
        self.nb_layers = nb_layers
        self.growth_rate = growth_rate
        self.nb_filter =nb_filter

    def build_net(self):
        # initial input and output list
        main_inputs = []
        outputs = []

        # use LSTM
        conf = self.c_conf
        if conf is not None:
            time, channels, map_height, map_width = conf
            input = Input(shape=(time, channels, map_height, map_width))
            main_inputs.append(input)
            # LSTM layer
            LSTM_out = ConvLSTM2D(filters=self.st_lstm_filter, kernel_size=(3, 3),
                                  strides=1, padding='same',activation='tanh',
                                  return_sequences=False)(input)

            LSTM_out = BatchNormalization(axis=-1, gamma_regularizer=l2(1E-4),
                                          beta_regularizer=l2(1E-4))(LSTM_out)
            # Dense Layer
            if self.use_dense_c == True:
                dense_output = mDense2D(LSTM_out, nb_dense_block=self.nb_dense_block,
                                        nb_layers=self.nb_layers,growth_rate=self.growth_rate,
                                        nb_filter=self.nb_filter, base_filter=self.base_filter).layer
            else:
                dense_output = Convolution2D(nb_filter=32, nb_row=self.kernel_size,
                                             nb_col=self.kernel_size,
                                             border_mode="same")(LSTM_out)
            # Conv Layer
            x = Convolution2D(nb_filter=8, nb_row=self.kernel_size,
                              nb_col=self.kernel_size,
                              border_mode="same")(dense_output)
            activation = Activation('relu')(x)
            conv2 = Convolution2D(nb_filter=self.output_ch, nb_row=self.kernel_size,
                                  nb_col=self.kernel_size,
                                  border_mode="same")(activation)
            outputs.append(conv2)

        # not use LSTM
        for conf in [self.p_conf, self.t_conf]:
            if conf is not None:
                len_seq, nb_flow, map_height, map_width = conf
                input = Input(shape=(nb_flow * len_seq, map_height, map_width))
                main_inputs.append(input)
                # DenseLayer
                if self.use_dense_pt == True:
                    dense_output = mDense2D(input, nb_dense_block=self.nb_dense_block,
                                            nb_layers=self.nb_layers,growth_rate=self.growth_rate,
                                            nb_filter=self.nb_filter, base_filter=self.base_filter).layer
                else:
                    dense_output = Convolution2D(nb_filter=32, nb_row=self.kernel_size,
                                                 nb_col=self.kernel_size,
                                                 border_mode="same")(input)
                # Conv Layer
                x = Convolution2D(nb_filter=8, nb_row=self.kernel_size,
                                  nb_col=self.kernel_size,
                                  border_mode="same")(dense_output)
                activation = Activation('relu')(x)
                conv2 = Convolution2D(nb_filter=self.output_ch, nb_row=self.kernel_size,
                                      nb_col=self.kernel_size,
                                      border_mode="same")(activation)
                outputs.append(conv2)

        _, _, map_height, map_width = self.c_conf
        return main_inputs,outputs, map_height, map_width

    def build(self,):

        main_inputs,outputs, map_height, map_width = self.build_net()

        # parameter-matrix-based fusion(weighted sum)
        if len(outputs) == 1:
            main_output = outputs[0]
        else:
            new_outputs = []
            # assign weight
            for output in outputs:
                new_outputs.append(myLayer()(output))
            # element-wise sum
            main_output = merge(new_outputs, mode='sum')

        # fusing with external component
        if self.external_dim != None and self.external_dim > 0:
            # external input
            external_input = Input(shape=(self.external_dim,))
            main_inputs.append(external_input)
            # embedding
            embedding = Dense(output_dim=10)(external_input)
            embedding = Activation('relu')(embedding)
            # rescale
            h1 = Dense(output_dim=self.output_ch * map_height * map_width)(embedding)
            activation = Activation('relu')(h1)
            # reshape for sake of consequently fusing
            external_output = Reshape((self.output_ch, map_height, map_width))(activation)
            # element-wise sum
            main_output = merge([main_output, external_output], mode='sum')
        else:
            print('do not use external input, external_dim:', self.external_dim)

        main_output = Activation('tanh')(main_output)
        # main_inputs is a list contains 4 elements
        model = Model(input=main_inputs, output=main_output)

        return model