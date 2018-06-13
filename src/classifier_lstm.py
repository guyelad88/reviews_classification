from __future__ import print_function

global_auc_list = list()            # global list which store auc per epoch to find max auc

# structure as follow
# fold -> epoch -> fpr, tpr, auc
global_statistic_auc_dict = dict()

# save best epoch for every fold
global_max_auc_epoch_dict = {
    'auc': 0,
    'epoch': None
}

from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints

############################################## attention with context ##############################################

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):

        uit = K.dot(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)

        mul_a = uit * self.u  # with this
        ait = K.sum(mul_a, axis=2)  # and this

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number epsilon to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


class PredictDescriptionModelLSTM:
    '''
        this class build and evaluate model for a given configuration.
        input:
            1. data - already split to train and test ana label columns
            2. network configuration data
        output:
            1. log with calculation - accuracy and auc results
            2. ROC plot for each epoch
                2.1 under directory ../results/ROC
            3. tensor board graph
                3.1 under directory ../results/tensor_board_graph
    '''
    def __init__(self,
                 file_directory,
                 logging,
                 cur_time,
                 x_train,
                 y_train,
                 x_test,
                 y_test,
                 train_reason,
                 test_reason,
                 fold_counter,
                 network_dict,
                 df_configuration_dict,
                 multi_class_configuration_dict,
                 attention_configuration_dict,
                 tensor_board_dir,
                 embedding_pre_trained,
                 embedding_type,
                 vertical_type
                 ):

        # file arguments
        self.file_directory = file_directory                # directory contain all data for all traits
        self.logging = logging
        self.cur_time = cur_time

        self.x_train = x_train      # list of string - string per item description (before insert into count vec)
        self.x_test = x_test        # list of string - string per item description (before insert into count vec)
        self.y_train = y_train      # list of bool train text label
        self.y_test = y_test        # list of bool text text label

        self.train_reason = train_reason        # series of train reviews and their failure reason
        self.test_reason = test_reason          # series of test reviews and their failure reason

        self.fold_counter = fold_counter    #

        self.x_train_sequence = None
        self.x_test_sequence = None
        self.word_index = None              # keras tokenizer
        self.index_word_dict = None         # transpose of word_index

        self.color_list = None              # color correspond to percentile values
        self.percentile_list = None         # percentile list of attention values in a sentence

        self.max_auc_list = list()

        # LSTM parameters
        self.max_features = network_dict['max_features']                # 20000
        self.maxlen = network_dict['maxlen']                            # 200 (item description maximum length)
        self.batch_size = network_dict['batch_size']                    # 16
        self.embedding_size = network_dict['embedding_size']            # 16
        self.lstm_hidden_layer = network_dict['lstm_hidden_layer']
        self.num_epoch = network_dict['num_epoch']                      # 20
        self.dropout = network_dict['dropout']                          # 0.2
        self.recurrent_dropout = network_dict['recurrent_dropout']      # 0.2
        self.tensor_board_bool = network_dict['tensor_board_bool']
        self.max_num_words = network_dict['max_num_words']
        self.optimizer = network_dict['optimizer']
        self.patience = network_dict['patience']

        self.df_configuration_dict = df_configuration_dict
        self.multi_class_configuration_dict = multi_class_configuration_dict  # dict multi-class classification data
        self.attention_configuration_dict = attention_configuration_dict      # attention configuration

        self.tensor_board_dir = tensor_board_dir                        # bool - if to use tensor board
        self.embedding_pre_trained = embedding_pre_trained              # bool - if to use pre trained embedding
        self.embedding_type = embedding_type                            # dict: type(glove or gensim), path to WV
        self.vertical_type = vertical_type                              # vertical type

        # initialize global variable
        global global_auc_list  # update global variable
        global_auc_list = list()

        global global_statistic_auc_dict
        global_statistic_auc_dict = dict()

        global global_max_auc_epoch_dict
        global_max_auc_epoch_dict = {
            'auc': 0,
            'epoch': None
        }

        self.logging.info('')
        self.logging.info('Network parameters: ')
        for param, value in network_dict.iteritems():
            self.logging.info('Parameter: ' + str(param) + ', Val: ' + str(value))

    # build log object
    def init_debug_log(self):
        import logging

        lod_file_name = self.log_dir + 'predict_personality_from_desc_' + str(self.cur_time) + '.log'

        # logging.getLogger().addHandler(logging.StreamHandler())

        logging.basicConfig(filename=lod_file_name,
                            format='%(asctime)s, %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

        # print result in addition to log file
        if self.verbose_flag:
            stderrLogger = logging.StreamHandler()
            stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
            logging.getLogger().addHandler(stderrLogger)

        logging.info("")
        logging.info("")
        logging.info("start log program")
        return

    # a. prepare data - tokenizer e.g
    # b. build and run lstm model
    def run_experiment(self):

        self.prepare_data()                             # tokenizer, fit
        self.model()     # build model and train + inference it

        return global_statistic_auc_dict, global_max_auc_epoch_dict
        # return test_score, test_accuracy

    # tokenizer sentences by keras and prepare them to lstm model
    def prepare_data(self):

        from keras.preprocessing import text

        t = text.Tokenizer(num_words=self.max_num_words,                            # max words in tokenizer
                           filters="!'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n",
                           lower=True,
                           split=" ",
                           char_level=False,
                           oov_token='UNK')

        # fit the tokenizer on the documents
        t.fit_on_texts(self.x_train)

        self.logging.info('')
        self.logging.info('token properties: ')
        self.logging.info('filter using: ' + str("!'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n"))
        self.logging.info('OOV token: ' + str('UNK'))
        # print tokenizer results
        # self.logging.info('# docs: ' + str(t.document_count))
        # self.logging.info('t word index: ' + str(t.word_index))
        # self.logging.info('t word counts: ' + str(t.word_counts))
        # self.logging.info('t word docs: ' + str(t.word_docs))

        self.word_index = t.word_index      # will used for pre trained glove embedding
        self.index_word_dict = dict((v, k) for k, v in self.word_index.iteritems())
        self.x_train = t.texts_to_sequences(self.x_train)
        self.x_test = t.texts_to_sequences(self.x_test)

        return

    # core function
    # run LSTM using keras library
    def model(self):

        self._padding_sentences()                                # pad sentences regards to max len input
        model = self._build_model()                              # build lstm model
        tensor_board_dir = self._create_tensor_board_dir()       # create tensor board dir if needed
        self._fit_model(model, tensor_board_dir)                 # run epoch
        self._evaluation(model)       # evaluation - store test results after final epoc
        self._update_folder_name()                               # change name of roc folder name
        return

    # pad sentences regards to length parameter
    def _padding_sentences(self):

        from keras.preprocessing import sequence

        self.logging.info('')
        self.logging.info(str(len(self.x_train)) + ' train sequences')
        self.logging.info(str(len(self.x_test)) + ' test sequences')

        self.logging.info('Pad sequences (samples x time)')

        self.x_train = sequence.pad_sequences(
            self.x_train,
            maxlen=self.maxlen
        )

        self.x_test = sequence.pad_sequences(
            self.x_test,
            maxlen=self.maxlen
        )

        self.logging.info('sentences shape after padding')
        self.logging.info('x_train shape: ' + str(self.x_train.shape))
        self.logging.info('x_test shape: ' + str(self.x_test.shape))

        return

    # build keras lstm model
    # model option
        # a.  with/without attention
        # a.1 attention before/after lstm
        # b.  multi-class/binary-class prediction

    def _build_model(self):

        from keras.models import Sequential
        from keras.layers import Dense, Embedding, Input, Concatenate
        from keras.layers import LSTM, Dropout, Bidirectional
        from keras.models import Model

        self.logging.info('')
        self.logging.info('Start to build a model...')

        # multi-class classification (e.g. Good vs. Bad (Subjective sentence) vs. Bad (Missing context))
        if self.multi_class_configuration_dict['multi_class_bool']:

            self.logging.info('build a multi-task classification model')

            comment_input = Input(shape=(self.maxlen,))
            embedding_layer = self._add_pre_trained_embedding()
            embedded_sequences = embedding_layer(comment_input)

            self.logging.info('')
            self.logging.info('LSTM properties: ')
            self.logging.info('num hidden neurons: ' + str(self.lstm_hidden_layer))
            self.logging.info('dropout: ' + str(self.dropout))
            self.logging.info('recurrent_dropout: ' + str(self.recurrent_dropout))

            lstm_layer = (LSTM(
                    self.lstm_hidden_layer,
                    dropout=self.dropout,
                    recurrent_dropout=self.recurrent_dropout
            ))
            if len(self.y_train) == 2:
                self.logging.info('multi-task with 2 outputs')
                x = lstm_layer(embedded_sequences)

                out1 = Dense(1, activation='sigmoid')(x)
                out2 = Dense(1, activation='sigmoid')(x)
                # mid2 = Dense(1, activation='sigmoid')(x)
                # con_layer = Concatenate()([out1, mid2])
                # out2 = Dense(1, activation='sigmoid')(con_layer)

                model = Model(inputs=comment_input, outputs=[out1, out2])           # , out3])
                model.compile(loss='binary_crossentropy',
                              optimizer=self.optimizer,
                              loss_weights=[4, 1],
                              metrics=['accuracy'])

            elif len(self.y_train) == 3:
                self.logging.info('multi-task with 3 outputs')
                x = lstm_layer(embedded_sequences)
                # shared = Dense(32)(x)
                sub1 = Dense(16)(x)
                sub2 = Dense(16)(x)
                sub3 = Dense(16)(x)
                dropout1 = Dropout(self.dropout)(sub1)
                dropout2 = Dropout(self.dropout)(sub2)
                dropout3 = Dropout(self.dropout)(sub3)

                out1 = Dense(1, activation='sigmoid')(dropout1)

                mid2 = Dense(1, activation='sigmoid')(dropout2)
                con_layer = Concatenate()([out1, mid2])
                out2 = Dense(1, activation='sigmoid')(con_layer)

                mid3 = Dense(1, activation='sigmoid')(dropout3)
                con_layer_3 = Concatenate()([out1, out2, mid3])
                out3 = Dense(1, activation='sigmoid')(con_layer_3)

                model = Model(inputs=comment_input, outputs=[out1, out2, out3])  # , out3])
                model.compile(loss='binary_crossentropy',
                              optimizer=self.optimizer,
                              loss_weights=[4, 1, 1],
                              metrics=['accuracy'])

            elif len(self.y_train) == 5:
                self.logging.info('multi-task with 5 outputs')
                x = lstm_layer(embedded_sequences)

                sub = Dense(16)(x)
                sub1 = Dropout(self.dropout)(sub)

                out2 = Dense(1, activation='sigmoid')(sub1)
                out3 = Dense(1, activation='sigmoid')(sub1)
                out4 = Dense(1, activation='sigmoid')(sub1)
                out5 = Dense(1, activation='sigmoid')(sub1)

                con_layer = Concatenate()([out2, out3, out4, out5, sub1])
                out1 = Dense(1, activation='sigmoid')(con_layer)

                model = Model(inputs=comment_input, outputs=[out1, out2, out3, out4, out5])  # , out3])
                model.compile(loss='binary_crossentropy',
                              optimizer=self.optimizer,
                              loss_weights=[3, 1, 1, 1, 1],
                              metrics=['accuracy'])
            else:
                raise('multi-task support 2 or 3 or 5 output classes')

            self.logging.info(model.summary())

        # single class classification (e.g. Bad-Good)
        # keras sequential model
        elif not self.multi_class_configuration_dict['multi_class_bool']:

            model = Sequential()

            # add embedding layer
            if self.embedding_pre_trained:
                embedding_layer = self._add_pre_trained_embedding()
            else:
                embedding_layer = Embedding(self.max_features, self.embedding_size)  # train word embedding as well

            if not self.attention_configuration_dict['use_attention_bool']:     # "regular"

                model.add(embedding_layer)
                model.add(LSTM(
                    self.lstm_hidden_layer,
                    dropout=self.dropout,
                    recurrent_dropout=self.recurrent_dropout
                ))
                model.add(Dense(1, activation='sigmoid'))

                # try using different optimizers and different optimizer configs
                model.compile(loss='binary_crossentropy',
                              optimizer=self.optimizer,
                              metrics=['accuracy'])
            else:

                self.logging.info('add attention mechanism')

                from keras.layers.core import *
                from keras.layers.recurrent import LSTM
                from keras.models import *
                from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation

                # AttentionWithContext()(lstm_layer)
                model.add(embedding_layer)
                model.add(LSTM(
                    self.lstm_hidden_layer,
                    dropout=self.dropout,
                    recurrent_dropout=self.recurrent_dropout,
                    return_sequences=True))
                model.add(AttentionWithContext())
                model.add(Dense(1, activation='sigmoid'))
                model.compile(loss='binary_crossentropy',
                              optimizer=self.optimizer,
                              metrics=['accuracy'])

            self.logging.info(model.summary())

        return model

    def _add_pre_trained_embedding(self):
        """
            a. load pre trained glove/word2vec
            b. compute embedding matrix
            c. build embedding layer
        :return: embedding layer
        """

        if self.embedding_type['type'] == 'glove':
            self.logging.info('use pre-trained glove word2vec')
            # a. load pre trained glove
            GLOVE_DIR = '../data/golve_pretrained/glove.6B'
            glove_suffix_name = 'glove.6B.' + str(self.embedding_size) + 'd.txt'
            import os
            import numpy as np

            embeddings_index = {}
            f = open(os.path.join(GLOVE_DIR, glove_suffix_name))    # 'glove.6B.100d.txt'))
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            f.close()

            self.logging.info('')
            self.logging.info('Found %s word vectors.' % len(embeddings_index))

            # b. compute embedding matrix
            embedding_matrix = np.zeros((len(self.word_index) + 1, self.embedding_size))
            cnt = 0
            for word, i in self.word_index.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector  # words not found in embedding index will be all-zeros.
                else:
                    self.logging.info('token in train missing in word2vec: ' + str(word))
                    cnt += 1
            self.logging.info('total tokens missing: ' + str(cnt) + ' / ' + str(len(self.word_index)))

            # c. build embedding layer
            from keras.layers import Embedding
            embedding_layer = Embedding(len(self.word_index) + 1,
                                        self.embedding_size,
                                        weights=[embedding_matrix],
                                        input_length=self.maxlen,
                                        trainable=False)

        elif self.embedding_type['type'] == 'gensim':
            self.logging.info('use pre-trained gensim word2vec')

            import gzip
            import gensim
            from keras.layers import Embedding
            import numpy as np

            # fname = '../data/word2vec_pretrained/motors/d_300_k_712904_w_6_e_60_v_motors'
            # fname = '../data/word2vec_pretrained/fashion/d_300_k_1341062_w_6_e_70_v_fashion'

            self.logging.info('load word2vec path: ' + str(self.embedding_type['path']))
            model = gensim.models.Word2Vec.load(self.embedding_type['path'])
            pretrained_weights = model.wv.syn0
            vocab_size, vector_dim = pretrained_weights.shape

            method = 3
            if method == 1:
                self.logging.info('word2vec attempt to fit into embedding layer - middle complex')
                # convert the wv word vectors into a numpy matrix that is suitable for insertion
                # into our TensorFlow and Keras models

                embedding_matrix = np.zeros((len(model.wv.vocab), vector_dim))
                for i in range(len(model.wv.vocab)):
                    embedding_vector = model.wv[model.wv.index2word[i]]
                    if embedding_vector is not None:
                        embedding_matrix[i] = embedding_vector

                embedding_layer = Embedding(input_dim=embedding_matrix.shape[0],
                                            output_dim=embedding_matrix.shape[1],
                                            # input_length=self.maxlen,
                                            weights=[embedding_matrix],
                                            trainable=False)
            elif method == 2:
                self.logging.info('word2vec simple embedding matching - simple complex')
                embedding_layer = Embedding(input_dim=vocab_size,
                                            output_dim=vector_dim,
                                            input_length=self.maxlen,
                                            weights=[pretrained_weights],
                                            trainable=False)
            elif method == 3:

                self.logging.info('word2vec match using word_index from keras tokenizer - as used in glove match above')
                # b. compute embedding matrix

                # sd = 1 / np.sqrt(len(self.word_index) + 1)
                # embedding_matrix = np.random.normal(0, scale=sd, size=(len(self.word_index) + 1, self.embedding_size))

                embedding_matrix = np.zeros((len(self.word_index) + 1, self.embedding_size))
                cnt = 0
                for word, i in self.word_index.items():
                    if word in model.wv:
                        embedding_vector = model.wv[word]
                        embedding_matrix[i] = embedding_vector
                    else:
                        self.logging.info('token in train missing in word2vec: ' + str(word))
                        cnt += 1
                self.logging.info('total tokens missing: ' + str(cnt))


                # c. build embedding layer
                from keras.layers import Embedding
                embedding_layer = Embedding(len(self.word_index) + 1,
                                            self.embedding_size,
                                            weights=[embedding_matrix],
                                            input_length=self.maxlen,
                                            trainable=False)
            else:
                raise ValueError('unknown method value')

        else:
            raise ValueError('unknown embedding type')
        self.logging.info('create glove pre-trained embedding: ' + str(self.embedding_size))
        return embedding_layer

    # create tensor board dir path
    # TODO save sophisticate regards to current fold
    def _create_tensor_board_dir(self):

        if self.tensor_board_bool:
            file_suffix = self._get_file_suffix()

            # tensor board folder with separation by vertical and positive class
            tensor_board_dir = self.tensor_board_dir + \
                               str(self.vertical_type) + '_' + \
                               str(self.df_configuration_dict['y_positive_name']) + '/' + \
                               file_suffix + '_fold=' + str(self.fold_counter)

            import os
            if not os.path.exists(tensor_board_dir):
                os.makedirs(tensor_board_dir)
            return tensor_board_dir
        else:
            return ''

    # run network
    def _fit_model(self, model, tensor_board_dir):

        from keras.callbacks import TensorBoard, EarlyStopping
        from sklearn.metrics import roc_auc_score, roc_curve
        import keras

        self.logging.info('')
        self.logging.info('Train...')

        class RocCallback(keras.callbacks.Callback):

            def __init__(self, training_data, validation_data, logging, file_suffix, vertical_type, batch_size,
                         y_positive_name, fold_counter, multi_class_flag=None, class_names=None):
                self.x = training_data[0]
                self.y = training_data[1]
                self.x_val = validation_data[0]
                self.y_val = validation_data[1]
                self.logging = logging
                self.file_suffix = file_suffix
                self.vertical_type = vertical_type
                self.batch_size = batch_size
                self.y_positive_name = y_positive_name
                self.fold_counter = fold_counter
                self.multi_class_flag = multi_class_flag
                self.class_names = class_names

            def on_train_begin(self, logs={}):
                return

            def on_train_end(self, logs={}):
                return

            def on_epoch_begin(self, epoch, logs={}):
                return

            # plot auc score for current epoch
            def on_epoch_end(self, epoch, logs={}):

                if self.multi_class_flag:
                    for idx, class_name in enumerate(self.class_names):
                        y_pred = self.model.predict(self.x)[idx]
                        auc_train = roc_auc_score(self.y[idx], y_pred)

                        y_pred_val = self.model.predict(self.x_val)[idx]
                        auc_test = roc_auc_score(self.y_val[idx], y_pred_val)

                        test_metric = self.model.evaluate(
                            self.x_val,
                            self.y_val,
                            batch_size=self.batch_size)
                        test_loss = test_metric[idx + 1]
                        test_accuracy = test_metric[idx + 1 + len(self.class_names)]

                        train_metric = self.model.evaluate(
                            self.x,
                            self.y,
                            batch_size=self.batch_size)
                        train_loss = train_metric[idx + 1]
                        train_accuracy = train_metric[idx + 1 + len(self.class_names)]

                        self.logging.info('')
                        self.logging.info('epoch number: ' + str(epoch + 1))

                        self.logging.info('class name: ' + str(self.class_names[idx]))
                        self.logging.info('train:')
                        self.logging.info('train AUC: ' + str(round(auc_train, 3)))
                        self.logging.info('train accuracy: ' + str(round(train_accuracy, 3)))
                        self.logging.info('train loss: ' + str(round(train_loss, 3)))

                        self.logging.info('')
                        self.logging.info('test:')
                        self.logging.info('test AUC: ' + str(round(auc_test, 3)))
                        self.logging.info('test accuracy: ' + str(round(test_accuracy, 3)))
                        self.logging.info('test loss: ' + str(round(test_loss, 3)))

                        fpr_test, tpr_test, thresholds_test = roc_curve(self.y_val[idx], y_pred_val)

                        if self.class_names[idx] == 'review_tag':
                            PredictDescriptionModelLSTM.store_roc_results(self.fold_counter, fpr_test, tpr_test, auc_test,
                                                                          epoch + 1, self.logging)

                        PredictDescriptionModelLSTM.plot_roc_curve(fpr_test, tpr_test, auc_test, self.class_names[idx], file_suffix,
                                                                   self.logging, epoch=epoch + 1,
                                                                   vertical_type=self.vertical_type,
                                                                   y_positive_name=self.y_positive_name,
                                                                   fold_counter=self.fold_counter,
                                                                   class_name=self.class_names[idx])

                else:
                    y_pred = self.model.predict(self.x)
                    auc_train = roc_auc_score(self.y, y_pred)

                    y_pred_val = self.model.predict(self.x_val)
                    auc_test = roc_auc_score(self.y_val, y_pred_val)

                    test_loss, test_accuracy = self.model.evaluate(
                        self.x_val,
                        self.y_val,
                        batch_size=self.batch_size)

                    train_loss, train_accuracy = self.model.evaluate(
                        self.x,
                        self.y,
                        batch_size=self.batch_size)

                    self.logging.info('')
                    self.logging.info('epoch number: ' + str(epoch+1))

                    self.logging.info('')
                    self.logging.info('train:')
                    self.logging.info('train AUC: ' + str(round(auc_train, 3)))
                    self.logging.info('train accuracy: ' + str(round(train_accuracy, 3)))
                    self.logging.info('train loss: ' + str(round(train_loss, 3)))

                    self.logging.info('')
                    self.logging.info('test:')
                    self.logging.info('test AUC: ' + str(round(auc_test, 3)))
                    self.logging.info('test accuracy: ' + str(round(test_accuracy, 3)))
                    self.logging.info('test loss: ' + str(round(test_loss, 3)))

                    fpr_test, tpr_test, thresholds_test = roc_curve(self.y_val, y_pred_val)

                    PredictDescriptionModelLSTM.store_roc_results(self.fold_counter, fpr_test, tpr_test, auc_test,
                                                                  epoch+1, self.logging)

                    PredictDescriptionModelLSTM.plot_roc_curve(fpr_test, tpr_test, auc_test, 'test', file_suffix,
                                                               self.logging, epoch=epoch+1,
                                                               vertical_type=self.vertical_type,
                                                               y_positive_name=self.y_positive_name,
                                                               fold_counter=self.fold_counter)

                confusion_matrix_bool = False
                if confusion_matrix_bool:
                    import itertools
                    list2d_val = self.model.predict_classes(self.x_val)
                    list1d_val = list(itertools.chain.from_iterable(list2d_val))

                    list2d_train = self.model.predict_classes(self.x)
                    list1d_train = list(itertools.chain.from_iterable(list2d_train))

                    self.logging.info('************************************************************')
                    self.logging.info('test sum prediction: ' + str(sum(list1d_val)))
                    self.logging.info('train sum prediction: ' + str(sum(list1d_train)))
                    self.logging.info('************************************************************')

                return

            def on_batch_begin(self, batch, logs={}):
                return

            def on_batch_end(self, batch, logs={}):
                return

        # run model when tensor board is True
        if self.tensor_board_bool:
            tensor_board = TensorBoard(
                log_dir=tensor_board_dir,  # './Graph',
                histogram_freq=0,          # if not 0, file will be very large
                write_graph=False,
                write_images=False
            )

            file_suffix = self._get_file_suffix()       # to send into callback to save ROC plot

            # TODO check
            import numpy as np

            class_names = self.multi_class_configuration_dict['multi_class_label']
            '''self.y_train = [self.y_train['review_tag'],
                            self.y_train['subjective_sentence'],
                            self.y_train['missing_context']]
            self.y_test = [self.y_test['review_tag'],
                            self.y_test['subjective_sentence'],
                            self.y_test['missing_context']]'''

            model.fit(self.x_train,
                      self.y_train,
                      batch_size=self.batch_size,
                      epochs=self.num_epoch,
                      validation_data=(self.x_test, self.y_test),
                      shuffle=True,
                      callbacks=[
                          tensor_board,     # tensor board object to store data

                          # parameters to callback class
                          RocCallback(training_data=(self.x_train, self.y_train),
                                      validation_data=(self.x_test, self.y_test),
                                      logging=self.logging,
                                      file_suffix=file_suffix,
                                      vertical_type=self.vertical_type,     # to insert into relevant folder
                                      batch_size=self.batch_size,
                                      y_positive_name=self.df_configuration_dict['y_positive_name'],
                                      fold_counter=self.fold_counter,
                                      multi_class_flag=self.multi_class_configuration_dict['multi_class_bool'],
                                      class_names=class_names),

                          # add early stopping
                          EarlyStopping(monitor='val_loss',
                                        min_delta=0,
                                        patience=self.patience,  # 6 epochs with no improvement needs to stop model
                                        verbose=2,
                                        mode='auto')
                      ]
                      # we add this to store tensor board data
                      )
        else:
            raise('currently only support with tensorbpard output')

        return

    # evaluate model on test data
    # accuracy and AUC metrics
    def _evaluation(self, model):

        # test_loss, test_accuracy = model.evaluate(self.x_test, self.y_test, batch_size=self.batch_size)
        # train_loss, train_accuracy = model.evaluate(self.x_train, self.y_train, batch_size=self.batch_size)

        # attention visualization if flag is True
        if self.attention_configuration_dict['use_attention_bool']:
            try:
                self._plot_html_attention_contribute(model)

            except Exception as e:
                self.logging.info('')
                self.logging.info('attention fail to present')
                self.logging.info('exception: ' + str(e))
                self.logging.info('')
                pass

        return

    # update AUC folder regards to epoch with best auc score
    def _update_folder_name(self):

        global global_auc_list
        max_auc = max(global_auc_list)      # max auc epoch score

        import os
        self.logging.info('')
        self.logging.info('max test auc: ' + str(round(max_auc, 3)))

        # change dir name (add best auc score)
        file_suffix = self._get_file_suffix()

        #  new directory name with AUC score
        new_dir = '../results/ROC/' + \
                  str(self.vertical_type) + '_' + str(self.df_configuration_dict['y_positive_name']) + '/' + \
                  file_suffix + '/' + \
                  str(round(max_auc, 3)) + '_' + str(self.fold_counter) + '/'

        os.rename('../results/ROC/' +
                  str(self.vertical_type) + '_' + str(self.df_configuration_dict['y_positive_name']) + '/' +
                  file_suffix + '/' +
                  str(self.fold_counter) + '/',
                  new_dir)

        self.logging.info('')
        self.logging.info('change dir name: ' + new_dir)

        return

    # compute auc
    # produce ROC plot
    def _compute_roc_produce_plot(self, model):

        from sklearn.metrics import roc_auc_score, roc_curve

        y_test_pred = model.predict_proba(self.x_test)
        y_train_pred = model.predict_proba(self.x_train)

        auc_test = roc_auc_score(self.y_test, y_test_pred)
        auc_train = roc_auc_score(self.y_train, y_train_pred)

        fpr_test, tpr_test, thresholds_test = roc_curve(self.y_test, y_test_pred)
        # fpr_test, tpr_test, thresholds_test = roc_curve(self.y_test, y_test_pred)
        file_suffix = self._get_file_suffix()
        PredictDescriptionModelLSTM.plot_roc_curve(fpr_test, tpr_test, auc_test, 'test', file_suffix, self.logging,
                                                   epoch=self.num_epoch, vertical_type=self.vertical_type,
                                                   y_positive_name=self.df_configuration_dict['y_positive_name'],
                                                   fold_counter=self.fold_counter)

        return auc_test, auc_train

    # plot ROC plot
    @classmethod
    def plot_roc_curve(cls, fpr, tpr, auc, type_data, file_suffix, logging, epoch, vertical_type, y_positive_name,
                       fold_counter, class_name=None):

        global global_auc_list  # update global variable
        global_auc_list.append(auc)

        logging.info('*****************************  auc  *************************************')
        logging.info('global auc: ' + str(global_auc_list))

        import matplotlib.pyplot as plt
        plt.figure()
        lw = 2
        plt.plot(fpr,
                 tpr,
                 color='darkorange',
                 lw=lw,
                 label='ROC curve (area = %0.3f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.01])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC - ' + str(type_data))
        plt.legend(loc="lower right")
        import os
        plot_dir = '../results/ROC/' +\
                   str(vertical_type) + '_' + str(y_positive_name) + '/' \
                   + str(file_suffix) + '/' \
                   + str(fold_counter) + '/'

        if class_name is not None:
            plot_dir = plot_dir + str(class_name) + '/'

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        plot_path = plot_dir \
                    + str(round(auc, 3)) + \
                    '_epoch=' + str(epoch) + '_' + \
                    str(type_data)

        plt.savefig(plot_path + '.png')
        plt.close()
        logging.info('save ROC plot: ' + str(plot_path))
        return

    # store statistic results (later after all fold will finished to run, I'll calculate meta statistics)
    @classmethod
    def store_roc_results(cls, fold_counter, fpr_test, tpr_test, auc_test, epoch, logging):

        global global_statistic_auc_dict  # update global variable

        # if fold_counter not in global_statistic_auc_dict:
        #     global_statistic_auc_dict[fold_counter] = {}

        global_statistic_auc_dict[epoch] = {
            'auc': auc_test,
            'fpr': fpr_test,
            'tpr': tpr_test
        }

        global global_max_auc_epoch_dict
        if auc_test > global_max_auc_epoch_dict['auc']:
            global_max_auc_epoch_dict['auc'] = auc_test
            global_max_auc_epoch_dict['epoch'] = epoch

        logging.info('')
        logging.info('store statistic roc results, epoch number: ' + str(epoch))

        return

    # @classmethod
    # def calculate_confusion_matrix(cls):

    # return

    # build file suffix for tensor board and roc using model configuration
    def _get_file_suffix(self):
        file_suffix = 'sen_len=' + str(self.maxlen) + \
                        '_batch=' + str(self.batch_size) + \
                        '_optimizer=' + str(self.optimizer) + \
                        '_embedding=' + str(self.embedding_size) + \
                        '_lstm_hidden=' + str(self.lstm_hidden_layer) + \
                        '_pre_trained=' + str(self.embedding_pre_trained) + \
                        '_pre_trained_type=' + str(self.embedding_type['type']) + \
                        '_epoch=' + str(self.num_epoch) + \
                        '_dropout=' + str(self.dropout) + \
                        '_multi=' + str(self.multi_class_configuration_dict['multi_class_bool']) + \
                        '_attention=' + str(self.attention_configuration_dict['use_attention_bool']) + \
                        '_time=' + str(self.cur_time)
        return file_suffix

    ##################  HTML attention  ##################

    def _keras_get_alpha_vector_attention(self, x, W, b, u, mask=None, bias=True):

        uit = K.dot(x, W)

        if bias:
            uit += b

        uit = K.tanh(uit)

        mul_a = uit * u  # with this
        ait = K.sum(mul_a, axis=2)  # and this

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number epsilon to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)

        return a

    def _numpy_get_alpha_vector_attention(self, x, W, b, u, mask=None, bias=True):

        import numpy as np

        uit = np.dot(x, W)

        if bias:
            uit += b

        uit = np.tanh(uit)

        mul_a = uit * u  # with this
        ait = np.sum(mul_a, axis=2)  # and this

        a = np.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= np.cast(mask, np.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number epsilon to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        # a /= np.cast(np.sum(a, axis=1, keepdims=True) + np.epsilon(), np.floatx())
        a /= np.sum(a, axis=1, keepdims=True)

        # a = np.expand_dims(a)

        return a

    # attention over test sentences
    def _plot_html_attention_contribute(self, model):   # , sentence_word, sentence_contribute):

        file_suffix = self._get_file_suffix()
        file_dir = '../results/html/' + str(file_suffix) + '/'

        global global_max_auc_epoch_dict
        max_auc = global_max_auc_epoch_dict['auc']
        file_path = file_dir + 'fold=' + str(self.fold_counter) + '_auc=' + str(round(max_auc,3)) + '.html'
        import os
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        # word cont dict - learn from corpus - self.word_cont_dict
        # TODO change gradient for all sentences (not for every sentence)
        with open(file_path, 'w') as myFile:
            myFile.write('<html>')
            myFile.write('<body>')

            import numpy as np

            all_pred = model.predict(self.x_test)
            self.create_red_green_gradient(all_pred.flatten())

            for i in range(200):
                self.logging.info('')
                self.logging.info('get activation over test sentence: i=' + str(i+1))

                from keras import backend as K

                inp = model.input  # input placeholder
                outputs = [layer.output for layer in model.layers]  # all layer outputs
                functor = K.function([inp] + [K.learning_phase()], outputs)  # evaluation function

                # Testing
                test = self.x_test[i][np.newaxis, :]
                layer_outs = functor([test, 1.])

                x = layer_outs[1]       # lstm output

                import numpy as np

                lstm_out = x  # tf.convert_to_tensor(x, np.float32)

                for layer in model.layers:
                    if layer.name.startswith('attention_with_context_'):
                        W, b, u = layer.get_weights()[0], layer.get_weights()[1], layer.get_weights()[2]
                        a = self._numpy_get_alpha_vector_attention(lstm_out, W, b, u)
                        attention_vector = a[0]
                        self.logging.info('attention_vector: ' + str(attention_vector))

                sentence_contribute = []
                sentence_word = []

                for token_place, word_index in enumerate(self.x_test[i]):

                    # padding (sentence length shorter than max length)
                    if word_index == 0:
                        self.logging.info('word: ' + ', attention val: ' + str(round(attention_vector[token_place], 4)))
                        sentence_word.append(' _ ')
                        sentence_contribute.append(attention_vector[token_place])

                    # word has index
                    else:
                        self.logging.info('word: ' + str(self.index_word_dict[word_index]) + ', attention val: ' + str(
                            round(attention_vector[token_place], 4)))
                        sentence_word.append(self.index_word_dict[word_index])
                        sentence_contribute.append(attention_vector[token_place])

                self.create_red_white_gradient(sentence_contribute)
                # write sentence with background color
                myFile.write('<p>')
                myFile.write('<span> ' + str(i) + ':' + '</span>')
                # TODO add good/bad sentences

                for idx, cur_word in enumerate(sentence_word):
                    cur_word_cont = sentence_contribute[idx]
                    cur_background = self.get_background_color(cur_word_cont)
                    myFile.write('<span style="background-color: ' + str(cur_background) + ';opacity: 0.8;">' +
                                     str(' ') + str(cur_word) + str('     ') +
                                     '</span>')

                cur_review_failure_reason = self.test_reason.iloc[i]

                # if failure reason is not nan (a string)
                if isinstance(cur_review_failure_reason, basestring):
                    myFile.write('<span> ,Group: ' + str(self.y_test.iloc[i]) + ', reason: ' + str(cur_review_failure_reason) +'</span>')
                else:
                    myFile.write('<span> ,Group: ' + str(self.y_test.iloc[i]) + '</span>')

                # add predict value

                y_pred = all_pred[i][0]
                cur_background = self.get_background_color_proba(y_pred)

                myFile.write('<span style="background-color: ' + str(cur_background) + ';opacity: 0.8;">' +
                             ', predict:' + str(round(y_pred, 3)) + '</span>')
                myFile.write('</p>')

            myFile.write('</body>')
            myFile.write('</html>')
            myFile.close()

        self.logging.info('save html file: ' + str(file_path))

        return

    ################## background color for sentence attention HTML ######################

    # create list of color and percentile settings
    def create_red_white_gradient(self, sentence_contribute):
        from colour import Color
        red = Color("red")
        white = Color("white")
        self.color_list = list(white.range_to(red, 10))              # 10 color correspond to percentile values
        self.percentile_list = self.percentile_color(sentence_contribute)               # 10 diff percentile values
        return

    # get percentile of word contribute values
    def percentile_color(self, cur_sentence_contribute):
        import numpy as np
        c_l = cur_sentence_contribute
        percentile_list = [
            np.percentile(c_l, 10),
            np.percentile(c_l, 20),
            np.percentile(c_l, 30),
            np.percentile(c_l, 40),
            np.percentile(c_l, 50),
            np.percentile(c_l, 60),
            np.percentile(c_l, 70),
            np.percentile(c_l, 80),
            np.percentile(c_l, 90),
            np.percentile(c_l, 100)
        ]
        self.logging.info('Percentile contribute list: ' + str(percentile_list))
        return percentile_list

    # determine background color of word regard to personality contribute value
    def get_background_color(self, contribute):
        for per_idx, per_val in enumerate(self.percentile_list):
            if contribute <= per_val:
                hex_color = self.color_list[per_idx]
                return hex_color

    def create_red_green_gradient(self, y_pred_list):
        from colour import Color
        red = Color("red")
        green = Color("green")
        self.color_list_proba = list(red.range_to(green, 10))  # 10 color correspond to percentile values
        self.percentile_list_proba = self.percentile_color_proba(y_pred_list)  # 10 diff percentile values
        return

    # get percentile of word contribute values
    def percentile_color_proba(self, cur_sentence_contribute):
        import numpy as np
        c_l = cur_sentence_contribute
        percentile_list = [
            np.percentile(c_l, 10),
            np.percentile(c_l, 20),
            np.percentile(c_l, 30),
            np.percentile(c_l, 40),
            np.percentile(c_l, 50),
            np.percentile(c_l, 60),
            np.percentile(c_l, 70),
            np.percentile(c_l, 80),
            np.percentile(c_l, 90),
            np.percentile(c_l, 100)
        ]
        self.logging.info('Percentile contribute list proba: ' + str(percentile_list))
        return percentile_list

    # determine background color of word regard to personality contribute value
    def get_background_color_proba(self, contribute):
        for per_idx, per_val in enumerate(self.percentile_list_proba):
            if contribute <= per_val:
                hex_color = self.color_list_proba[per_idx]
                return hex_color

def main():
    raise ('currenlty you can only run script from train,py')


if __name__ == '__main__':
    main()
