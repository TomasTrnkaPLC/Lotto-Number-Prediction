import sys
from datetime import datetime
import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, Embedding, Concatenate
from tensorflow.keras.layers import SpatialDropout1D, Lambda, Bidirectional, concatenate
from tensorflow.keras.layers import Attention
from tensorflow.keras import callbacks
from tensorflow.keras import backend
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import logging
import csv

# Variables
#options for data compare if you already know data and you want to know if they match
# from the last draw. It is not necessary to use it. It is only for a more accurate verification of the prediction model against reality
data_compare_array_set = [17,39,22,46,6]
EPOCHS = 20000
BATCH_SIZE = 100 #500 CPU 2000+ Based on RAM and hidden_nerons
LR_MAX = 0.0001
LR_MIN = 0.000001
# Category embedding dimension
embed_dim_value = 200
#Density of the category embedding dimension
dense_value = 100
#dimension continuous-number vector for each number
embed_dim = (embed_dim_value // 2) + 1
dropout_rate = 0.2
spatial_dropout_rate = 0.2
w = 5
steps_after = 5
feature_count = embed_dim * 5
# How many neurons in each layer of the LSTM
hidden_neurons = [64,32] # [128,64], [64,32], [32,16] 
# On/Off Bidirectional LSTM
bidirectional = True 
# sgd or adam
optimizer_selected = 'adam'
# Loose sparse_categorical_crossentropy ,  mse
loose_selected = 'sparse_categorical_crossentropy'
#By setting verbose 0, 1 or 2 you just say how do you want to 'see' the training progress for each
varbose_param = 1
# how many loops to predict
beam_width = 5
# dont change just for loop
#minimal_limit for successful prediction
minimal_limit = 54.00
run = 1
# patience_param means stop training when a monitored metric has stopped improving
patience_param = 50

class EarlyStoppingCallback:
    def __init__(self, monitor='val_loss', patience=5):
        self.early_stopping = EarlyStopping(monitor=monitor, patience=patience)

    def on_train_begin(self, logs=None):
        self.best_val_loss = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get('val_loss')
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss
        if self.early_stopping.on_epoch_end(epoch=epoch, logs=logs):
            print('Early stopping due to no improvement on validation loss.')
            self.model.stop_training = True
# Class for Cosine Annealing Scheduler
class CosineAnnealingScheduler(callbacks.Callback):
        """Cosine annealing scheduler.
        """
        def __init__(self, T_max, eta_max, eta_min = 0, verbose = 0):
            super(CosineAnnealingScheduler, self).__init__()
            self.T_max = T_max
            self.eta_max = eta_max
            self.eta_min = eta_min
            self.verbose = verbose

        def on_epoch_begin(self, epoch, logs = None):
            if not hasattr(self.model.optimizer, 'lr'):
                raise ValueError('Optimizer must have a "lr" attribute.')
            lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
            backend.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                    'rate to %s.' % (epoch + 1, lr))
            
        def on_epoch_end(self, epoch, logs = None):
            logs = logs or {}
            logs['lr'] = backend.get_value(self.model.optimizer.lr)
            
AUTO = tf.data.experimental.AUTOTUNE
version = '0.0.3'
#For logging purposes
output_path = os.path.dirname(os.path.abspath(__file__))
time_now = datetime.now()
current_time_day = time_now.strftime("%d_%m_%Y")
current_time_day = str(current_time_day) 
current_time = time_now.strftime("%H_%M_%S")
current_time = str(current_time) 
logging_file_string = 'Lotto_Number_Predictor_'+current_time_day+'.log'
csv_file_string = 'Lotto_Number_Predictor_'+current_time_day+'.csv'
logging_file = os.path.join('C:\Temp'+logging_file_string) 
filepath = os.path.join('C:\Temp', logging_file_string)
if (os.path.exists(csv_file_string) == False):
    f = open(csv_file_string, "w")
try:
    if not os.path.exists('C:\Temp'):
        os.makedirs('C:\Temp')        
    logging.basicConfig(filename=filepath , filemode='a', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    logging.warning('Lotto Number Predictor %s Started ', version)
    logging.warning('https://github.com/TomasTrnkaPLC')
except Exception as e:
    print(e)
    if getattr(sys, 'frozen', False): # we are running in a bundle
      output_path = sys._MEIPASS # This is where the files are unpacked to
    else: # normal Python environment
      output_path = os.path.dirname(os.path.abspath(__file__))
    print(output_path)
    start = True
    print('Logging failed')

# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)
MIXED_PRECISION = False
XLA_ACCELERATE = True

if MIXED_PRECISION:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    if tpu: policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    else: policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print('Mixed precision enabled')
    logging.warning('Mixed precision enabled')

if XLA_ACCELERATE:
    tf.config.optimizer.set_jit(True)
    print('Accelerated Linear Algebra enabled')
    logging.warning('Accelerated Linear Algebra enabled')
    
print('Welcome to Lotto Number Predictor ',version,' by @tomastrnkaplc')
print('System Patch: ',output_path)  

# beam search
def beam_search_decoder(data, k, replace = True):
        sequences = [[list(), 0.0]]
        # walk over each step in sequence
        for row in data:
            all_candidates = list()
            # expand each current candidate
            for i in range(len(sequences)):
                seq, score = sequences[i]
                best_k = np.argsort(row)[-k:]
                for j in best_k:
                    candidate = [seq + [j], score + math.log(row[j])]
                    if replace:
                        all_candidates.append(candidate)
                    elif (replace == False) and (len(set(candidate[0])) == len(candidate[0])):
                        all_candidates.append(candidate)
            # order all candidates by score
            ordered = sorted(all_candidates, key = lambda tup:tup[1], reverse = True)
            # select k best
            sequences = ordered[:k]
        return sequences

def predict_loop():
   try:
    file = open(csv_file_string, mode='a', newline='')
    writer = csv.writer(file)

    time_now = datetime.now()
    current_time_day = time_now.strftime("%d_%m_%Y")
    current_time_day = str(current_time_day) 
    current_time = time_now.strftime("%H_%M_%S")
    current_time = str(current_time) 
    # Load data    
    lotto = pd.read_csv(output_path+'/data.csv', index_col = 'Datum')
    data = lotto.values - 1
    # how many percet is data for training/test
    train = data[:-50]
    test = data[-50:]
    X_train = []
    y_train = []
    for i in range(w, len(train)):
        X_train.append(train[i - w: i, :])
        y_train.append(train[i])
    X_train, y_train = np.array(X_train), np.array(y_train)

    inputs = data[data.shape[0] - test.shape[0] - w:]
    X_test = []
    for i in range(w, inputs.shape[0]):
        X_test.append(inputs[i - w: i, :])
    X_test = np.array(X_test)
    y_test = test

    # Create layers
    with strategy.scope():   
        inp0 = Input(shape = (w, X_train.shape[2]))
        inp = [Embedding(embed_dim_value, embed_dim)(Lambda(lambda x: x[:, :, i])(inp0)) for i in range(X_train.shape[2])]
        inp = [SpatialDropout1D(spatial_dropout_rate)(inp[i]) for i in range(X_train.shape[2])]
        inp = Concatenate()(inp)
        
        # Seq2Seq model with attention or bidirectional encoder    
        num_layers = len(hidden_neurons)    
        sh_list, h_list, c_list = [inp], [], []    
        if bidirectional:       
            for i in range(num_layers):
                sh, fh, fc, bh, bc = Bidirectional(LSTM(hidden_neurons[i],
                                                        dropout = dropout_rate, 
                                                        return_state = True, 
                                                        return_sequences = True))(sh_list[-1])
                h = Concatenate()([fh, bh])
                c = Concatenate()([fc, bc]) 
                sh_list.append(sh)
                h_list.append(h)
                c_list.append(c)        
        else:
            for i in range(num_layers):

                sh, h, c = LSTM(hidden_neurons[i], 
                                dropout = dropout_rate,
                                return_state = True, 
                                return_sequences = True)(sh_list[-1])

                sh_list.append(sh)
                h_list.append(h)
                c_list.append(c)
        
        decoder = RepeatVector(steps_after)(h_list[-1])
        if bidirectional:
            decoder_hidden_neurons = [hn * 2 for hn in hidden_neurons]
        else:
            decoder_hidden_neurons = hidden_neurons
        
        for i in range(num_layers):
            decoder = LSTM(decoder_hidden_neurons[i],
                        dropout = dropout_rate, 
                        return_sequences = True)(decoder, initial_state = [h_list[i], c_list[i]])
        
        context = Attention(dropout = dropout_rate)([decoder, sh_list[-1]])
        decoder = concatenate([context, decoder])
        out = Dense(dense_value, activation = 'softmax')(decoder)
        model = Model(inputs = inp0, outputs = out) 

        cas = CosineAnnealingScheduler(EPOCHS, LR_MAX, LR_MIN)
        # This help you to save only the best model. It may happen that the result oscillates and therefore the last model does not have the best parameters. This helps to save the best result
        ckp = callbacks.ModelCheckpoint('best_model.hdf5', monitor = 'val_sparse_top_k', verbose = 1, 
                                    save_best_only = True, save_weights_only = False, mode = 'max')
        early_stop = EarlyStopping(monitor='val_loss', patience=patience_param, verbose=1)
        sparse_top_k = tf.keras.metrics.SparseTopKCategoricalAccuracy(k = 1, name = 'sparse_top_k')
        model.compile(optimizer = optimizer_selected, loss = loose_selected, metrics = [sparse_top_k]) 
        history = model.fit(X_train, y_train, 
                        validation_data = (X_test, y_test), 
                        callbacks = [cas, early_stop],
                        epochs = EPOCHS, 
                        batch_size = BATCH_SIZE, 
                        verbose = varbose_param) 

    # Model summary and end 
    model.summary() 

    # Start Preidction
    pred = model.predict(X_test)
    pred = np.argmax(pred, axis = 2)
    X_latest = X_test[-1][1:]
    X_latest = np.concatenate([X_latest, y_test[-1].reshape(1, 5)], axis = 0)
    X_latest = X_latest.reshape(1, X_latest.shape[0], X_latest.shape[1])
   
    pred_latest = model.predict(X_latest)
    pred_latest = np.squeeze(pred_latest)
    replace = False
    result = beam_search_decoder(pred_latest, beam_width, replace)
    print('Our top 10 predictions for the next round: ')
    logging.warning('Our top 10 predictions for the next round: ')

    for seq in result:
        print('Predicted: ', np.array(seq[0]) + 1, '\t Probability of success: ', round(seq[1], 3), '%')
        number = np.array(seq[0]) + 1
        number = np.array_str(number)
        percentage = seq[1]
        percentage = round(percentage, 2) 
        percentage = str(percentage)
        logging.warning('Predicted %s , Probability of success: %s', number, percentage)
        try:
            #This is only to verify the prediction if we already know the numbers in advance. It helps to quickly validate the accuracy of the prediction against real numbers. 
            comparation_result = set(data_compare_array_set) & set(np.array(seq[0]) + 1)
            percentages = len(comparation_result)/5 *100
            percentages = round(percentages, 2)
            percentages_str = str(percentages)
            comparation_result = ' '.join(str(x) for x in comparation_result)
            print('Match number: ', comparation_result, '\t Real Match: ', percentages_str, '%')
            logging.warning('Match number %s ,Real Match %s ', comparation_result, percentages_str)
            if (percentages < 80):
               logging.warning('The prediction is not good enough. < 80 % We need to improve the model')
            row = [run] + ['Number']+ [value for value in seq[0]] + ['Result'] + [percentages_str] 
            writer.writerow(row) 
     
        except Exception as e:
            print(e)
            logging.warning(e)
    file.close()
   except Exception as e:
      # Save Parameter
    logging.warning('----Calulation error')   
    logging.warning('----Dropout rate %s', dropout_rate)
    logging.warning('----Hidden_neurons %s', hidden_neurons)
    logging.warning('----Category embedding dimension %s', embed_dim_value)
    logging.warning('----Error %s', e)
# Save Parameter
logging.warning('Model Parameter set')
logging.warning('EPOCHS %s', EPOCHS)
logging.warning('BATCH_SIZE %s', BATCH_SIZE)
logging.warning('LR_MAX %s', LR_MAX)
logging.warning('LR_MIN %s', LR_MIN)
logging.warning('Category embedding dimension %s', embed_dim_value)
logging.warning('Dropout rate %s', dropout_rate)
logging.warning('Hidden_neurons %s', hidden_neurons)
logging.warning('Optimizer %s', optimizer_selected)
logging.warning('Loose %s', loose_selected)      
logging.warning('Patience %s', patience_param)
# For CSV loggin

loop = 1    
run = 1
while 1:
   logging.warning('Run number %s', run)
   logging.warning('Embedding dimension %s', embed_dim_value)
   logging.warning('Density %s', dense_value)
   logging.warning('Hidden_neurons %s', hidden_neurons)
   print('Run number', run)
   if embed_dim_value < 1000:
    embed_dim_value = embed_dim_value + 100
   else:
     embed_dim_value = 200
     if dense_value < 1000:
         dense_value = dense_value + 100
     else:
         if loop == 1:
             hidden_neurons = [128,64] # [128,64], [64,32], [32,16] 
             loop = 2 
             dense_value = 200
             embed_dim_value = 200
             
         elif loop == 2:
             hidden_neurons = [256,158] # [128,64], [64,32], [32,16] 
             loop = 3 
             dense_value = 200
             embed_dim_value = 200    
         elif loop == 3:
             hidden_neurons = [512,256] # [128,64], [64,32], [32,16] 
             loop = 4 
             dense_value = 200
             embed_dim_value = 200        
         else:    
            print('End of the prediction')
            logging.warning('End of the prediction')
            break    
   predict_loop()
   run = run +1
# Close the file

# Načítanie dát zo súboru csv
with open(csv_file_string) as csvfile:
    reader = csv.reader(csvfile)
    data = [row for row in reader]

data.sort(key=lambda x: x[-1], reverse=True)
print('Run', 'Result %')
for row in data:
    if float(row[-1]) >= minimal_limit:
        print(row[0], row[-1])
