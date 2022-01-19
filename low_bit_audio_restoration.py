import numpy as np 
import matplotlib.pyplot as plt
import librosa as lr
import librosa.display
from tensorflow.keras import backend as K 

import tensorflow as tf
import os
from tensorflow.keras.optimizers import Adam


'''load compressed clips'''
data_path_compressed = '/home/navin/op/dataset_500clps/compressed/'
clips_compressed = os.listdir(data_path_compressed)
clips_compressed.sort()
one = clips_compressed[0]
clips_compressed = clips_compressed[0:200]

'''load original clips'''
data_path_original = '/home/navin/op/dataset_500clps/original/'
clips_original = os.listdir(data_path_original)
clips_original.sort()
clips_original = clips_original[0:200] 

n_fft = 1024
hop_length = 512

'''extract feature compressed'''
compressed_holder_time = []
compressed_holder_freq = []
for each in clips_compressed:
    compressed, sr = lr.load(data_path_compressed   +each, sr = 44100)
    
    stft1 = lr.stft(compressed, hop_length = hop_length, n_fft = n_fft)
    spectrogram1 = np.abs(stft1)**2
    log_spectrogram = lr.power_to_db(spectrogram1) 
    log_spectrogram1 = log_spectrogram.T
    for i in range(int(log_spectrogram1.shape[0]/100)):
        data = log_spectrogram1[i*100: (i+1)*100]
        compressed_holder_time.append(data)
        compressed_holder_freq.append(data.T)
        
        
'''extract feature original'''
original_holder = []        
for each in clips_original:
    original, sr = lr.load(data_path_original +each, sr = 44100)
    stft2 = lr.stft(original, hop_length = hop_length, n_fft = n_fft)
    spectrogram2 = np.abs(stft2)**2
    log_spectrogram_original = lr.power_to_db(spectrogram2)
    log_spectrogram_original1 = log_spectrogram_original.T
    for i in range(int(log_spectrogram_original1.shape[0]/100)):
        data = log_spectrogram_original1[i*100: (i+1)*100]
        original_holder.append(data)
#%%
'''Training, Validation and Test Dataset'''
compressed_time = np.array(compressed_holder_time)
max_c = np.max(compressed_time)
min_c = np.min(compressed_time)
div = max_c-min_c
compressed_time1 = np.divide(np.subtract(compressed_time, min_c), div)
c_time_train = compressed_time1[0:4000]
c_time_vali = compressed_time1[4000:]


compressed_freq = np.array(compressed_holder_freq)
compressed_freq1 = np.divide(np.subtract(compressed_freq, min_c), div)
c_freq_train = compressed_freq1[0:4000]
c_freq_vali = compressed_freq1[4000:]


original = np.array(original_holder)
max_ori = np.max(original)
min_ori = np.min(original)
div_ori = max_ori-min_ori
original1 = np.divide(np.subtract(original, min_ori), div_ori)
o_train = original1[0:4000]
o_vali = original1[4000:]
o_test = original1[4975:]
#%%
'''Calculate SNR'''
def SNR(y_true, y_pred):
    numerator = K.sum((y_true)**2)
    denominator = K.sum(K.abs(y_true - y_pred)**2)
    snr1 = 10 * (K.log(numerator) - K.log(denominator))
    snr = snr1/2.3025
    return snr
#%%
'TF-LSTM model'
#c_time_train.shape[2]
t_model_input = tf.keras.layers.Input(shape = (c_time_train.shape[1], c_time_train.shape[2]))
t = tf.keras.layers.LSTM(c_time_train.shape[2], activation = 'relu', return_sequences = True)(t_model_input)
t = tf.keras.layers.BatchNormalization()(t)
t = tf.keras.layers.LSTM(c_time_train.shape[2], return_sequences = True, activation = 'relu')(t)
t = tf.keras.layers.BatchNormalization()(t)
t = tf.keras.layers.LSTM(c_time_train.shape[2], return_sequences = True, activation = 'relu')(t)
t = tf.keras.layers.BatchNormalization()(t)
t = tf.keras.layers.LSTM(c_time_train.shape[2], return_sequences = True, activation = 'relu')(t)
t = tf.keras.layers.BatchNormalization()(t)
t = tf.keras.layers.LSTM(c_time_train.shape[2], return_sequences = True, activation = 'relu')(t)

f_model_input = tf.keras.layers.Input(shape = (c_freq_train.shape[1], c_freq_train.shape[2]))
f = tf.keras.layers.LSTM(c_freq_train.shape[2], activation = 'relu', return_sequences = True)(f_model_input)
f = tf.keras.layers.BatchNormalization()(f)
f = tf.keras.layers.LSTM(c_freq_train.shape[2], return_sequences = True, activation = 'relu')(f)
f = tf.keras.layers.BatchNormalization()(f)
f = tf.keras.layers.LSTM(c_freq_train.shape[2], return_sequences = True, activation = 'relu')(f)
f = tf.keras.layers.BatchNormalization()(f)
f = tf.keras.layers.LSTM(c_freq_train.shape[2], return_sequences = True, activation = 'relu')(f)
f = tf.keras.layers.BatchNormalization()(f)
f = tf.keras.layers.LSTM(c_freq_train.shape[2], return_sequences = True, activation = 'relu')(f)
f = tf.keras.layers.Permute((2,1))(f)

add = tf.keras.layers.Add()([t,f])
add = tf.keras.layers.LSTM(c_time_train.shape[2], return_sequences = True, activation = 'relu')(add)
add = tf.keras.layers.BatchNormalization()(add)
TF_LSTM_model = tf.keras.models.Model([t_model_input,f_model_input], add)
TF_LSTM_model.summary()

'''Optimizer'''
opt = Adam(lr=0.00001 , beta_1=0.9, beta_2=0.999)
TF_LSTM_model.compile(optimizer= opt, loss = 'mse', metrics = ['accuracy', SNR])
#%%
'''Training'''
history = TF_LSTM_model.fit([c_time_train, c_freq_train], o_train, epochs=200, batch_size =32,  validation_data=([c_time_vali, c_freq_vali],o_vali))
#%%
'''Testing the Model '''
test_data = clips_compressed[199:200]
test_sample, sr = lr.load(data_path_compressed + test_data[0], sr = 44100)
test_stft = lr.stft(test_sample, hop_length = hop_length, n_fft = n_fft)


test_stft_div = test_stft.T
test_holder_time = []
test_holder_freq = []
for i in range (0, int(test_stft_div.shape[0]/100)):
    each_slice = test_stft_div[i*100: (i+1)*100]
    test_holder_time.append(each_slice)
    test_holder_freq.append(each_slice.T)
#%%
predict_holder = []
for i in range(0, len(test_holder_freq)):
    mag_time = np.abs(test_holder_time[i])**2
    mag_freq = np.abs(test_holder_freq[i])**2
    time_db = lr.power_to_db(mag_time)
    freq_db = lr.power_to_db(mag_freq)
    time_norm = np.divide(np.subtract(time_db, min_c), div)
    freq_norm = np.divide(np.subtract(freq_db, min_c), div)
    predict = TF_LSTM_model.predcit([time_norm, freq_norm])
    predict_holder.append(predict)
#%%
predicted_un_normalize = np.add(np.multiply(predict_holder, div), min_c) # remove normalization
predicted_power = lr.db_to_power(predicted_un_normalize) #db to power
#%%
c = predict.reshape(25*100, 513)
c = c.T
#%%
reconstruct_time = lr.istft(c , hop_length=512, win_length=1024, window='hann' , length=1280512)
import soundfile as sf
sf.write('/home/navin/op/reconstruct.wav', reconstruct_time, 44100)
#%%
import IPython.display as ipd
ipd.Audio('/home/navin/op/reconstruct.wav')
#%%
#%%
#plot SNR vs Epochs
plt.plot(history.history['SNR'], label = 'Training')
plt.plot(history.history['val_SNR'], label = 'Validation')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('SNR')
plt.title('Epochs vs SNR @ lr=0.0001')
plt.grid()


#plot Loss vs Epochs
plt.plot(history.history['loss'], label = 'Training')
plt.plot(history.history['val_loss'], label = 'Validation')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title('Epochs vs loss @ lr=0.0001')
plt.grid()
#%%