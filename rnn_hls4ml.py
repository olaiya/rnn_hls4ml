import os
import sys
import operator
import time
import numpy as np
import pandas as pd
import uproot
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import argparse
import hls4ml
import plotting

print('Running using python version: {}.{}.{}'.format(sys.version_info[0],sys.version_info[1],sys.version_info[2]))
print('Running using tensorflow version: {}'.format(tf.__version__))
print('Running using hls4ml version {}\n'.format(hls4ml.__version__))

#Function to generate batch_size time series with number of steps n_steps
def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)   # + noise
    return series[..., np.newaxis].astype(np.float32)

#Function to plot time series
def plot_series(series, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$"):
    plt.plot(series, ".-")
    if y is not None:
        plt.plot(n_steps, y, "bx", markersize=10)
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "ro")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, n_steps + 1, -1, 1])


#Generate data
np.random.seed(42)
n_steps = 50
series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

model = tf.keras.models.Sequential([
    tf.keras.layers.GRU(20, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.GRU(20),
    tf.keras.layers.Dense(1, name="output_dense")
])

model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))

#Predict time series
y_pred = model.predict(X_valid)
plot_series(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])

#Start the process of converting thr RNN to VHDL using hls4ml
board_model = 'xcvu9p-flga2577-2-e'


#make firmware output directory
firmWareOutPutDir = 'hls4ml_firmware'
CHECK_FOLDER = os.path.isdir(firmWareOutPutDir)

# If folder doesn't exist, then create it.
if not CHECK_FOLDER:
    os.makedirs(firmWareOutPutDir)
    print("created folder : ", firmWareOutPutDir)

hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(layers=['Activation'])
hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(rounding_mode='AP_RND')
hls4ml.model.optimizer.get_optimizer('output_rounding_saturation_mode').configure(saturation_mode='AP_SAT')

hls_config_reg = hls4ml.utils.config_from_keras_model(model, granularity='name')

parallelFactor = 12
reuseFactor = 1

for layer in hls_config_reg['LayerName'].keys():
    hls_config_reg['LayerName'][layer]['Trace'] = True
    #Set parallelisation
    hls_config_reg['LayerName'][layer]['ParallelizationFactor'] = parallelFactor

hls_config_reg['Model']['Precision'] = 'ap_fixed<48,24>'
hls_config_reg['Model']['ReuseFactor'] = reuseFactor

#If you want best numerical performance for high-accuray models, while the default latency strategy is faster but numerically more unstable
hls_config_reg['LayerName']['output_dense']['Strategy'] = 'Stable'

print('Plotting Dict Params')
plotting.print_dict(hls_config_reg)

cfg_reg = hls4ml.converters.create_config(backend='Vivado')
cfg_reg['IOType']     = 'io_stream' # Must set this if using CNNs!
cfg_reg['HLSConfig']  = hls_config_reg
cfg_reg['KerasModel'] = model
cfg_reg['OutputDir']  = firmWareOutPutDir
cfg_reg['XilinxPart'] = board_model

hls_model_reg = hls4ml.converters.keras_to_hls(cfg_reg)

if testHls4ml_q:
    print('Compiling hls4ml model')
    hls_model_reg.compile()

    y_predict_reg        = qmodel.predict(x_test)
    y_predict_reg_hls4ml = hls_model_reg.predict(np.ascontiguousarray(x_test))

    print(y_predict_reg[0:20])
    print(y_predict_reg_hls4ml[0:20])

    pred_hls4ml_diff = np.abs(y_predict_reg - y_predict_reg_hls4ml)
    print('Predicted difference')
    print(pred_hls4ml_diff[0:20])

    np.savez('./accuracy_evaluation/hls4ml_error_'+label+'_trained_'+decay+'_'+tag+'_nFeature'+str(nFeatures)+'_nEpochs'+str(numEpochs)+'_nBatchSize'+str(batchSize)+'.npz',
        pred_diff=pred_hls4ml_diff)

if synth:
    print('STARTING VIVADO SYTHNESIS')
    hls_model_reg.build(csim=True, synth=True, vsynth=False, cosim=False)
    #hls_model_q.build(csim=False, synth=False, vsynth=True)