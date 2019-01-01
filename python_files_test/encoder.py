import codecs
import csv
import bisect
import sys
import numpy as np
import gc

FIRST_TIME_SERIE_DATE = 59580 # Wednesday November 17, 1858 is the zero date
PB_COUNT = 6
VAR_COUNT = 5
VAR_META_COUNT = 10
TIME_SERIES_DIM = 1095 # 3 years of history(3 * 365)
TEST_N = 3490000
TRAIN_N = 7848
TRAIN_PATH = "all/training_set.csv"
TEST_PATH = "all/test_set.csv"
TRAIN_METADATA_PATH = "all/training_set_metadata.csv"
TEST_METADATA_PATH = "all/test_set_metadata.csv"

ENCODING = 'utf-8'

def rescale_time_serie(time_serie):
    '''
    Standardize the time-serie for flux and flux error for each passband and object.
    ''' 
    '''
    scaling_time_serie = time_serie[:,0:2,:,:]
    mask = time_serie[:,4:5,:,:]
    aggregation_axis = 2
    n = mask.sum(axis=aggregation_axis, keepdims=True)
    mean = scaling_time_serie.sum(axis=aggregation_axis, keepdims=True) / n
    std = np.sum((scaling_time_serie - mean) * (scaling_time_serie - mean),\
                                  axis=aggregation_axis, keepdims=True) / n
    std = np.sqrt(std)
    
    std[std==0] = 1
    
    return ((scaling_time_serie - mean) / std, mean, std)
    '''
    
    scaling_time_serie = time_serie[:,0:2,:,:]
    mask = time_serie[:,4:5,:,:]
    aggregation_axis = 2
    max = scaling_time_serie.max(axis=aggregation_axis, keepdims=True)
    min = scaling_time_serie.max(axis=aggregation_axis, keepdims=True)
    
    diff = min - max
    diff[diff==0] = 1
    
    return ((scaling_time_serie - min) / diff, min, diff)
    
    
def parse_line_to_timeserie(time_serie,line):
    
    mjd = float(line[1])
    day = int(mjd)
    hour = mjd % 1
    
    passband = int(line[2])
    flux = float(line[3])
    flux_err = float(line[4])
    detected = int(line[5])
    
    time_serie[0,day - FIRST_TIME_SERIE_DATE,passband] = flux
    time_serie[1,day - FIRST_TIME_SERIE_DATE,passband] = flux_err
    time_serie[2,day - FIRST_TIME_SERIE_DATE,passband] = detected
    time_serie[3,day - FIRST_TIME_SERIE_DATE,passband] = hour
    time_serie[4,day - FIRST_TIME_SERIE_DATE,passband] = 1
    
    return time_serie

def create_time_serie():
    # TODO: sparse np format?
    # One extra for the mask
    time_serie = np.zeros((VAR_COUNT,TIME_SERIES_DIM,PB_COUNT))
    return time_serie

def initialize_time_serie(line):
    time_serie = create_time_serie()
    return parse_line_to_timeserie(time_serie,line)

def iter_time_serie_by_id(file, chunk_size = 10000000,max_n=sys.maxsize):
    """
    The method loops over the csv file and yields
    all the time series in a chunks with the specified
    chunk size. The function assumes that the lines of
    the csv files are grouped by object id. The chunk
    size represents the count of time series within a chunk.
    The time serie is the collection of all chronologically
    ordered observations for an object id and it consists of
    the dimensions for each feature value of interest.
    """
    reader = csv.reader(file)
    chunk = list()
    ids = list()
    headers = next(reader)
    first_line = next(reader)
    time_serie = initialize_time_serie(first_line)
    id = first_line[0]

    counter = 0
    for l in reader:
        if id == l[0]:
            id = l[0]
            time_serie = parse_line_to_timeserie(time_serie,l)
            continue

        chunk.append(time_serie)
        ids.append(id)
        counter += chunk_size
        if counter > max_n:
            break
        if len(chunk) >= chunk_size:
            yield (np.array(chunk), ids)
            chunk = list()
            ids = list()
            
        id = l[0]
        time_serie = initialize_time_serie(l)
        
    if counter < max_n:
        yield (np.array(chunk), ids)
    
def write_input(file, max_n=sys.maxsize): 
    '''
    The function assumes that the lines of
    the csv files are grouped by object id.
    '''
    flux_list = list()
    id_to_idx = dict()

    reader = csv.reader(file)
    headers = next(reader)

    for l in reader:
        object_id = l[0]
        if object_id in id_to_idx:
            flux_list[id_to_idx[object_id]] = parse_line_to_timeserie(time_serie,l)
        else:
           # TODO: we have to use some shuffeling insted of taking the first n samples
            if len(flux_list) > max_n:
                break

            id_to_idx[object_id] = len(flux_list)
            time_serie = initialize_time_serie(l)      
            flux_list.append(time_serie)
                
    return (np.array(flux_list), id_to_idx)
VAL_COUNT = int(TRAIN_N * 0.4)

'''with codecs.open(TRAIN_PATH, "r", ENCODING) as fp: 
    train_ts, train_id_to_idx = write_input(fp)

val_ts = train_ts[(N-VAL_COUNT):(N-1),]

(val_scaled, val_add_factor, val_mult_factor) = rescale_time_serie(val_ts)
(train_scaled, train_add_factor, train_mult_factor) = rescale_time_serie(train_ts)
'''
from keras.layers import Input, Dense, Dropout, Convolution2D, Reshape, Flatten,\
                         concatenate, BatchNormalization, Multiply, Add
from keras.models import Model
from keras import backend as K, objectives
from keras.optimizers import Adam
from keras import regularizers

ENCODING_DIM = 16
DROPOUT_RATE = 0.03
LR = 0.00001

FACTOR_INPUT_SHAPE = (2, 1, PB_COUNT)

# The input shape of the scaled time-series
SCALED_INPUT_SHAPE = (2, TIME_SERIES_DIM, PB_COUNT)

# The output shape of the scaled time-series
SCALED_OUTPUT_DIM = 2 * TIME_SERIES_DIM * PB_COUNT
SCALED_OUTPUT_SHAPE = (2, TIME_SERIES_DIM, PB_COUNT)

# The shape of the inputs between 0 and 1
BIN_INPUT_SHAPE = (3, TIME_SERIES_DIM, PB_COUNT)

def tailored_loss(true_output, loss_input):
    flux_loss = objectives.mean_squared_error(loss_input[:,0,:,:] * true_output[:,2,:,:], true_output[:,0,:,:])
    flux_err_loss = objectives.mean_squared_error(loss_input[:,1,:,:] * true_output[:,2,:,:], true_output[:,1,:,:])

    return 0.5 * flux_loss + 0.5 * flux_err_loss

scaled_ts_input = Input(shape = SCALED_INPUT_SHAPE)
bin_ts_input = Input(shape = BIN_INPUT_SHAPE)
add_factor = Input(shape = FACTOR_INPUT_SHAPE)
mult_factor = Input(shape = FACTOR_INPUT_SHAPE)

xf = concatenate([add_factor, mult_factor])
xf = Dense(2 * ENCODING_DIM, activation='relu')(xf)
xf = Dropout(DROPOUT_RATE)(xf)
xf = Flatten()(xf)

# Weekly pattern convolution
xs = Convolution2D(32, (7, 6), strides=(7, 6), activation='relu', data_format="channels_first")(scaled_ts_input)
xs = BatchNormalization()(xs)
xb = Convolution2D(32, (7, 6), strides=(7, 6), activation='relu', data_format="channels_first")(bin_ts_input)
print("Output dimensions of the weekly pattern convolution", xs.shape)

# Monthly pattern convolution
xs = Convolution2D(16, (4, 1), strides=(4, 1), activation='relu', data_format="channels_first")(xs)
xs = BatchNormalization()(xs)
xb = Convolution2D(16, (4, 1), strides=(4, 1), activation='relu', data_format="channels_first")(xb)
print("Output dimensions of the monthly pattern convolution", xs.shape)

# Annual pattern convolution
xs = Convolution2D(16, (12, 1), strides=(12, 1), activation='relu', data_format="channels_first")(xs)
xs = BatchNormalization()(xs)
xb = Convolution2D(16, (12, 1), strides=(12, 1), activation='relu', data_format="channels_first")(xb)
print("Output dimensions of the annual pattern convolution", xs.shape)

x = concatenate([xs, xb])
x = Dense(4 * ENCODING_DIM, activation='relu')(x)
x = Dropout(DROPOUT_RATE)(x)
x = Flatten()(x)
x = concatenate([x, xf])

print("Input dimensions of the embedding layer.", x.shape)
encoded = Dense(4 * ENCODING_DIM, activation='relu')(x)
encoded = Dropout(DROPOUT_RATE)(encoded)
encoded = Dense(2 * ENCODING_DIM, activation='relu')(encoded)
encoded = Dropout(DROPOUT_RATE)(encoded)
encoded = Dense(ENCODING_DIM, activation='relu')(encoded)
print("Output dimensions of the embedding layer.", encoded.shape)

x_mid = Dropout(DROPOUT_RATE)(encoded)
x_mid = Dense(2 * ENCODING_DIM, activation='relu')(x_mid)
x_mid = Dropout(DROPOUT_RATE)(x_mid)
x_mid = Dense(4 * ENCODING_DIM, activation='relu')(x_mid)
x_mid = Dropout(DROPOUT_RATE)(x_mid)

decoded = Dense(SCALED_OUTPUT_DIM, activation='sigmoid')(x_mid)
decoded = Reshape(SCALED_OUTPUT_SHAPE, input_shape=(SCALED_OUTPUT_DIM,))(decoded)

decoded = Multiply()([decoded, mult_factor]) # Scaling up in the output layer
decoded = Add()([decoded, add_factor]) # Scaling up in the output layer

print("Decoded input dimensions.", decoded.shape)

autoencoder = Model([scaled_ts_input, bin_ts_input, add_factor, mult_factor], decoded)

optimizer = Adam(lr = LR)
autoencoder.compile(optimizer = optimizer, loss = tailored_loss)

encoder = Model([scaled_ts_input, bin_ts_input, add_factor, mult_factor], encoded)

from keras.models import model_from_json

json_file = open('encoder_bs-2048_loss-mse_el-16.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("encoder_bs-2048_loss-mse_el-16.h5")
print("Loaded model from disk")
#embedded_features = encoder.predict([val_scaled, val_ts[:,2:5,], val_add_factor, val_mult_factor])

CHUNK_SIZE = 2048
STEPS_PER_EPOCH = 2 * 853
import pandas as pd
from tqdm import tqdm
def time_series_generator(train_path, test_path, chunk_size,\
                          steps_per_epoch, train_n=sys.maxsize, test_n=sys.maxsize):
    with codecs.open(train_path, "r", ENCODING) as fp:
            time_series_redaer =iter_time_serie_by_id(fp, chunk_size)
            for (time_series, ids) in tqdm(time_series_redaer):
                (scaled_ts, add_factor, mult_factor) = rescale_time_serie(time_series)
                embedded = (loaded_model.predict([scaled_ts, time_series[:,2:5,], add_factor, mult_factor]))
                embedded = (pd.DataFrame(embedded))
                with open("train/neg_flux/encoder_train.csv", 'a') as f:
                    embedded.to_csv(f)

                
time_series_generator(TRAIN_PATH, TEST_PATH, CHUNK_SIZE, STEPS_PER_EPOCH, TRAIN_N, TEST_N)