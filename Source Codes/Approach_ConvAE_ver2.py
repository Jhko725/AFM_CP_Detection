from keras.models import Model, model_from_json
from keras import metrics
from keras import backend as K

from keras.layers import Input, Activation, Dropout, BatchNormalization, Conv1D, Concatenate

import numpy as np




'''
** Dataprep(filepath, test_ratio) **
Inputs:
- filepath: path to the pre-processed AFM data file, in .npz format
- test_ratio: the ratio between the total number of approach curve data and the number of the approach curve data used in the validation set
- seed: seed for the random generator used in shuffling training and test sets

Outputs:
- train_dataset: dataset corresponding to the training set
- test_dataset: dataset corresponding to the test set
- train: training set
- test: test set
'''

def DataPrep(filepath, test_ratio = 0.1, seed = None): # the filepath must point to pre-processed AFM datafile, in .npz format
    # load the npz file
    dataset = np.load(filepath, allow_pickle = True)

    mech_dataset = dataset['mech'] # Mechanical Amp, Phas,[z, Amp(V), Phas(rad)] format
    imin_dataset = dataset['imin'] # array index of the turning point : Appoach is from 0 ~ imin, Retract is from imin ~ end

    # Resonance curve fitting parameters
    Q_dataset = dataset['Q']
    w_dataset = dataset['w']
    w0_dataset = dataset['w0']
    
    N = Q_dataset.size
    N_test = int(N*test_ratio)

    print('Number of test samples : %d' %(N_test))
    print('Number of training samples : %d' %(N - N_test))

    sample_index = np.arange(N)
    np.random.seed(seed)
    np.random.shuffle(sample_index) # shuffle the input data
    
    seq_length = np.amin(imin_dataset) + 1
    
    #create data array : (trials, seq_length, 5)
    data = np.zeros((N, seq_length, 5))
    for i in range(N):
        data[i,:,0:3] = mech_dataset[i][imin_dataset[i]+1-seq_length:imin_dataset[i]+1, 0:3]
        data[i,:,3] = np.ones((1, seq_length))*Q_dataset[i]/w0_dataset[i] # 3rd channel is Q/w0
        data[i,:,4] = np.ones((1, seq_length))*w_dataset[i]/w0_dataset[i] # 4th channel is w/w0
    
    # create test and train datasets
    test_index = sample_index[0:N_test]
    train_index = sample_index[N_test:]
    
    test = data[test_index,:,:]
    train = data[train_index,:,:]
    
    keys = np.array(list(dataset.keys()))
    values = np.array(list(dataset.values()))
    print(values.shape)
    
    train_dataset = dict(zip(keys, values[:, train_index]))
    test_dataset = dict(zip(keys, values[:, test_index]))
    
    return [train_dataset, test_dataset, train, test]


'''
** Conv1dAE(** hyperparameters)
Inputs:
- input_channels
- latent_channels
- seq_length
- filters 
- output_dim
- kernel_size
- activation
- batch_norm
- dropout_rate
- optimizer

Outputs:
- Keras model of the created AE in the form [model, encoder, decoder]
- Dictionary containing the hyperparameters (those used in the function call)

<References>
1) 
'''

# Since Conv1D -> Batch Normalizaton -> Activation is used a lot, we wrap this into a single function
def Conv1dBnActivation(filters = 32, kernel_size = 5, activation = 'relu', batch_normalization = True, dropout_rate = 0.0):
    def inner_function(x):
        x = Conv1D(filters = filters, kernel_size = kernel_size, padding = 'same')(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        x = Activation(activation = activation)(x)
        if dropout_rate != 0.0:
            x = Dropout(rate = dropout_rate)(x)
        return x
    return inner_function

# We also define the wrapping function for Pointwise Convolution -> Batch Normalization -> Activation
def PwConvBnActivation(output_dim = 32, activation = 'relu', batch_normalization = True, dropout_rate = 0.0):
    def inner_function(x):
        x = Conv1D(filters = output_dim, kernel_size = 1, padding = 'same')(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        x = Activation(activation = activation)(x)
        if dropout_rate != 0.0:
            x = Dropout(rate = dropout_rate)(x)
        return x
    return inner_function

# Custom loss function for the model
def weighted_seq_mse(y_true, y_pred):
        weights = K.variable(value = np.array([[1],[10]]))
        seq_mse_loss = K.mean(K.dot(K.mean(K.square(y_true - y_pred), axis = 1), weights))
        return seq_mse_loss

# Wrap the model into a function which takes the hyperparameters as input
def Conv1dAE(input_channels = (2, 2), latent_channels = 1, seq_length = 1000, filters = 32, output_dim = 32, kernel_size = 5, activation =  'relu', batch_norm = True, dropout_rate = 0.0, optimizer = 'adam'):
    
    # Model parameters
    seq_channels, const_channels = input_channels # seq_channels - number of sequence inputs : Amp, Phas; const_channels - number of constant inputs : Q/w0, w/w0
    
    conv1d_params = {'filters': filters, 'kernel_size': kernel_size, 'activation': activation, 'batch_normalization': batch_norm, 'dropout_rate': dropout_rate}
    pwconv_params = {'output_dim': output_dim, 'activation': activation, 'batch_normalization': batch_norm, 'dropout_rate': dropout_rate}

    # Note that batch normalization needs to be applied after the linear operation, but before the nonlinear activation.
    # Dropout is applied after the activation

    ## Encoder Architecture ##
    seq_input = Input(shape = (seq_length, seq_channels), name = 'Sequence_Input')
    # First, use convolutional layer to extract high level feature from the sequence inputs
    processed_seq = Conv1dBnActivation(**conv1d_params)(seq_input)
    processed_seq = Conv1dBnActivation(**conv1d_params)(processed_seq)
    
    const_input = Input(shape = (seq_length, const_channels), name = 'Constant_Input') 
    encoder_combined = Concatenate(name = 'encoder_input')([processed_seq, const_input])
   
    x = PwConvBnActivation(**pwconv_params)(encoder_combined)
    x = PwConvBnActivation(**pwconv_params)(x)
    
    x = Conv1dBnActivation(**conv1d_params)(x)
    x = Conv1dBnActivation(**conv1d_params)(x)

    x = PwConvBnActivation(**pwconv_params)(x)
    x = PwConvBnActivation(**pwconv_params)(x)
    
    x = Conv1dBnActivation(**conv1d_params)(x)
    x = Conv1dBnActivation(**conv1d_params)(x)
    
    latent_output = PwConvBnActivation(output_dim = 1, activation = 'linear', batch_normalization = batch_norm, dropout_rate = 0.0)(x)
    
    ## Decoder Architecture ##
    decoder_input = Input(shape = (seq_length, latent_channels), name = 'Latent_Channel_Input')
    processed_latent = Conv1dBnActivation(**conv1d_params)(decoder_input)
    processed_latent = Conv1dBnActivation(**conv1d_params)(processed_latent)

    decoder_combined = Concatenate(name = 'decoder_input')([processed_latent, const_input])
    
    x = PwConvBnActivation(**pwconv_params)(decoder_combined)
    x = PwConvBnActivation(**pwconv_params)(x)

    x = Conv1dBnActivation(**conv1d_params)(x)
    x = Conv1dBnActivation(**conv1d_params)(x)

    x = PwConvBnActivation(**pwconv_params)(x)
    x = PwConvBnActivation(**pwconv_params)(x)
    
    x = Conv1dBnActivation(**conv1d_params)(x)
    x = Conv1dBnActivation(**conv1d_params)(x)
    
    decoder_output = PwConvBnActivation(output_dim = 2, activation = 'tanh', batch_normalization = batch_norm, dropout_rate = 0.0)(x)

    # encoder model statement
    encoder = Model([seq_input, const_input], latent_output, name = 'Encoder')
    encoder.summary()
    #plot_model(encoder, to_file = 'vae_mlp_encoder.png', show_shapes = True)

    # decoder model statement
    decoder = Model([decoder_input, const_input], decoder_output, name = 'Decoder')
    decoder.summary()
    #plot_model(decoder, to_file = 'vae_mlp_decoder.png', show_shapes = True)

    # VAE model statement
    decoder_output = decoder([encoder([seq_input, const_input]), const_input])
    autoencoder = Model([seq_input, const_input], decoder_output, name = '1D_All_Convolutional_AE')
    
    # Reconstruction loss
    # average over square error of along z, then add amp and phase error, then average over batches
  
    autoencoder.compile(optimizer = optimizer, loss = weighted_seq_mse)
    autoencoder.summary()

    return {'autoencoder':autoencoder, 'encoder':encoder, 'decoder':decoder} # To return multiple outputs, including the model, need to wrap in a dictionary

