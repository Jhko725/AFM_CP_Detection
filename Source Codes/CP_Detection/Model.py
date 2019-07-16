from keras.models import Model
from keras import metrics
from keras import backend as K
from keras.layers import Input, Activation, Dropout, BatchNormalization, Conv1D, Concatenate
from keras import optimizers

import numpy as np
import re

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

# Note that batch normalization needs to be applied after the linear operation, but before the nonlinear activation.
# Dropout is applied after the activation

# Since Conv1D -> Batch Normalizaton -> Activation is used a lot, we wrap this into a single function
def Conv1dBnActivation(filters = 32, kernel_size = 5, activation = 'relu', batch_normalization = True, dropout_rate = 0.0, layer_name = None):
    def inner_function(x):
        x = Conv1D(filters = filters, kernel_size = kernel_size, padding = 'same')(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        
        # layer_name argument is only applied to the last layer
        if layer_name == None:
            x = Activation(activation = activation)(x)
            if dropout_rate != 0.0:
                x = Dropout(rate = dropout_rate)(x)
        else:
            if dropout_rate == 0.0:
                x = Activation(activation = activation, name = layer_name)(x)
            else:
                x = Activation(activation = activation)(x)
                x = Dropout(rate = dropout_rate, name = layer_name)(x)
        return x

    return inner_function

# We also define the wrapping function for Pointwise Convolution -> Batch Normalization -> Activation
def PwConvBnActivation(output_dim = 32, activation = 'relu', batch_normalization = True, dropout_rate = 0.0, layer_name = None):
    def inner_function(x):
        x = Conv1D(filters = output_dim, kernel_size = 1, padding = 'same')(x)
        if batch_normalization:
            x = BatchNormalization()(x)

        # layer_name argument is only applied to the last layer
        if layer_name == None:
            x = Activation(activation = activation)(x)
            if dropout_rate != 0.0:
                x = Dropout(rate = dropout_rate)(x)
        else:
            if dropout_rate == 0.0:
                x = Activation(activation = activation, name = layer_name)(x)
            else:
                x = Activation(activation = activation)(x)
                x = Dropout(rate = dropout_rate, name = layer_name)(x)
        return x

    return inner_function

# Defines the function that returns the AbstractionBlock
def String2Block(format_string, conv1d_params, pwconv_params, layer_name = None):
    # Use the re module to check the string is in the form (C or P):(number of channels)|...|(C or P):(number of channels)
    pattern = re.compile('[P,C]:[1-9]\d*\|{1,}[P,C]:[1-9]\d*')
    match = re.search(pattern, format_string)
    if not match:
        raise Exception('The input format must be of the form (C or P):(number of channels)|...|(C or P):(number of channels)')

    layer_param = format_string.split('|')

    def inner_function(x):
        for i in range(len(layer_param)):
            layer_type, channels = layer_param[i].split(':')
            channels = int(channels)

            if layer_type == 'C':
                # The name argument is only applied to the last layer
                if i != len(layer_param) - 1:
                    x = Conv1dBnActivation(filters = channels, **conv1d_params, layer_name = None)(x)
                else:
                    x = Conv1dBnActivation(filters = channels, **conv1d_params, layer_name = layer_name)(x)
                
            else:
                if i != len(layer_param) - 1:
                    x = PwConvBnActivation(output_dim = channels, **pwconv_params, layer_name = None)(x)
                else:
                    x = PwConvBnActivation(output_dim = channels, **pwconv_params, layer_name = layer_name)(x)
        return x

    return inner_function

# Custom loss function for the model
def amp_mse(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis = [0,1])[0]

def phas_mse(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis = [0,1])[1]

def weighted_seq_mse(y_true, y_pred):
    return amp_mse(y_true, y_pred) + 100*phas_mse(y_true, y_pred)

# Wrap the model into a function which takes the hyperparameters as input
def Conv1dAE(input_channels = (2, 2), latent_channels = 1, seq_length = 1000, abstraction_block = 'C:32|C:32|C:32', composition_block = 'P:32|P:32|P:32', kernel_size = 5, activation =  'relu', batch_norm = True, dropout_rate = 0.0, optim_type = 'adam', lr = 1e-3, batch_size = 32):
    
    # Model parameters
    seq_channels, const_channels = input_channels # seq_channels - number of sequence inputs : Amp, Phas; const_channels - number of constant inputs : Q/w0, w/w0
    
    conv1d_params = {'kernel_size': kernel_size, 'activation': activation, 'batch_normalization': batch_norm, 'dropout_rate': dropout_rate}
    pwconv_params = {'activation': activation, 'batch_normalization': batch_norm, 'dropout_rate': dropout_rate}

    # Create the optimizer
    if optim_type == 'adam':
        optimizer = optimizers.Adam(lr = lr)
    elif optim_type == 'rmsprop':
        optimizer = optimizers.rmsprop(lr = lr)
    else:
        raise Exception('Select the proper optimizer type: adam or rmsprop')

    # Create the AbstractionBlock and the CompositionBlock
    AbstractionBlock = String2Block(abstraction_block, conv1d_params, pwconv_params)
    CompositionBlock = String2Block(composition_block, conv1d_params, pwconv_params)

    ## Encoder Architecture ##
    seq_input = Input(shape = (seq_length, seq_channels), name = 'Sequence_Input')
    
    # First, use the AbstractionBlock to extract high level feature from the sequence inputs
    processed_seq = AbstractionBlock(seq_input)
    
    const_input = Input(shape = (seq_length, const_channels), name = 'Constant_Input') 
    encoder_combined = Concatenate(name = 'encoder_input')([processed_seq, const_input])
   
    x = CompositionBlock(encoder_combined)
   
    latent_output = PwConvBnActivation(output_dim = 1, activation = 'linear', batch_normalization = batch_norm, dropout_rate = 0.0, layer_name = 'Encoder_Output')(x)
    
    ## Decoder Architecture ##
    decoder_input = Input(shape = (seq_length, latent_channels), name = 'Latent_Channel_Input')

    processed_latent = AbstractionBlock(decoder_input)
 
    decoder_combined = Concatenate(name = 'decoder_input')([processed_latent, const_input])
    
    x = CompositionBlock(decoder_combined)
        
    decoder_output = PwConvBnActivation(output_dim = 2, activation = 'tanh', batch_normalization = batch_norm, dropout_rate = 0.0, layer_name = 'Decoder_Output')(x)

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
  
    autoencoder.compile(optimizer = optimizer, loss = weighted_seq_mse, metrics = [amp_mse, phas_mse])
    autoencoder.summary()

    # Create a dictionary of hyperparameters
    hyperparameters = {'abstraction_block': abstraction_block, 'composition_block': composition_block, 'kernel_size': kernel_size, 'activation': activation, 'batch_norm': batch_norm, 
    'dropout_rate': dropout_rate, 'optim_type': optim_type, 'lr': lr, 'batch_size': batch_size, 'seq_length': seq_length, 'input_channels': input_channels, 'latent_channels': latent_channels}
    
    return {'autoencoder':autoencoder, 'encoder':encoder, 'decoder':decoder}, hyperparameters # To return multiple outputs, including the model, need to wrap in a dictionary