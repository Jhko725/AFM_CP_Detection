# Training Variational Autoencoder with Multiple Lock-In Data

# Import necessary modules
import numpy as np
from keras.models import Model
from keras import metrics
from keras import backend as K
from keras.layers import Input, Dense, Activation, Dropout, BatchNormalization, Lambda, Conv1D, Layer

# Define the custom ChannelWise_FC layer
class ChannelWise_FC(Layer):
    
    def __init__(self, output_dim):
        self.output_dim = output_dim
        super(ChannelWise_FC, self).__init__()
        
    def build(self, input_shape):
        self.W = self.add_weight(name = 'weight', shape = (input_shape[2], self.output_dim), initializer = 'uniform', trainable = True)
        self.b = self.add_weight(name = 'bias', shape = (1, self.output_dim), initializer = 'uniform', trainable = True)
        super(ChannelWise_FC, self).build(input_shape)
    
    def call(self, x):
        return K.bias_add(K.dot(x, self.W), self.b)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

'''
** CreateVAE(** hyperparameters)
Inputs:
- input_dim: dimension of the input data, in the form (length of the input sequences, number of input channels)
- n_conv: number of Conv1D layers to be used
- n_fc: number of the ChannelWise_FC layers to be used
- fc_dim: output dimension of the ChannelWise_FC layer
- latent_dim: number of channels of the latent sequence
- conv_param: dictionary holding the hyperparameters for the Conv1D layer. Needs n_filters, kernel_size, and activation
- batch_normalization: True or False
- activation: activation function of the hidden layer; 'relu', 'elu', ...
- optimizer: 'adam', 'rmsprop', ...

Outputs:
- Keras model of the created VAE in the form [model, encoder, decoder]

<References>
1) https://arxiv.org/pdf/1604.07379.pdf
'''

# Wrap the model into a function which takes the hyperparameters as input
def CreateVAE(input_dim = (1000, 2), n_conv = 2, n_fc = 3, fc_dim = 16, latent_dim = 1, conv_param = {'filters' : 32, 'kernel_size' : 5, 'activation' : 'relu'}, optimizer = 'adam'):
    
    # Model parameters
    seq_length, n_channels = input_dim
    
    # Retrieve the Conv1D layer parameters
    filters = conv_param['filters']
    kernel_size = conv_param['kernel_size']
    activation = conv_param['activation']
    
    # Encoder archetecture
    encoder_input = Input(shape = (seq_length, n_channels), name = 'Encoder_Input') # (
    for i in range(n_conv):
        if i==0:
            x = Conv1D(**conv_param, padding = 'same')(encoder_input)
        else:
            x = Conv1D(**conv_param, padding = 'same')(x)

    for i in range(n_fc-1):       
        x = ChannelWise_FC(output_dim = fc_dim)(x)
    
    latent_channel = ChannelWise_FC(output_dim = latent_dim)(x)
  
    # Decoder archetecture
    decoder_input = Input(shape = (seq_length, latent_dim), name = 'Decoder_Input')
    for i in range(n_fc-1):
        if i==0:
            x = ChannelWise_FC(output_dim = fc_dim)(decoder_input)
        else:
            x = ChannelWise_FC(output_dim = fc_dim)(x)
    x = ChannelWise_FC(output_dim = filters)(x)

    for i in range(n_conv-1):
        x = Conv1D(**conv_param, padding = 'same')(x)
    decoder_output = Conv1D(filters = n_channels, kernel_size = kernel_size, activation = 'tanh', padding = 'same')(x)
   

    # encoder model statement
    encoder = Model(encoder_input, latent_channel, name = 'encoder')
    encoder.summary()
    #plot_model(encoder, to_file = 'vae_mlp_encoder.png', show_shapes = True)

    # decoder model statement
    decoder = Model(decoder_input, decoder_output, name = 'decoder')
    decoder.summary()
    #plot_model(decoder, to_file = 'vae_mlp_decoder.png', show_shapes = True)

    # VAE model statement
    decoder_output = decoder(encoder(encoder_input))
    vae = Model(encoder_input, decoder_output, name = 'vae')
    
    # Reconstruction loss
    # average over square error of along z, then add amp and phase error, then average over batches
    weights = K.variable(value = np.array([[1],[10]]))
    seq_mse_loss = K.mean(K.dot(K.mean(K.square(encoder_input - decoder_output), axis = 1), weights))
    
    vae.add_loss(seq_mse_loss)
    
    vae.compile(optimizer = optimizer)
    vae.summary()

    return {'model':vae, 'encoder':encoder, 'decoder':decoder} # To return multiple outputs, including the model, need to wrap in a dictionary