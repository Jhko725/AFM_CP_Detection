import matplotlib.pyplot as plt
import numpy as np

# the data must be in the format PreProcessing.DataPrep returns
def PlotApproachCurve(data, figsize = (11, 3), fontsize = 11, wspace = 0.3, **kwargs):
    fig = plt.figure(figsize = figsize)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig.subplots_adjust(wspace = wspace)

    for i in range(data.shape[0]):
        ax1.plot(data[i,:,0], data[i,:,1], 'k', **kwargs)
        ax2.plot(data[i,:,0], data[i,:,2], 'k', **kwargs)
    
    ax1.set_xlabel('Distance (nm)', fontsize = fontsize)
    ax1.set_ylabel('Amplitude (normalized)', fontsize = fontsize)
    ax1.grid(ls = '--')

    ax2.set_xlabel('Distance (nm)', fontsize = fontsize)
    ax2.set_ylabel('Phase (rad)', fontsize = fontsize)
    ax2.grid(ls = '--')
    
    return fig

def PlotHistory(history, figsize = (14, 3), wspace = 0.3, fontsize = 11, **kwargs):
    fig, axes = plt.subplots(1, 3, figsize = figsize)
    fig.subplots_adjust(wspace = wspace)

    axes[0].plot(history['loss'], **kwargs)
    axes[0].plot(history['val_loss'], **kwargs)
    axes[0].legend(['train', 'test'], loc='upper right', fontsize = fontsize - 1)
    axes[0].set_ylabel('Loss', fontsize = fontsize)
    axes[0].set_title('Training and Validation Losses per Epoch', fontsize = fontsize)
    
    axes[1].plot(history['amp_mse'], **kwargs)
    axes[1].plot(history['phas_mse'], **kwargs)
    axes[1].legend(['amplitude mse', 'phase mse'], loc='best', fontsize = fontsize - 1)
    axes[1].set_ylabel('Mean Squared Error', fontsize = fontsize)
    axes[1].set_title('Amplitude and Phase MSE for the Training Set', fontsize = fontsize)
    
    axes[2].plot(history['val_amp_mse'], **kwargs)
    axes[2].plot(history['val_phas_mse'], **kwargs)
    axes[2].legend(['amplitude mse', 'phase mse'], loc='best', fontsize = fontsize - 1)
    axes[2].set_ylabel('Mean Squared Error', fontsize = fontsize)
    axes[2].set_title('Amplitude and Phase MSE for the Test Set', fontsize = fontsize)
    
    for ax in axes:
        ax.set_xlabel('Epoch', fontsize = fontsize)
        ax.set_yscale('log')
        ax.grid(ls = '--')
        
    return fig

def PlotReconstruction(plot_index, model, data, figsize = (24, 5), wspace = 0.3, size = 4, width = 1.5, fontsize = 14, **kwargs):
    autoencoder = model['autoencoder']
    encoder = model['encoder']

    seq = np.expand_dims(data[plot_index, :, :], axis = 0)
    reconst = autoencoder.predict([seq[:,:,1:3], seq[:,:,3:]])
    latent = encoder.predict([seq[:,:,1:3], seq[:,:,3:]])

    fig = plt.figure(figsize = figsize)
    ax_l1 = fig.add_subplot(131)
    ax_m1 = fig.add_subplot(132)
    ax_r1 = fig.add_subplot(133)
    fig.subplots_adjust(wspace = wspace)

    ax_l2 = ax_l1.twinx()
    ax_m2 = ax_m1.twinx()
    ax_r2 = ax_r1.twinx()

    # Plot the data
    line_l1 = ax_l1.scatter(seq[0,:,0], seq[0,:,1], color = 'black', label = 'Experiment', **kwargs)
    line_l2 = ax_l2.scatter(seq[0,:,0], reconst[0,:,0], color = 'tab:red', label = 'Reconstructed', **kwargs)

    line_m1 = ax_m1.scatter(seq[0,:,0], seq[0,:,2], color = 'black', label = 'Experiment', **kwargs)
    line_m2 = ax_m2.scatter(seq[0,:,0], reconst[0,:,1], color = 'tab:red', label = 'Reconstructed', **kwargs)

    line_r1 = ax_r1.scatter(seq[0,:,0], seq[0,:,1], color = 'black', label = 'Amplitude', **kwargs)
    line_r2 = ax_r2.scatter(seq[0,:,0], latent[0,:,0], color = 'tab:red', label = 'Latent Variable', **kwargs)

    ax_l1.set_xlabel('Distance z (nm)', fontsize = fontsize)
    ax_l1.set_ylabel('Experimental Amplitude', fontsize = fontsize)
    ax_l2.set_ylabel('Reconstructed Amplitude', color = 'tab:red', fontsize = fontsize)

    ax_m1.set_xlabel('Distance z (nm)', fontsize = fontsize)
    ax_m1.set_ylabel('Experimental Phase', fontsize = fontsize)
    ax_m2.set_ylabel('Reconstructed Phase', color = 'tab:red', fontsize = fontsize)

    ax_r1.set_xlabel('Distance z (nm)', fontsize = fontsize)
    ax_r1.set_ylabel('Amplitude (normalized)', fontsize = fontsize)
    ax_r2.set_ylabel('Latent Variable', color = 'tab:red', fontsize = fontsize)

    # Set axis color and size
    ax_l2.yaxis.label.set_color('tab:red')
    ax_m2.yaxis.label.set_color('tab:red')
    ax_r2.yaxis.label.set_color('tab:red')

    tkw = dict(size = size, width = width)

    ax_l1.tick_params(axis='x', **tkw)
    ax_l1.tick_params(axis='y', **tkw)
    ax_l2.tick_params(axis='y', colors='tab:red', **tkw)

    ax_m1.tick_params(axis='x', **tkw)
    ax_m1.tick_params(axis='y', **tkw)
    ax_m2.tick_params(axis='y', colors='tab:red', **tkw)

    ax_r1.tick_params(axis='x', **tkw)
    ax_r1.tick_params(axis='y', **tkw)
    ax_r2.tick_params(axis='y', colors='tab:red', **tkw)

    lines_l = [line_l1, line_l2]
    labels_l = [l.get_label() for l in lines_l]
    ax_l1.legend(lines_l, labels_l, loc = 'upper right')

    lines_m = [line_m1, line_m2]
    labels_m = [l.get_label() for l in lines_m]
    ax_m1.legend(lines_m, labels_m, loc = 'upper right')

    lines_r = [line_r1, line_r2]
    labels_r = [l.get_label() for l in lines_r]
    ax_r1.legend(lines_r, labels_r, loc = 'upper right')

    ax_l1.grid(ls = '--')
    ax_m1.grid(ls = '--')
    ax_r1.grid(ls = '--')

    return fig

def PlotLatent(encoder, data, figsize = (7, 5), kwargs = {'c':'k', 'marker':'.', 'alpha':0.1}):
    latent = encoder.predict([data[:,:,1:3], data[:,:,3:]])
    
    fig, ax = plt.subplots(1,1, figsize = figsize)
    for i in range(data.shape[0]):
        ax.scatter(data[i,:,0], latent[i,:,0], **kwargs)

    ax.grid(ls = '--')
    ax.set_xlabel('z (a.u.)')
    ax.set_ylabel('latent parameter (a.u.)')
    
    return fig


    # Create latent variable visualization function as well as reconstruction graph visualization function