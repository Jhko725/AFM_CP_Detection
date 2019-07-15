import matplotlib.pyplot as plt

# the data must be in the format PreProcessing.DataPrep returns
def PlotApproachCurve(data, fontsize = 11, wspae = 0.3, **kwargs):
    fig = plt.figure(figsize = (11, 3))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    fig.subplots_adjust(wspace = 0.3)

    for i in range(data.shape[0]):
        ax1.plot(data[i,:,0], data[i,:,1], 'k', **kwargs)
        ax2.plot(data[i,:,0], data[i,:,2], 'k', **kwargs)
    fontsize = 11
    
    ax1.set_xlabel('Distance (nm)', fontsize = fontsize)
    ax1.set_ylabel('Amplitude (normalized)', fontsize = fontsize)
    ax1.grid(ls = '--')

    ax2.set_xlabel('Distance (nm)', fontsize = fontsize)
    ax2.set_ylabel('Phase (rad)', fontsize = fontsize)
    ax2.grid(ls = '--')
    
    return fig

def PlotHistory(history, **kwargs):
    fig, ax = plt.subplots(1,1)
    
    ax.plot(history['loss'], **kwargs)
    ax.plot(history['val_loss'], **kwargs)
    ax.legend(['train', 'test'], loc='upper right')
    
    ax.set_yscale('log')
    ax.grid(ls = '--')
    
    ax.set_title('Learning Curve of the Model (1D Convolution AE)')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    
    return fig