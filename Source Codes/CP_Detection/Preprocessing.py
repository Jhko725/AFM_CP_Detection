import numpy as np
import pandas as pd
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numba as nb
from numba import jit, vectorize, float64

## Top level private functions for raw data import ##
    
# Resonance curve function
@vectorize("float64(float64, float64, float64, float64, float64)", nopython = True)
def _ResCurve(w, w0, Q, I_0, C0_C):
    W = w/w0
    return (I_0*W/Q)*np.sqrt(np.divide(1+2*C0_C*(1-W**2)+(C0_C*(1-W**2))**2+(C0_C*W/Q)**2, (1-W**2)**2+(W/Q)**2))

# Calibrate electrical phase by converting into radian, then adding the phase offset
@vectorize("float64(float64, float64, float64, float64, float64, float64)", nopython = True)
def _CalibratePhase(Pe, Pe_far, w, w0, Q, C0_C):
    Pe = (Pe - Pe_far)*np.pi/10
    W = w/w0
    Pe_0 = np.arctan2(1-W**2+C0_C*(1-W**2)**2+C0_C*(W/Q)**2, W/Q)
    
    return Pe+Pe_0
    
# Functions to compute physical quantities
@vectorize("float64(float64, float64, float64, float64, float64, float64, float64)", nopython = True)
def _MechAmp(Ae, Pe, w, w0, Q, C0_C, I0): # Takes numpy array Ae and Pe, then returns the mechanical amplitude Am
    Am = np.sqrt(Ae**2-2*w*C0_C*I0*Ae*np.sin(Pe)/(Q*w0)+(w*C0_C*I0/(Q*w0))**2)
    return Am

@vectorize("float64(float64, float64, float64, float64, float64, float64, float64)", nopython = True)
def _MechPhas(Ae, Pe, w, w0, Q, C0_C, I0): # Takes numpy array Ae and Pe, then returns the mechanical phase Pm
    Pm = np.arctan2(Ae*np.sin(Pe)-w*C0_C*I0/(w0*Q), Ae*np.cos(Pe))
    return Pm

@vectorize("float64(float64, float64, float64, float64, float64)", nopython = True)
def _kint(Am, Pm, w, w0, Q): # Takes numpy array Ae and Pe, then returns k_int normalized by k
    kint = np.sin(Pm)/(Am*Q) + (w/w0)**2 - 1
    return kint

@vectorize("float64(float64, float64, float64, float64, float64)", nopython = True)
def _bint(Am, Pm, w, w0, Q): # Takes numpy array Ae and Pe, then returns b_int normalized by k
    bint = (np.cos(Pm)/(Am*w) - (1/w0))/Q
    return bint

@vectorize("float64(float64, float64, float64, float64, float64)", nopython = True)
def _Edis(Am, Pm, w, w0, Q): # Takes numpy array Ae and Pe, then returns E_dis normalized by kA0^2
    Edis = (np.pi/Q)*(Am*np.cos(Pm)-(w/w0)*Am**2)
    return Edis

'''
** ImportResCurve **

    Inputs: filepath 
        - filepath: filepath of the resonance curve file
    
    Outputs: res_data, fit_param, w
        - res_data: resonance curve data in pandas dataframe format
        - fit_param: resonance curve fitting parameters, [w0, Q, I0, C0/C]
        - w: driving frequency (assumes that the driving frequency is the frequency at which the resonance curve takes its maximum)     
'''

def ImportResCurve(filepath, visualize = False, fit_param_init = [5000, 3, 1/800]):
    
    # Import resonance curve data using pandas
    # The parameter skiprows ensures that the comments on the top 
    res_data = pd.read_csv(filepath, header = None, delimiter = '\t', skiprows = 9)
    
    # Res curve is in the format [freq, amp, freq, phas, aux], so drop the redundant 3rd column
    res_data = res_data.drop(res_data.columns[4], axis = 1)
    res_data = res_data.drop(res_data.columns[2], axis = 1)
    
    # Set appropriate column name
    res_data.rename(columns = {0:'Frequency(Hz)', 1:'Amplitude(V)', 3:'Phase(V)'}, inplace = True)
    
    # Numpy array of the resonance curve data in [freq, amp, phas] format
    res_array = res_data.values
    
    # Frequency of  the maximum amplitude measured. This is equal the driving frequency used
    w0_init = res_array[res_array[:,1].argmax(),0] 
    w = w0_init
    
    # Fit the resonance curve
    Q_init, I0_init, C0_C_init = fit_param_init
    fit_param, _ = curve_fit(_ResCurve, res_array[:,0], res_array[:,1], p0 = [w0_init, Q_init, I0_init, C0_C_init], bounds = (0, np.inf))
    
    fig = None

    if visualize:
        # For the visualize keyword, create a figure of the experimental and fitted resonance curve
        fontsize = 13
        labelpad = 10
        tkw = dict(size = 6, width = 1.5, labelsize = fontsize)
        
        fig, ax_left = plt.subplots(1, 1, figsize = (7,5))
        ax_left.plot(res_array[:,0], res_array[:,1], '.-k', label = 'Amplitude(V)')
        ax_left.plot(res_array[:,0], _ResCurve(res_array[:,0], *fit_param), '-r', alpha = 0.8, label = 'Fitted Curve')
        ax_left.set_xlabel('Frequency (Hz)', fontsize = fontsize, labelpad = labelpad)
        ax_left.set_ylabel('Amplitude (V)', fontsize = fontsize, labelpad = labelpad)
        
        ax_right = ax_left.twinx()
        ax_right.plot(res_array[:,0], res_array[:,2], '.-b', alpha = 0.2, label = 'Phase(V)')
        ax_right.set_ylabel('Phase(V)', fontsize = fontsize, labelpad = labelpad)
        ax_left.grid(ls = '--')
        
        h_right, l_right = ax_right.get_legend_handles_labels()
        h_left, l_left = ax_left.get_legend_handles_labels()
        
        ax_left.tick_params(axis='x', **tkw)
        ax_left.tick_params(axis='y', **tkw)
        ax_right.tick_params(axis='y', colors='blue', **tkw)
        ax_right.yaxis.label.set_color('blue')
            
        ax_right.legend(h_right+h_left, l_right+l_left, loc = 'upper right', fontsize = fontsize - 1)
        
    return res_data, fit_param, w, fig


def ImportAppCurve(filepath, visualize = False):
    data = pd.read_csv(filepath, header = None, delimiter = '\t', engine = 'python', skipfooter = 20) # Skip all the comments at the bottom
    data = data.drop(data.columns[3], axis=1) # Drop the Aux1 measurements
    data.rename(columns = {0:'z(nm or bits)', 1:'Amplitude(V)', 2:'Phase(V)'}, inplace = True)

    imin = data.idxmin(axis = 0)[0]
    
    if visualize:
        # For the visualize keyword, create a figure of the experimental and fitted resonance curve
        fontsize = 13
        labelpad = 10
        tkw = dict(size = 6, width = 1.5, labelsize = fontsize)
        
        fig, ax_left = plt.subplots(1, 1, figsize = (7,5))
        ax_left.plot(data.iloc[0:imin+1,0], data.iloc[0:imin+1,1], '.-k', alpha = 0.8, label = 'Amplitude_Approach(V)')
        ax_left.plot(data.iloc[imin+1:,0], data.iloc[imin+1:,1], '.-b', alpha = 0.8, label = 'Amplitude_Retract(V)')
        ax_left.set_xlabel('Frequency (Hz)', fontsize = fontsize, labelpad = labelpad)
        ax_left.set_ylabel('Amplitude (V)', fontsize = fontsize, labelpad = labelpad)
        
        ax_right = ax_left.twinx()
        ax_right.plot(data.iloc[0:imin+1,0], data.iloc[0:imin+1,2], '.-r', alpha = 0.8, label = 'Phase_Approach(V)')
        ax_right.plot(data.iloc[imin+1:,0], data.iloc[imin+1:,2], '.-m', alpha = 0.8, label = 'Phase_Retract(V)')
        ax_right.set_ylabel('Phase(V)', fontsize = fontsize, labelpad = labelpad)
        ax_left.grid(ls = '--')
        
        h_right, l_right = ax_right.get_legend_handles_labels()
        h_left, l_left = ax_left.get_legend_handles_labels()
        
        ax_left.tick_params(axis='x', **tkw)
        ax_left.tick_params(axis='y', **tkw)
        ax_right.tick_params(axis='y', colors='red', **tkw)
        ax_right.yaxis.label.set_color('red')
            
        ax_right.legend(h_right+h_left, l_right+l_left, loc = 'upper right', fontsize = fontsize - 1)
        
    return data, imin


'''
** DataReformatter(filepath, savepath) **

Inputs:
- folderpath: path to the folder containing the AFM raw data. Each data run in the folder must be organized into subfolders, 
with one resonance curve and multiple approach curves per subfolder

- savepath: path to save the reformatted file, which is in the npz format

- n_deriv: maximum order of the derivatives(with respect to z) of amplitude and phase to compute

- savgol_param: parameters for the savgol_filter used in the differentiation. In the format of [window_size, poly_order].

Outputs:
- total_dataset: the reformatted AFM dataset in numpy dictionary .npz format
'''
def DataReformatter(folderpath, savepath):
        
    ## 1. initialize dataset arrays ##
    
    filename_dataset = [] # filename
    
    raw_dataset = [] # Lock-in raw data, [z, Amp(V), Phas(rad)] format
    mech_dataset = [] # Mechanical Amp, Phas,[z, Amp(V), Phas(rad)] format

    imin_dataset = [] # array index of the turning point : Appoach is from 0 ~ imin, Retract is from imin ~ end

    # Resonance curve fitting parameters
    Q_dataset = []
    w0_dataset = []
    w_dataset = [] # Driving frequency used in the experiment
    I0_dataset = []
    C0C_dataset = []

    E_dataset = [] # Dissipated energy, normalized by kA0^2
    kint_dataset = [] # k_int, normalized by k
    bint_dataset = [] # b_int, normalized by k

    Fk_dataset = [] # <F_k> = kA, normalized by kA0
    Fb_dataset = [] # <F_b> = bwA, normalized by kA0

    
    ## 2. Parse each subfolder and identify the resonance curve ##
    subfolder = os.listdir(folderpath)
    
    for sf in subfolder:
        files = os.listdir(os.path.join(folderpath, sf)) 
    
        # Isolate the resonance curve file from the approach curve files
        i = 0
        while i < len(files):
            if 'res' in files[i]:
                res_file = files.pop(i)
            else:
                i += 1
            
        # Extract Q and w0 from the resonance curve file
        Q = None
        w0 = None
        w = None # Driving frequency used in the experiment
        I_0 = None
        C0_C = None
    
        res_file_path = os.path.join(folderpath, sf, res_file)

        res_data = pd.read_csv(res_file_path, header = None, delimiter = '\t', skiprows = 9)
        res_data = res_data.values
    
        w0_init = res_data[res_data[:,1].argmax(),0] # Frequency of the maximum amplitude measured. This is the driving frequency used
        w = w0_init

        popt, _= curve_fit(_ResCurve, res_data[:,0], res_data[:,1], p0 = [w0_init, 5000, 3, 1/800], bounds = (0, np.inf))
        [w0, Q, I_0, C0_C] = popt
    
        # Parse datafiles in the subfolder
        for i, f in enumerate(files):
                    
            filepath = os.path.join(folderpath, sf, f)
            filename_dataset.append(sf+'-'+f)
        
            Q_dataset.append(Q)
            w0_dataset.append(w0)
            w_dataset.append(w)
            I0_dataset.append(I_0)
            C0C_dataset.append(C0_C)
        
            raw_data = pd.read_csv(filepath, header = None, delimiter = '\t', engine = 'python', skipfooter = 20) # Skip all the comments at the bottom
            raw_data = raw_data.drop(raw_data.columns[3], axis=1) # Drop the Aux1 measurements

            # Convert pandas dataframe to numpy array
            data = raw_data.values # z = data[:,0], Ae = data[:,1], Pe = data[:,2]

            # Convert phase to radian and offset it
            data[:,2] = _CalibratePhase(data[:,2], data[0,2], w, w0, Q, C0_C)
        
            # Create raw data dataset
            raw_dataset.append(data)
        
            # Find turning point index imin
            imin = np.argmin(data[:,0])
            imin_dataset.append(imin)
        
            # Calculate mechanical amplitude and phase, at frequency w!=w0
            
            mech_data = np.empty(data.shape) # z = data[:,0], Am = mech_data[:,1], Pm = mech_data[:,2]
            
            mech_data[:,0] = data[:,0]
            mech_data[:,1] = _MechAmp(data[:,1], data[:,2], w, w0, Q, C0_C, I_0)
            mech_data[:,2] = _MechPhas(data[:,1], data[:,2], w, w0, Q, C0_C, I_0)
            
            mech_dataset.append(mech_data)
            
            # Normalize mechanical amplitude
            mech_data[:,1] = mech_data[:,1]/mech_data[0,1]

            # Offset mechanical phase
            mech_data[:,2] = mech_data[:,2] - mech_data[0,2]
        
            # Calculate normalized energy dissipation E
            E = _Edis(mech_data[:,1], mech_data[:,2], w, w0, Q)
            E = E - E[0]
            E_dataset.append(E)
        
            # Calculate normalized kint and bint
            kint = _kint(mech_data[:,1], mech_data[:,2], w, w0, Q)
            kint = kint - kint[0]
            
            bint = _bint(mech_data[:,1], mech_data[:,2], w, w0, Q)
            bint = bint - bint[0]
        
            kint_dataset.append(kint)
            bint_dataset.append(bint)
        
            # Calculate normalized Fk and Fb
            Fk = kint*mech_data[:,1]
            Fb = bint*w*mech_data[:,1]
            Fk_dataset.append(Fk)
            Fb_dataset.append(Fb)

            
    ## Convert the results into numpy arrays ##    
    filename_dataset = np.array(filename_dataset)

    raw_dataset = np.array(raw_dataset)
    mech_dataset = np.array(mech_dataset)

    imin_dataset = np.array(imin_dataset)

    Q_dataset = np.array(Q_dataset)
    w0_dataset = np.array(w0_dataset)
    w_dataset = np.array(w_dataset)
    I0_dataset = np.array(I0_dataset)
    C0C_dataset = np.array(C0C_dataset)

    E_dataset = np.array(E_dataset)

    kint_dataset = np.array(kint_dataset)
    bint_dataset = np.array(bint_dataset)

    Fk_dataset = np.array(Fk_dataset)
    Fb_dataset = np.array(Fb_dataset)

    ## Arrange all data into dictionary format for saving ##

    total_dataset = dict([('filename', filename_dataset), ('raw', raw_dataset), ('mech', mech_dataset), ('imin', imin_dataset), ('Q', Q_dataset), ('w0', w0_dataset), ('w', w_dataset), ('I0', I0_dataset), ('C0C', C0C_dataset), ('E', E_dataset), ('kint', kint_dataset), ('bint', bint_dataset), ('Fk', Fk_dataset), ('Fb', Fb_dataset)])
    np.savez(savepath, **total_dataset)

    print('Saved data to: {}'.format(savepath))             
    
    return total_dataset

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
    
    # create data array : (trials, seq_length, 5)
    data = np.zeros((N, seq_length, 5))
    for i in range(N):
        data[i,:,0:3] = mech_dataset[i][imin_dataset[i]+1-seq_length:imin_dataset[i]+1, 0:3]
        data[i,:,3] = np.ones((1, seq_length))*Q_dataset[i]/w0_dataset[i] # 3rd channel is Q/w0
        data[i,:,4] = np.ones((1, seq_length))*w_dataset[i]/w0_dataset[i] # 4th channel is w/w0
    
    # change the phase to cos(phase) (this is to prevent the true value from becoming 0, which results in /0 blowup when calculating the relative loss)
    data[:,:,2] = np.cos(data[:,:,2])
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
