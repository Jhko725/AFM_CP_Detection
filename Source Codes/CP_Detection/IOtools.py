import os, shutil
from numpy import savez, load
import pandas as pd
from zipfile import ZipFile
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from .Model import weighted_seq_mse, amp_mse, phas_mse

def RunandSave(model, hyperparameters, train, test, filepath, max_epochs = 100000, save_best = True, save_final = True):
    
    # Create temporary folder to save the files in
    base_dir = os.path.dirname(filepath)
    filename = os.path.basename(filepath).split('.')[0]
    temp_dir = os.path.join(base_dir, filename)

    # Create keras checkpoint for saving the best weight
    autoencoder = model['autoencoder']
    best_weight_path = os.path.join(temp_dir, "weights.best.hdf5")
    checkpoint = ModelCheckpoint(best_weight_path, monitor='val_acc', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    autoencoder.fit(x = [train[:,:,1:3], train[:,:,3:]], y = train[:,:,1:3], epochs = max_epochs, batch_size = hyperparameters['batch_size'], validation_data = ([test[:,:,1:3], test[:,:,3:]], test[:,:,1:3]), callbacks = callbacks_list)

    # Need to add more code... Only needs to save the autoencoder structure
    # Change load model so that the encoder and the decoder parts are created from the autoencoder weights


def SaveModel(filepath, model, hyperparameters):
    
    # Define subfunction that saves the model architecture and weights
    def SaveModelandWeights(savepath, model_name, model):
        model_path = os.path.join(savepath, model_name+'.h5')
        model.save(model_path)
        
    # Define subfunction that gets the list of all files in a given directory
    def GetAllFilepaths(directory):
        filepaths = []
        
        for root, _, files in os.walk(directory): 
            for filename in files: 
                filepath = os.path.join(root, filename) 
                filepaths.append(filepath) 
   
        return filepaths         
  
    base_dir = os.path.dirname(filepath)
    filename = os.path.basename(filepath).split('.')[0]
    
    # Create a temporary directory to save the files in
    temp_dir = os.path.join(base_dir, filename)
    if os.path.exists(temp_dir):
        raise Exception('The path of the temporary directory clashes with an existing one. Change the filename and try again')
    os.makedirs(temp_dir)
    
    # Save the Keras model
    SaveModelandWeights(temp_dir, 'encoder', model['encoder'])
    SaveModelandWeights(temp_dir, 'decoder', model['decoder'])
    SaveModelandWeights(temp_dir, 'autoencoder', model['autoencoder'])
    
    # Save the model hyperparameters
    param_path = os.path.join(temp_dir, 'hyperparameters.npz')
    savez(param_path, **hyperparameters)
    
    # Save the model history
    history_path = os.path.join(temp_dir, 'history.csv')
    history = pd.DataFrame(model['autoencoder'].history.history)
    history.to_csv (history_path, index = None, header=True)

    # Now zip all the files in the temporary directory
    filepaths = GetAllFilepaths(temp_dir)
    
    with ZipFile(filepath, 'w') as zip: 
        for file in filepaths: 
            zip.write(file, os.path.basename(file))
    
    # Finally, delete the temporary directory and all its contents
    shutil.rmtree(temp_dir)
    
    print('Model saved successfully at %s' %(filepath))
    
    
def LoadModel(filepath):
    
    # Define subfunction that saves the model architecture and weights
    def LoadModelandWeights(savepath, model_name):
        model_path = os.path.join(savepath, model_name+'.h5')
        model = load_model(model_path, custom_objects={'weighted_seq_mse': weighted_seq_mse, 'amp_mse': amp_mse, 'phas_mse': phas_mse})

        return model

    # Create a temporary directory to unzip the files
    base_dir = os.path.basename(filepath)
        
    temp_dir = os.path.join(base_dir, 'temp_dir')
    if os.path.exists(temp_dir):
        raise Exception('The path of the temporary directory clashes with an existing one. Change the filename and try again')
    os.makedirs(temp_dir)

    with ZipFile(filepath, 'r') as zip:
        zip.extractall(path = temp_dir)
    
    # Load the unzipped files 
    encoder = LoadModelandWeights(temp_dir, 'encoder')
    decoder = LoadModelandWeights(temp_dir, 'decoder')
    autoencoder = LoadModelandWeights(temp_dir, 'autoencoder')
    model = {'encoder': encoder, 'decoder': decoder, 'autoencoder': autoencoder}
    
    param_path = os.path.join(temp_dir, 'hyperparameters.npz')
    hyperparameters = dict(load(param_path))
    
    history_path = os.path.join(temp_dir, 'history.csv')
    history = pd.read_csv(history_path, header = 0)

    # Delete the temporary directory and all its contents
    shutil.rmtree(temp_dir)

    return model, hyperparameters, history

