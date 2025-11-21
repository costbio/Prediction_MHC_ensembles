###################################
# REQUIREMENTS
###################################
import mdtraj as md
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


###################################
# DATA PREPROCESSING
###################################

def load_trajectory(dcd_file, pdb_file, atom_selection="name N or name CA or name C or name O"):
 
    traj = md.load_dcd(dcd_file, top=pdb_file)
    traj = traj.atom_slice(traj.topology.select(atom_selection))
    print(f"> Loaded {traj.n_frames} frames, {traj.n_atoms} selected atoms.")
    return traj

def align_trajectory(traj):

    ref = traj[0]
    traj.superpose(ref)
    print("> All frames aligned to reference (frame 0).")
    return traj

def split_train_test(traj, train_ratio=0.1):

    coords = traj.xyz.reshape(traj.n_frames, -1)  
    n_train = int(train_ratio * traj.n_frames)
    x_train = deepcopy(coords[:n_train])
    x_test = deepcopy(coords[n_train:])
    print(f"> Split into {x_train.shape[0]} train and {x_test.shape[0]} test frames.")
    return x_train, x_test

def save_data(base_name, x_train, x_test, output_dir="processed_data"):

    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(os.path.join(output_dir, f"{base_name}_train.dat"), x_train)
    np.savetxt(os.path.join(output_dir, f"{base_name}_test.dat"), x_test)
    print(f"> Data saved under '{output_dir}/'")


def load_data(base_name, output_dir="processed_data"):
 
    x_train = np.loadtxt(os.path.join(output_dir, f"{base_name}_train.dat"))
    x_test = np.loadtxt(os.path.join(output_dir, f"{base_name}_test.dat"))
    print(f"> Data loaded from '{output_dir}/'")
    return x_train, x_test

def normalize_data(x_train, x_test):

    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    print("> Data normalized to [0, 1].")
    return x_train_scaled, x_test_scaled, scaler


def preprocess_pipeline(dcd_file, pdb_file, protein_name, train_ratio):

    traj = load_trajectory(dcd_file, pdb_file)
    traj_align = align_trajectory(traj)
    x_train, x_test = split_train_test(traj_align, train_ratio)
    save_data(protein_name, x_train, x_test)
    x_train_orig, x_test_orig = load_data(protein_name)
    x_train_scaled, x_test_scaled , scaler = normalize_data(x_train, x_test)
    print("> Preprocessing complete.")
    return x_train_scaled, x_test_scaled, x_train_orig, x_test_orig , scaler

###################################
# MODEL STRUCTURE
###################################


def sampling(args):
    mean, logvar = args
    epsilon = K.random_normal(shape=K.shape(mean), mean=0., stddev=1.)
    return mean + K.exp(0.5 * logvar) * epsilon
def vae_structure(x_train, encoding_dim):
    input_dim = x_train.shape[1]
    input_prot = Input(shape=(input_dim,))

    # encoder
    encoded = Dense(512, activation='relu')(input_prot)
    encoded = Dense(256, activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.3)(encoded)
    encoded = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(16, activation='relu')(encoded)
    encode_mean = Dense(encoding_dim)(encoded)
    encode_log_var = Dense(encoding_dim)(encoded)
    sample = Lambda(sampling)([encode_mean, encode_log_var])

    # decoder
    decoded = Dense(16, activation='relu')(sample)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(256, activation='relu')(decoded)
    decoded = Dense(512, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    sampling_model = Model(input_prot, decoded)

    # define loss, optimizer
    #Reconstruction Loss
    msle = tf.keras.losses.MeanSquaredLogarithmicError()
    xent_loss = K.mean(msle(input_prot, decoded))
    
    #KL divergence loss
    kl_loss = -1e-3 * K.mean(1 + encode_log_var - K.square(encode_mean) - K.exp(encode_log_var))
    vae_loss = K.mean(xent_loss +  kl_loss)  

    sampling_model.add_loss(vae_loss)

    adam = Adam(learning_rate=0.0005)
    sampling_model.compile(optimizer=adam, metrics='mean_squared_error')

    return sampling_model
    


###################################    
# MODEL TRAINING
###################################

def model_training(Batch_size, epochs, xtrain, xtest):
    sampling_model = vae_structure(xtrain, 4)
   
    #adding early stopping that means stops training when validation performance stops improving
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = sampling_model.fit(
        xtrain, xtrain,
        epochs=epochs,
        batch_size=Batch_size,
        shuffle=True,
        validation_data=(xtest, xtest),
        callbacks=[early_stopping]
    )

     # plot for training nd validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return history , sampling_model

###################################    
# MODEL PREDICTION
###################################


def reconstruct(x_train_scaled, x_test_scaled, model, scaler):
 
    rc_train = model.predict(x_train_scaled)
    reconstruct_train = scaler.inverse_transform(rc_train)
    
    rc_test = model.predict(x_test_scaled)
    reconstruct_test = scaler.inverse_transform(rc_test)
  
    decoded_reshaped_train = reconstruct_train.reshape(
        reconstruct_train.shape[0],
        int(reconstruct_train.shape[1] / 3),
        3
    )
    decoded_reshaped_test = reconstruct_test.reshape(
        reconstruct_test.shape[0],
        int(reconstruct_test.shape[1] / 3),
        3
    )
    
    return decoded_reshaped_train, reconstruct_train, decoded_reshaped_test, reconstruct_test



###################################    
# ANALYSES
###################################

def save_selected_atoms(pdb_file, atom_selection="name N or name CA or name C or name O", output_dir="selected_pdbs"):
    os.makedirs(output_dir, exist_ok=True)
    
    traj = md.load_pdb(pdb_file)

    atom_indices = traj.topology.select(atom_selection)
    
    selected_traj = traj.atom_slice(atom_indices)
  
    base_name = os.path.splitext(os.path.basename(pdb_file))[0]
    output_path = os.path.join(output_dir, f"{base_name}_selected.pdb")
    
    selected_traj.save_pdb(output_path)
    
    return output_path


def create_dcd_from_decoded(pdb_file, decoded_coords, output_prefix=None, output_dir="generated_files"):
   
    os.makedirs(output_dir, exist_ok=True)
    
   
    ref_pdb = md.load_pdb(pdb_file)
    

    new_traj = md.Trajectory(decoded_coords, ref_pdb.topology)
    
    align_traj = new_traj.superpose(ref_pdb)

 

    if output_prefix is None:
        base = os.path.splitext(os.path.basename(pdb_file))[0]
    else:
        base = output_prefix
    
    dcd_path = os.path.join(output_dir, f"{base}_generated.dcd")
    
    align_traj.save_dcd(dcd_path)
    print(f"> DCD file saved: {dcd_path}")
    
    return dcd_path



def compute_rmsd(reference_pdb, dcd_file, atom_selection="name N or name CA or name C or name O", output_txt=None):

    ref = md.load_pdb(reference_pdb)

    atom_indices = ref.topology.select(atom_selection)
   
    traj = md.load_dcd(dcd_file, top=reference_pdb)
    

    traj = traj.atom_slice(atom_indices)
    ref = ref.atom_slice(atom_indices)
    
    rmsd_values = md.rmsd(traj, ref)
    
    plt.figure(figsize=(8,6))
    plt.plot(rmsd_values)
    plt.xlabel("Frame")
    plt.ylabel("RMSD (nm)")
    plt.title("RMSD vs Frame")
    plt.show()
  
    if output_txt:
        np.savetxt(output_txt, rmsd_values)
        print(f"> RMSD values saved: {output_txt}")
    
    
    return rmsd_values
