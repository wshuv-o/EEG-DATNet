import os
import numpy as np

from plot_c3c4cz import MotorImageryDataset

# Class mapping for Motor Imagery
class_mapping = {
    'tongue': 0,
    'foot': 1,
    'right': 2,
    'left': 3,
    'unknown': 4
}

# Function to load and process the data files
def process_dataset_files(input_folder, output_folder):
    # Get all .npz files in the dataset folder
    files = [f for f in os.listdir(input_folder) if f.endswith('.npz')]
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through all the files
    for file in files:
        print(f"Processing {file}...")
        
        # Load the dataset (assuming the class is similar to MotorImageryDataset)
        datasetA1 = MotorImageryDataset(os.path.join(input_folder, file))
        
        # Extract trials and classes
        trials, classes = datasetA1.get_trials_from_channels(range(22))
        classes=classes[0]
        # Print the shape of trials and classes
        print(f"Original trials shape: {np.array(trials).shape}")
        print(f"Original classes shape: {np.array(classes).shape}")

        mapped_classes = np.array([class_mapping[label] for label in classes])

        # Reshape trials: [n_epochs, n_channels, n_times]
        reshaped_trials = np.array(trials).transpose(1, 0, 2)
        print(f"Reshaped trials shape: {reshaped_trials.shape}")

        # Prepare the output file names based on the input file name
        file_name = file[:-4]  # Remove .npz extension
        
        # Save the trials and classes as .npy files in the processed folder
        np.save(os.path.join(output_folder, f"{file_name}_data.npy"), reshaped_trials)
        np.save(os.path.join(output_folder, f"{file_name}_label.npy"), mapped_classes)
        
        print(f"Saved {file_name}_data.npy and {file_name}_label.npy to {output_folder}")

# Define the input and output folder paths
input_folder = 'dataset'  # Folder containing the original .npz files
output_folder = 'processed'  # Folder where the .npy files will be saved

# Process the dataset files
process_dataset_files(input_folder, output_folder)
