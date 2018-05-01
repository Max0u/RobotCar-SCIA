from augmentation import *
from model import *

# Visualizations will be shown in the notebook.
import pandas as pd
import numpy as np


def load_csv(file_path, col_names, remove_header=False):
    csv = pd.read_csv(file_path, header=None, names=col_names)
    if remove_header:
        csv = csv[1:]
    
    return csv


# Define our column headers
col_header_names = ["Center", "Left", "Right", "Steering Angle", "Throttle", "Brake","Speed"]


# Let's load our standard driving dataset, where we drive in both directions
standard_csv = load_csv(standard_path + "/driving_log.csv", col_header_names)
standard_csv["Steering Angle"] = standard_csv["Steering Angle"].astype(float) 
print("Standard dataset has {0} rows".format(len(standard_csv)))

# Let's load the data set of driving in recovery mode (aka virtual drink driving dataset)
recovery_csv = load_csv(recovery_path + "/driving_log.csv", col_header_names)
recovery_csv["Steering Angle"] = recovery_csv["Steering Angle"].astype(float) 
print("Recovery dataset has {0} rows".format(len(recovery_csv)))

# Let's load dataset from track 2
track2_csv = load_csv(track2_path + "/driving_log.csv", col_header_names)
track2_csv["Steering Angle"] = track2_csv["Steering Angle"].astype(float) 
print("Track 2 dataset has {0} rows".format(len(track2_csv)))

# Finally let's load the udacity dataset
udacity_csv = load_csv(udacity_path + "/driving_log.csv", col_header_names, remove_header=True)
udacity_csv["Steering Angle"] = udacity_csv["Steering Angle"].astype(float) 
print("Standard dataset has {0} rows".format(len(udacity_csv)))


def get_steering_angles(data, st_column, st_calibrations, filtering_f=None):
    """
    Returns the steering angles for images referenced by the dataframe
    The caller must pass the name of the colum containing the steering angle 
    along with the appropriate steering angle corrections to apply
    """
    cols = len(st_calibrations)
    print("CALIBRATIONS={0}, ROWS={1}".format(cols, data.shape[0]))
    angles = np.zeros(data.shape[0] * cols, dtype=np.float32)
    
    i = 0
    for indx, row in data.iterrows():        
        st_angle = row[st_column]
        for (j,st_calib) in enumerate(st_calibrations):  
            angles[i * cols + j] = st_angle + st_calib
        i += 1
    
    # Let's not forget to ALWAYS clip our angles within the [-1,1] range
    return np.clip(angles, -1, 1)

# Defining our columns of interests as well as steering angle corrections
st_angle_names = ["Center", "Left", "Right"]
st_angle_calibrations = [0, 0.25, -0.25]

# Using an ensemble of datasets as training set
frames = [recovery_csv, udacity_csv, track2_csv]
ensemble_csv = pd.concat(frames)

# Our initially captured dataset on track 1 will act as the validation set
validation_csv = standard_csv

# In this section we load and train the model...

b_divider = 20
# Multiplying by 3 since we have center, left and right images per row
b_size = len(ensemble_csv)  * 3 // b_divider

m = nvidia_model()
gen_train = generate_images(ensemble_csv, (160, 320, 3), st_angle_names, "Steering Angle", st_angle_calibrations,  batch_size=b_size)

# Take 20% of the data for validation
gen_val = generate_images(validation_csv, (160, 320, 3), st_angle_names, "Steering Angle", st_angle_calibrations,  batch_size=(b_size * b_divider) // 5, data_aug_pct=0.0)
x_val, y_val = next(gen_val)

# Train the model
m.fit_generator(gen_train, validation_data=(x_val, y_val), samples_per_epoch=b_size * b_divider, nb_epoch=2, verbose=1)

m.save("model.h5")

print("Successfully saved model")


