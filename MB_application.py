########################################################################
# import default libraries
########################################################################
import os
import csv
import sys
import gc
########################################################################


########################################################################
# import additional libraries
########################################################################
import numpy as np
import scipy.stats
# from import
from tqdm import tqdm
try:
    from sklearn.externals import joblib
except:
    import joblib
# original lib
from MB import common as com
from MB import keras_model
########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
########################################################################

def load_model(machine_type):
    # load model file
    model_file = "./MB/model/model_{machine_type}.hdf5".format(machine_type=machine_type)
    model = keras_model.load_model(model_file)

    # load section names for conditioning
    section_names_file_path = "./MB/model/section_names_{machine_type}.pkl".format(machine_type=machine_type)
    trained_section_names = joblib.load(section_names_file_path)
    n_sections = trained_section_names.shape[0]

    # load anomaly score distribution for determining threshold
    score_distr_file_path = "./MB/model/score_distr_{machine_type}.pkl".format(machine_type=machine_type)
    shape_hat, loc_hat, scale_hat = joblib.load(score_distr_file_path)

    # determine threshold for decision
    decision_threshold = scipy.stats.gamma.ppf(q=param["decision_threshold"], a=shape_hat, loc=loc_hat, scale=scale_hat)

    return model, trained_section_names, decision_threshold

def get_prediction(model, trained_section_names, decision_threshold, test_file_path):
    n_sections = trained_section_names.shape[0]
    section_names = ["section_00", "section_01", "section_02"]
    #search for section_name
    #if the section_name is not found in the trained_section_names, store -1 in section_idx
    section_name = "section_" + test_file_path.split("/")[-1].split("_")[1]
    print(section_name)
    temp_array = np.nonzero(trained_section_names == section_name)[0]
    if temp_array.shape[0] == 0:
        section_idx = -1
    else:
        section_idx = temp_array[0] 
    print(section_idx)
    # for file_idx, file_path in tqdm(enumerate(files), total=len(files)):
    try:
        data = com.file_to_vectors(test_file_path,
                                        n_mels=param["feature"]["n_mels"],
                                        n_frames=param["feature"]["n_frames"],
                                        n_fft=param["feature"]["n_fft"],
                                        hop_length=param["feature"]["hop_length"],
                                        power=param["feature"]["power"])
    except:
        com.logger.error("File broken!!: {}".format(test_file_path))

    # make one-hot vector for conditioning
    condition = np.zeros((data.shape[0], n_sections), float)
    # if the id_name was found in the trained_section_names, make a one-hot vector
    if section_idx != -1:
        condition[:, section_idx : section_idx + 1] = 1

    # 1D vector to 2D image
    data = data.reshape(data.shape[0], param["feature"]["n_frames"], param["feature"]["n_mels"], 1)

    p = model.predict(data)[:, section_idx : section_idx + 1]
    result_score = np.mean(np.log(np.maximum(1.0 - p, sys.float_info.epsilon) 
                                - np.log(np.maximum(p, sys.float_info.epsilon))))

    # get decision results
    if result_score > decision_threshold:
        return 1 # Anomaly
    else:
        return 0 # Normal