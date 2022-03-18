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
from sklearn import metrics
try:
    from sklearn.externals import joblib
except:
    import joblib
# original lib
from AE import common as com
from AE import keras_model
########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
#######################################################################

def load_model(machine_type):
    # load model file
    model_file = "./AE/model/model_{machine_type}.hdf5".format(machine_type=machine_type)
    model = keras_model.load_model(model_file)

    # load anomaly score distribution for determining threshold
    score_distr_file_path = "./AE/model/score_distr_{machine_type}.pkl".format(machine_type=machine_type)
    shape_hat, loc_hat, scale_hat = joblib.load(score_distr_file_path)

    # determine threshold for decision
    decision_threshold = scipy.stats.gamma.ppf(q=param["decision_threshold"], a=shape_hat, loc=loc_hat, scale=scale_hat)

    return model, decision_threshold

def get_prediction(model, decision_threshold, test_file_path):
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

    result_score = np.mean(np.square(data - model.predict(data)))

    # get decision results
    if result_score > decision_threshold:
        return 1 # Anomaly
    else:
        return 0 # Normal