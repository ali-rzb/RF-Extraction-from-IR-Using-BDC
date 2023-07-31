import Functions.GlobalUtils as g_utils
import Functions.DataUtils as d_utils
import DATA_INFO as data_info
import Functions.LocalDB as db
from Functions.CNN import CNN
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import sklearn.metrics as metrics

def load_prediction_data(path):
    data = db.get_db(path, CNN.prediction_data)
    
    borders_denorm = [f_min] + list(g_utils.denormalize(data[i].borders[1:-1], f_max, f_min)) + [f_max]