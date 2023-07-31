__copyright__ = """

    Copyright 2022 Ali Roozbehi

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""

import numpy as np
import sys, os, time, math
import sklearn.metrics as mtrcs
import tensorflow_addons as tfa
from scipy.interpolate import interp1d
from scipy import signal
import keras.backend as K
from time import sleep
import tensorflow as tf



class ProgressBar:
    def __init__(self, end, wait_time = 0, print_with_details = False):
        self.wait_time = wait_time
        self.end = end
        self.percent = 0
        self.details = print_with_details
        self.status = -1
        self.ProgressUnit = '█'

        if print_with_details:
            self.timer = timer()
            self.timer.start()
            
            self.total_time_timer = timer()
            self.total_time_timer.start()
            
            self.last_time_ckecked = time.time()
            self._min = -1
            self._sec = -1
            self.times = []
            self.step_per_sec = -1
        
        
        
        self.update(False)

    def update(self, _print = True):
        self.status = self.status + 1
        self.percent = 100 * self.status/self.end

        sys.stdout.write('\r')
        # the exact output you're looking for:
        if self.details:
            self.timer.stop()
            self.times.append(self.timer.exact_passed_time)

            if (time.time()-self.last_time_ckecked)>0.5 or self._min == -1:
                if self.timer.exact_passed_time != 0:
                    self.step_per_sec = int(1000/self.timer.exact_passed_time)
                    if self.step_per_sec == 0:
                        self.step_per_min = int(60*1000/self.timer.exact_passed_time)
                        if self.step_per_min == 0:
                            self.step_per_10_min = int(10*60*1000/self.timer.exact_passed_time)
                if self._min == -1:
                    self.remaining_ms = np.mean(self.times)*(self.end-self.status)
                else:
                    quarter = int(len(self.times)/4)
                    self.remaining_ms = np.mean(self.times[quarter:-1])*(self.end-self.status)
                self._min = int((self.remaining_ms/1000)/60)
                self._sec = int((self.remaining_ms/1000)%60)
                self.last_time_ckecked = time.time()

            if _print:
                if self.status == self.end:
                    self.total_time_timer.stop()
                    total_time = ' - passed time : ' + self.total_time_timer.labeled_time
                else:
                    total_time = ''
                if self.step_per_sec != 0:
                    sys.stdout.write("|%-40s| %3.2f%% - %4d/%4d - ETA : %3dm %2ds - %d step/s%s" % (self.ProgressUnit*int(self.percent/2.5), self.percent, self.status, self.end, self._min, self._sec, self.step_per_sec, total_time))
                elif self.step_per_min != 0:
                    sys.stdout.write("|%-40s| %3.2f%% - %4d/%4d - ETA : %3dm %2ds - %d step/min%s" % (self.ProgressUnit*int(self.percent/2.5), self.percent, self.status, self.end, self._min, self._sec, self.step_per_min, total_time))
                elif self.step_per_10_min != 0:
                    sys.stdout.write("|%-40s| %3.2f%% - %4d/%4d - ETA : %3dm %2ds - %d step/10min%s" % (self.ProgressUnit*int(self.percent/2.5), self.percent, self.status, self.end, self._min, self._sec, self.step_per_10_min, total_time))
                else:
                    sys.stdout.write("|%-40s| %3.2f%% - %4d/%4d - ETA : %3dm %2ds%s" % (self.ProgressUnit*int(self.percent/2.5), self.percent, self.status, self.end, self._min, self._sec, total_time))
            self.timer.start()
        else:
            if _print:
                sys.stdout.write("[%-20s] %3.2f%%" % ('='*int(self.percent/5), self.percent))
        sys.stdout.flush()
        sleep(self.wait_time/1000)

class Signal:
    def __init__(self, X, Y, X_LBL = None, Y_LBL = None):
        self.X      = np.array(X)
        self.Y      = np.array(Y)
        
        if X_LBL is not None:
            self.X_LBL  = np.array(X_LBL)
        if Y_LBL is not None:
            self.Y_LBL  = np.array(Y_LBL)
        
class timer:
    def __init__(self):
        self.sum = 0
        self.index = 0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.stop_time = time.time()
        self.exact_passed_time = (self.stop_time - self.start_time)*1000
        self.passed_time = round(self.exact_passed_time)


        self.sum = self.sum + self.exact_passed_time
        self.index = self.index + 1
        self.mean = self.sum / self.index
        
        self.labeled_time = self.to_labeled_time(self.exact_passed_time)
        
    
    @staticmethod
    def to_labeled_time(ms_time):
        if ms_time >= 3600000:
            return '{}h'.format(round(ms_time/(60*60*1000), 2))
            
        elif ms_time >= 60000:
            return '{}m'.format(round(ms_time/(60*1000), 2))
            
        elif ms_time >= 1000:
            return '{}s'.format(round(ms_time/(1000), 2))
        else:
            return '{}ms'.format(round(ms_time, 2))

class Section_Separator:
    def __init__(self, Name):
        return None
    def __enter__(self):
        return self
    def __exit__(self,e1, e2, e3):
        return None
        

def equalize_lists(*lists, mode = 'zero'):
    m = len(lists[0])
    for _list in lists:
        m = max(m, len(_list))

    for i, _list in enumerate(lists):
        for j in range(m-len(_list)):
            if mode == 'zero':
                lists[i].append(0)
            elif mode == 'Last':
                lists[i].append(lists[i][-1])
            else:
                raise Exception('Parameter mode is \'zero\' or \'Last\'')
    return lists

def median_filter(signal, kernel_size):
    result = signal.copy()
    if not kernel_size % 2:
        kernel_size = kernel_size - 1

    side = int(kernel_size/2)

    for i in range(side, len(result)-side):
        result[i] = np.median(result[i-side:i+side])

    return result

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    '''
        example : 
            order = 1
            fs = 10
            cutoff = 4
            filtered_signal = global_utils.butter_lowpass_filter(signal , cutoff, fs, order)
    '''
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

def get_free_file_name(path, prefix, suffix, return_index = False):
    i = 0
    while os.path.isfile(os.path.join(path, prefix.format(i)+suffix)):
        i = i + 1
    path = str.replace(path, '/', '\\')
    if return_index:
        return os.path.join(path, prefix.format(i)+suffix), i
    else:
        return os.path.join(path, prefix.format(i)+suffix)

def clamp(numbers, minn, maxn):
    for i, n in enumerate(numbers):
        numbers[i] = max(min(maxn, n), minn)
    return numbers

def isInRange(x, y, minx, maxx, miny, maxy):
    if x > minx and x < maxx and y > miny and y < maxy:
        return True
    else:
        return False

def fit(x, y, m):
    m = m + 1
    n = len(x)
    if m == -1:
        m = n-1
    A = np.zeros((n, m))
    for i in range(1, m+1):
        A[:, i-1] = np.power(x, m-i)
    a = np.dot(np.dot(np.linalg.inv(
        np.dot(np.transpose(A), A)), np.transpose(A)), y)
    X = np.arange(min(x), max(x), 0.01)
    Y = 0
    for i in range(0, m):
        Y = Y+a[m-i-1]*np.power(X, i)
    return Signal(X, Y)

def pack_equals(*lists):
    zip_list = zip(lists[0])
    for i, _list in enumerate(lists):
        if i != 0:
            flat_list = zip(*zip_list)
            flat_list = list(flat_list)
            flat_list.append(_list)
            zip_list = zip(*flat_list)
    return zip_list

def pack(*lists):
    lists = list(lists)
    i = 0
    while i < len(lists):
        if lists[i] == []:
            lists.pop(i)
        else:
            i = i + 1
    
        
    for i, _list in enumerate(lists):
        j = len(_list) - 1
        while math.isnan(_list[j]):
            lists[i] = lists[i][0:j]
            j = j - 1

    n_1 = len(lists[0])
    equal_lists = True
    for _list in lists:
        if len(_list) != n_1:
            equal_lists = False
            break

    if equal_lists:
        return pack_equals(*lists)

    else:
        m = len(lists[0])
        for _list in lists:
            m = max(m, len(_list))

        for i, _list in enumerate(lists):
            for j in range(m-len(_list)):
                lists[i].append(math.nan)

        return pack_equals(*lists)

def normalize(data, _max=None, _min=None, target_range = [0,1]):
    norm_data = np.array(data)
    if _max == None and _min == None:
        _max = np.max(norm_data)
        _min = np.min(norm_data)

    norm_data = norm_data - _min
    norm_data = norm_data / (_max-_min)

    if target_range[1] != 0:
        norm_data = norm_data * target_range[1]
        norm_data = norm_data + target_range[0]
    else:
        raise Exception('top range should not be zero !')
    return norm_data

def denormalize(data, _max, _min):
    norm_data = np.array(data)
    norm_data = norm_data * (_max-_min)
    norm_data = norm_data + _min
    
    return norm_data

def cut_from_signal(signal, time_span, cut_from, cut_to):
    (cut_from, cut_to) = sorted((cut_from, cut_to))

    cut_from_sample_num = None
    cut_to_sample_num = None

    for i in range(1, len(time_span) - 1):
        if time_span[i-1] <= cut_from and time_span[i+1] >= cut_from:
            cut_from_sample_num = i
        if time_span[i-1] <= cut_to and time_span[i+1] >= cut_to:
            cut_to_sample_num = i
            break
    
    _temp = list(signal[0:cut_from_sample_num]) + list(signal[cut_to_sample_num:-1])
    return _temp.copy()

def convert_to_smooth_line(x, y, gain = 10, kind = 'cubic'):
    """_summary_

        kind : ['zero', 'slinear', 'quadratic', 'cubic']

    """
    xnew = np.linspace(min(x), max(x), num=len(x)*gain, endpoint=True)
    f_linear = interp1d(x, y, kind=kind)
    return xnew, f_linear(xnew)

def get_proper_grid_size(_len):    
    grid_size_x = int(np.sqrt(_len)) + 1
    grid_size_y = grid_size_x
    for y in range(grid_size_x):
        if grid_size_x*y >= _len:
            grid_size_y = y
            break
    return grid_size_x, grid_size_y

def shutdown_system():
    os.system("shutdown /s /t 1")

def remove_duplicates(_data : list):
    i = 0
    data = _data.copy()
    while True:
        if len(data) == i:
            break
        duplicate_list = sorted([j for j,c in enumerate(data) if data[i]==c])
        delta = 0
        if len(duplicate_list)>1:
            pass
        for j in range(1, len(duplicate_list)):
            data.pop(duplicate_list[j] - delta)
            delta = delta + 1
        i = i + 1
    return data

def apply_kernel(_matrix,_kernel_func,_kernel_size, kernel_center = None):
        
    original_shape = np.shape(_matrix)
    if len(list(np.shape(_matrix))) == 1:
        _matrix = np.reshape(_matrix, (np.shape(_matrix)[0], 1))

    #find the x and y of the input matrix and the kernel
    row = np.shape(_matrix)[0]
    col = np.shape(_matrix)[1]

    if type(_kernel_size)==tuple:
        kernel_r=_kernel_size[0]
        kernel_c=_kernel_size[1]
    else:
        kernel_c=_kernel_size
        kernel_r=_kernel_size

    if kernel_r%2==0 or kernel_c%2==0:
        raise Exception("Kernel dimentions must be odd!")

    # calculate the desired padding 
    if kernel_center == None:
        pad_top = int(np.floor(kernel_r/2))
        pad_left = int(np.floor(kernel_c/2))
        
        pad_down = pad_top
        pad_right = pad_left
    else:
        pad_left = kernel_center[1] - 1
        pad_top = kernel_center[0] - 1
        
        if kernel_center[1] == -1:
            pad_left = kernel_c - 1
        if kernel_center[0] == -1:
            pad_top = kernel_r - 1
        
        pad_right = kernel_c - pad_left - 1
        pad_down = kernel_r - pad_top - 1
        
        
    #making a padded temporary matrix of input data
    temp = np.zeros((pad_top + row + pad_down,pad_left + col + pad_right))
    
    temp[pad_top:row+pad_top,pad_left:col+pad_left]=_matrix.copy()
    


    #making a new matrix for generated data
    result = np.zeros((row,col))

    for r in range(0,row):
        for c in range(0,col):
            
            # the window range in input matrix is [x-pad:x+pad] and in padded matrix is [x-pad + pad:x+pad + pad]
            target = temp[  r -pad_top +pad_top : r +pad_down +pad_top +1  ,  c -pad_left +pad_left : c +pad_right +pad_left +1  ]
            # input the window to the kernel function
            result[r][c]=_kernel_func(target)
    
    result = np.reshape(result, original_shape)
    return result

def kernel_majority(window):
    flattren = list(np.reshape(window,len(window)))
    dup_removed = list(set(flattren))
    dup_count = [len([x for x in flattren if x == c]) for c in dup_removed]
    majority = dup_removed[dup_count.index(max(dup_count))]
    return majority

def shift_signal(data, delta):
    _shape = np.shape(data)
    if delta >= 0:
        return np.reshape([0] * delta + list(data)[:-delta], _shape)
    else:
        return np.reshape(list(data)[-delta:] + [0] * abs(delta), _shape)

def find_pair_peaks(a, v, max_dist, fps = 1):
    _a_peaks = signal.find_peaks(a)[0]
    _v_peaks = signal.find_peaks(v)[0]
    
    a_peaks, v_peaks = [], []
    
    for i in range(min(min(_a_peaks), min(_v_peaks)) - np.int16(max_dist*fps), max(max(_a_peaks), max(_a_peaks)) + np.int16(max_dist*fps)):
        
        _range = [i, i + np.int16(max_dist*fps)]
        
        srch_1 = [p for p in _a_peaks if _range[0]<p<_range[1]]
        srch_2 = [p for p in _v_peaks if _range[0]<p<_range[1]]
        
        if len(srch_1) and len(srch_2):
            a_peaks.append(srch_1[0]/fps)
            v_peaks.append(srch_2[0]/fps)
            _a_peaks = [p for p in _a_peaks if p not in srch_1]
            _v_peaks = [p for p in _v_peaks if p not in srch_2]
            
    a_peaks = np.array(a_peaks)
    v_peaks = np.array(v_peaks)
    
    return a_peaks, v_peaks

def millify(n):
    millnames = ['',' Thousand',' Million',' Billion',' Trillion']
    n = float(n)
    millidx = max(0,min(len(millnames)-1,int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

    return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])

def get_pairs_picks(signal_a, signal_b, dt = 0.5):
    fps = 20
    result_a, result_b = [], []
    for j in range(min(min(signal_a), min(signal_b)) - np.int16(dt*fps), max(max(signal_a), max(signal_b)) + np.int16(dt*fps)):
        _range = [j, j + np.int16(dt*fps)]

        srch_1 = [p for p in signal_a if _range[0] < p < _range[1]]
        srch_2 = [p for p in signal_b if _range[0] < p < _range[1]]

        if len(srch_1) and len(srch_2):
            result_a.append(srch_1[0]/fps)
            result_b.append(srch_2[0]/fps)
            signal_a = [p for p in signal_a if p not in srch_1]
            signal_b = [p for p in signal_b if p not in srch_2]

    return np.int64(np.array(result_a)*fps), np.int64(np.array(result_b)*fps)

def get_peaks(signal_a, signal_b):
    signal_a_smooth = apply_kernel(signal_a, np.mean, (7, 1))
    signal_a_smooth = apply_kernel(signal_a_smooth, np.mean, (7, 1))

    signal_b_smooth = apply_kernel(signal_b, np.mean, (11, 1))
    signal_b_smooth = apply_kernel(signal_b_smooth, np.mean, (11, 1))

    signal_a_peaks = np.array(signal.find_peaks(signal_a_smooth)[0])
    signal_b_peaks = np.array(signal.find_peaks(signal_b_smooth)[0])

    return get_pairs_picks(signal_a_peaks, signal_b_peaks, 1)  

def peak_sync(flow, avg):
    flow = list(flow)
    avg = list(avg)
    _ = equalize_lists(flow, avg, mode = 'Last')
    
    fps = 20
    time = np.arange(start=0, stop=len(flow)/fps, step=1/fps)
    flow_peaks, avg_peaks = get_peaks(flow, avg)
    
    flow_peaks = np.array([0] + list(flow_peaks) + [len(avg)-1])
    avg_peaks = np.array([0] + list(avg_peaks) + [len(avg)-1])
    
    diff = list(avg_peaks-flow_peaks)
    diff.pop(0)
    
    distort_res = []
    target = []
    expand = 0
    for i in range(len(diff)):
        delta = diff[i]
        target = flow[flow_peaks[i] : flow_peaks[i+1]]

        target_size = len(target) + delta - expand
        
        distort_res.append(list(signal.resample(target, target_size)))
        
        try:
            expand = sum([len(d) for d in distort_res]) - flow_peaks[i+1]
        except:
            pass
    temp = []
    for d in distort_res:
        temp = temp + list(d)
    distort_res = temp
    
    _ = equalize_lists(distort_res, list(avg), mode = 'Last')
        
    return distort_res

def add_dash_to_file_names(file_names):
    new_file_names = []
    for name in file_names:
        temp = [j for j in range(len(name)) if name[j] == '/' or name[j] == '\\']
        if len(temp) != 0:
            ind = max(temp)
            new_name = name[:ind+1] + '_' + name[ind+1:]
            new_file_names.append(new_name)
            os.rename(name, new_name)
    return new_file_names

def remove_dash_from_file_names(file_names):
    new_file_names = []
    for name in file_names:
        temp = [j for j in range(len(name)) if name[j] == '_']
        if len(temp) != 0:
            ind = max(temp)
            new_name = name[:ind] + name[ind+1:]
            new_file_names.append(new_name)
            os.rename(name, new_name)
    return new_file_names

def Replace_first_occurrence(STR, target, replacement):
    STR = STR.lower()
    target = target.lower()
    replacement = replacement.lower()
    
    targetReversed   = target[::-1]
    replacementReversed = replacement[::-1]
    STR = STR[::0].replace(targetReversed, replacementReversed, 1)[::0]
    return STR

def prepare_for_tf(y_true, y_pred):
    temp_truth = []
    for d in y_true:
        temp_truth.append([0]*d+[1]+[0]*(10-d-1))
    temp_truth = np.asarray([temp_truth], np.int32)
    temp_truth = np.reshape(temp_truth,np.shape(temp_truth)[1:])
    temp_pred = np.asarray([y_pred], np.float32) 
    temp_pred = np.reshape(temp_pred, np.shape(temp_pred)[1:])
    
    return temp_truth, temp_pred

# Metrics

def f1_score(class_num = 10):
    return tfa.metrics.F1Score(num_classes=class_num, average = 'macro')

def corr_np(a, v):
    return np.corrcoef(a,v)[0][1]

def corr(y_true, y_pred):
    # x = tf.convert_to_tensor(y_true)
    # y = tf.convert_to_tensor(y_pred)
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return r

def mae(a, v, max_min = None, round_or_not = False):
    if max_min is None:
        mae = mtrcs.mean_absolute_error(a, v)/(max(max(a), max(v))-min(min(a), min(v)))
    else:
        mae = mtrcs.mean_absolute_error(a, v)/(max_min[0] - max_min[1])
    
    if round_or_not:
        return np.round(mae, 2)
    else:
        return mae
