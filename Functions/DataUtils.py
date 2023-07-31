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

import Functions.GlobalUtils as g_utils
import Functions.VideoUtils as v_utils
import Functions.LocalDB as db
import DATA_INFO as data_info
import numpy as np
import os, shutil, cv2, Augmentor, random, math, secrets
import matplotlib.pyplot as plt
from PIL.PngImagePlugin import PngInfo
from PIL import Image
from sklearn.utils import shuffle
import tensorflow as tf


class File:
    def save(data, path, column_names=None, with_string_format = True):
        """
            save multiple lists to txt file:
            => data = zip(list_a, list_b, list_c)
        """
        data_list = list(data)
        file = open(path, 'w')

        if column_names is not None:
            for column in column_names:
                file.write('++{:10s}\t'.format(column))
            file.write('\n')

        for line in data_list:
            for column in line:
                if math.isnan(column):
                    file.write('$$nan\t\t')
                else:
                    if with_string_format:
                        file.write('$${:5.5f}\t'.format(float(column)))
                    else:
                        file.write('$${}\t\t'.format(float(column)))
            file.write('\n')
        file.close()
            
    def read(path):
        file = open(path, 'r')
        raw_data = file.readlines()
        column_names = None

        start_row = 0
        if raw_data[0][0] == '+':
            column_names = raw_data[0].split('++')
            for i, c in enumerate(column_names):
                column_names[i] = c.strip()
            column_names = column_names[1:]
            start_row = 1

        if len(raw_data) == 1 and column_names != None:
            data = {}
            for i in range(len(column_names)):
                data[column_names[i]] = []
            return data
                
        n_column = len(raw_data[start_row].split('$$')) - 1
        data_lists = []
        for i in range(n_column):
            data_lists.append([])

        for c in raw_data[start_row:]:
            line = c.split('$$')
            for i in range(n_column):
                if not math.isnan(float(line[i+1])):
                    data_lists[i].append(float(line[i+1]))

        data = {}
        if column_names != None:
            for i in range(n_column):
                data[column_names[i]] = data_lists[i]
        else:
            for i in range(n_column):
                data[i] = data_lists[i]

        return data

    def add_or_update(new_data, path, column_names=None):
        old_data = File.read(path)

        temp = list(zip(*new_data))
        _new_data = []
        for t in temp:
            _new_data.append(list(t))

        _old_data = list(old_data.values())

        if column_names is not None:
            
            _old_keys = list(old_data.keys())
            _new_keys = column_names

            n = len(_new_data)
            m = len(_old_data)

            _repeated = []
            for i in range(n):
                for j in range(m):
                    if _new_keys[i] == _old_keys[j]:
                        _repeated.append(_new_keys[i])
            
            for r in _repeated:
                _old_data[_old_keys.index(r)] = _new_data[_new_keys.index(r)]
                _new_data.pop(_new_keys.index(r))
                _new_keys.remove(r)
            
            keys = _old_keys + _new_keys
            data = _old_data + _new_data
            data = g_utils.pack(*data)
            File.save(data, path, keys)
        else:
            data = _old_data + _new_data
            data = g_utils.pack(*data)
            File.save(data, path, list(old_data.keys()) + ['None'] * len(_new_data))

    def save_dict(dict, path):
        data = g_utils.pack(*list(dict.values()))
        cols = list(dict.keys())
        File.save(data, path, cols)
    
    def add_or_update_dict(dict, path):
        data = g_utils.pack(*list(dict.values()))
        cols = list(dict.keys())
        File.add_or_update(data, path, cols)

class Spiro:
    def __init__(self, flow, paw, volume, time_span, time_size):
        self.flow = flow
        self.paw = paw
        self.volume = volume
        self.time_span = time_span
        self.time_size = time_size

    def read(path, fps = 20):
        if not os.path.isfile(path):
            raise Exception('data does not Exists! : \n' + path)
        else:
            data_file = open(path, "r")    
        
        data = data_file.readlines()

        flow = []
        paw = []
        volume = []
        for line in data:
            line = line.split(',')
            flow.append(float(line[0]))
            paw.append(float(line[1]))
            volume.append(float(line[2]))
        
        time_span = np.arange(start=0, stop=len(flow)/fps, step=1/fps)
        return Spiro(flow,paw,volume,time_span,len(flow))

class Labeled_data_info(db.data_object_class):
    def __init__(self, 
                 
                 total_count : int, 
                 train_total_count : int, 
                 test_total_count : int, 
                 val_total_count : int,
                 
                 train_classes_count : list, 
                 test_classes_count : list,
                 val_classes_count : list,
                 
                 classes_borders : list,
                 
                 flow_max : float,
                 flow_min : float,
                 
                 class_size : int,
                 excluded_videos_indexes : list,
                 test_videos_indexes : list,
                 total_test_data_percentage : float,
                 total_val_data_percentage : float):
        self.total_count = total_count
        self.train_total_count = train_total_count
        self.test_total_count = test_total_count
        self.val_total_count = val_total_count
        
        self.train_classes_count = train_classes_count
        self.test_classes_count = test_classes_count
        self.val_classes_count = val_classes_count
        
        self.classes_borders = classes_borders
        self.flow_max = flow_max
        self.flow_min = flow_min
        self.class_size = class_size
        self.excluded_videos_indexes = excluded_videos_indexes
        self.test_videos_indexes = test_videos_indexes
        
        self.total_test_data_percentage = total_test_data_percentage
        self.total_val_data_percentage = total_val_data_percentage

    def print(self):
        print('total frames count :', self.total_count)
        print('train frames count :' ,self.train_total_count)
        print('test frames count :' ,self.test_total_count)
        print('val frames count :' ,self.val_total_count)
        
        print('train classes count :' ,self.train_classes_count)
        print('test classes count :' ,self.test_classes_count)
        print('val classes count :' ,self.val_classes_count)
        
        print('Classification borders :' ,self.classes_borders)
        print('flow max :' ,self.flow_max)
        print('flow min :' ,self.flow_min)
        print('class size :' ,self.class_size)
        print('excluded videos indexes :' ,self.excluded_videos_indexes)
        print('test videos indexes :' ,self.test_videos_indexes)
        print('total test data percentage :' ,self.total_test_data_percentage)
        print('total val data percentage :' ,self.total_val_data_percentage)

class frame_cat(db.data_object_class):
    def __init__(self, nvid : int, nframe : int, nclass : int, cat : str, fold : int, id = None):
        self.nvid = nvid
        self.nframe = nframe
        self.nclass = nclass
        self.cat = cat
        self.fold = fold
        self.id = id
    def uid(self):
        return f'{self.nvid}_{self.nframe}'


def clear_folder_content(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def read(path,
         spiro_txt=data_info.Raw.FileNames.spiro_data_file,
         spiro_timer_txt=data_info.Raw.FileNames.spiro_timer,
         video_timer_txt=data_info.Raw.FileNames.vid_timer):

    # Reading Spirometry Data from .log file
    data = File.read(path + spiro_txt)
    fps = 20
    flow = data['flow']
    volume = data['volume']
    frames_number = len(flow)

    time_span = np.arange(start=0, stop=frames_number/fps, step=1/fps)
    spiro = Spiro(flow, None, volume, time_span, frames_number)

    # Reading Spirometry Timer Numbers from .txt file
    spiro_timer = None
    if os.path.isfile(path + spiro_timer_txt):
        file = open(path + spiro_timer_txt, 'r')
        Lines = file.readlines()
        spiro_timer = list(np.float32(Lines))

    # Reading Timer of Thermal Video from .txt file
    video_timer = None
    if os.path.isfile(path + video_timer_txt):
        file = open(path + video_timer_txt, 'r')
        Lines = file.readlines()
        video_timer = list(np.int32(Lines))

    return spiro, spiro_timer, None, video_timer

def plot(spiro, spiro_timer=None, video_avg=None, video_timer=None, name=None, save_path = None):
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(6)
    
    plt.subplot(3, 1, 1)
    plt.plot(spiro.time_span, spiro.flow, '#3891A6')
    # sin = utils.fit_sin(spiro.time_span,spiro.flow-np.mean(spiro.flow))
    # plt.plot(sin.X,sin.Y*10,'b', alpha = 0.5)
    if video_avg is not None:
        plt.plot(spiro.time_span, video_avg, 'r')
        # sin = utils.fit_sin(spiro.time_span,video_avg-np.mean(video_avg))
        # plt.plot(sin.X,sin.Y*10,'r', alpha = 0.5)
    if spiro_timer is not None:
        for line in spiro_timer:
            plt.axvline(x=line, color='g', alpha=0.5)
    if video_timer is not None:
        for line in video_timer:
            plt.axvline(x=line, color='r')
    plt.title('Flow')
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(spiro.time_span, spiro.paw, '#E3655B')
    plt.title('Paw')
    plt.grid()
    
    plt.subplot(3, 1, 3)
    plt.plot(spiro.time_span, spiro.volume ,'#4C5B5C')
    if spiro_timer is not None:
        for line in spiro_timer:
            plt.axvline(x=line, color='g', alpha=0.5)
    if video_timer is not None:
        for line in video_timer:
            plt.axvline(x=line, color='r')
    plt.title('Volume')
    plt.grid()

    plt.tight_layout()
    plt.gcf().canvas.manager.set_window_title(name)
    
    if save_path is not None:
        plt.savefig(save_path,dpi=300)
        
    plt.show()

def Trim(path, 
        txt_data_name = data_info.Raw.FileNames.spiro_data_file, 
        timer=data_info.Raw.FileNames.spiro_timer, 
        save_path=data_info.Trimmed.path, 
        save_name=data_info.Trimmed.FileNames.data_name, 
        fps=20):

    file = open(path + timer, 'r')
    lines = file.readlines()
    start_times = list(np.float32(lines))
    stop_times = list(np.float32(lines)+20)

    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    if not os.path.isfile(path + txt_data_name):
        raise Exception('data does not Exists! : \n' + path + txt_data_name)
    else:
        data_file = open(path + txt_data_name, "r")

    dt = 1/fps
    time = 0
    n_trims = len(start_times)
    current_trim = None
    data = data_file.readlines()

    flow_vol = []
    for line in data:
        line = line.split(',')
        flow_vol.append([float(line[0]), float(line[2])])
    flow_vol = np.array(flow_vol)
    # flow_vol[:, 0] = utils.normalize_zero_to_one(flow_vol[:, 0])
    # flow_vol[:, 1] = utils.normalize_zero_to_one(flow_vol[:, 1])
    flow_vol = np.round(flow_vol, 4)

    j = 0
    lendth = len(flow_vol)
    while(True):
        line = flow_vol[j]
        j = j + 1
        if j == lendth:
            break

        for i in range(n_trims):
            if (time >= start_times[i]) and (current_trim == None) and (time <= stop_times[i]):
                current_trim = i
                temp1 = []
                temp2 = []

            elif (time >= stop_times[i]) and current_trim == i:
                fileFullPath = g_utils.get_free_file_name(
                    save_path, save_name, '.txt')
                File.save(zip(temp1, temp2), fileFullPath, ['flow', 'volume'])
                print('\t\t', fileFullPath)
                current_trim = None

        if current_trim != None:
            temp1.append(line[0])
            temp2.append(line[1])

        time = time + dt
    
def sync_data_first(
        trim_index,
        sync_index,
        
        trim_path       =   data_info.Trimmed.path, 
        trim_video_name =   data_info.Trimmed.FileNames.video_name, 
        trim_txt_name   =   data_info.Trimmed.FileNames.data_name, 
        shifts_file_txt =   data_info.Trimmed.FileNames.shifts_file,
        
        sync_path       =   data_info.Synced.path_first_sync, 
        sync_vid_name   =   data_info.Synced.FileNames.video_name,
        sync_txt_name   =   data_info.Synced.FileNames.data_name,
        
        
        plot_or_not=False,
        normalize_to=None):

    data = File.read(trim_path+trim_txt_name.format(trim_index)+'.txt')

    if normalize_to != None:
        data['flow'] = g_utils.normalize(data['flow'],normalize_to[0],normalize_to[1])


    temp_flow, temp_volume, temp_average, temp_position_x, temp_position_y, temp_radius = g_utils.equalize_lists(data['flow'], data['volume'], data['average'],
                                                                                                               data['position_x'], data['position_y'], data['radius'])

    frames_number = len(temp_flow)
    fps = 20
    T = 1/fps
    time_span = np.arange(start=0, stop=frames_number/fps, step=1/fps)

    # sync_file = open(sync_path+sync_file_txt)
    # shifts = sync_file.readlines()
    # shift_for_index = shifts[index].split(',')
    # t1 = float(shift_for_index[1])
    # t2 = float(shift_for_index[2])
    shifts_file = File.read(trim_path+shifts_file_txt)
    t1 = shifts_file['vid_delay'][trim_index]
    t2 = shifts_file['flow_delay'][trim_index]


    n_t1 = int(t1/T)
    n_t2 = int(t2/T)

    
    positions_x_trimmed = data['position_x'][n_t1:]
    positions_y_trimmed = data['position_y'][n_t1:]
    radius_trimmed = data['radius'][n_t1:]
    vid_avg_trimmed = data['average'][n_t1:]

    flow_trimmed = data['flow'][n_t2:]
    vol_trimmed = data['volume'][n_t2:]

    n_vid_avg = len(vid_avg_trimmed)
    n_spiro = len(flow_trimmed)

    video = v_utils.VideoReader_cv2(trim_path + trim_video_name.format(trim_index)+'.mp4')

    if n_vid_avg > n_spiro:
        positions_x_trimmed = positions_x_trimmed[0:n_spiro]
        positions_y_trimmed = positions_y_trimmed[0:n_spiro]
        radius_trimmed = radius_trimmed[0:n_spiro]
        vid_avg_trimmed = vid_avg_trimmed[0:n_spiro]

        video.Trim(n_t1, n_spiro + n_t1 - 1,
                   sync_path, sync_vid_name.format(sync_index), False)

        time_span_trimmed = np.arange(start=0, stop=n_spiro/fps, step=1/fps)
        time_size = n_spiro
    else:
        flow_trimmed = flow_trimmed[0:n_vid_avg]
        vol_trimmed = vol_trimmed[0:n_vid_avg]

        video.Trim(n_t1, len(data['average']) - 1,
                   sync_path, sync_vid_name.format(sync_index), False)

        time_span_trimmed = np.arange(start=0, stop=n_vid_avg/fps, step=1/fps)
        time_size = n_vid_avg

    SPIRO = Spiro(flow_trimmed, None, vol_trimmed,
                  time_span_trimmed, time_size)

    filtered_vid_avg_trimmed = np.round(
        g_utils.median_filter(vid_avg_trimmed, 5), 5)

    if sync_txt_name is not None:

        save_data = g_utils.pack(g_utils.median_filter(flow_trimmed,15), vol_trimmed,
                               positions_x_trimmed, positions_y_trimmed,radius_trimmed,vid_avg_trimmed)
        File.save(save_data, sync_path + sync_txt_name.format(sync_index) + ".txt",
                  ['flow', 'volume', 'position_x', 'position_y', 'radius','average'])




    if plot_or_not:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.title('synced')
        plt.plot(time_span_trimmed, vid_avg_trimmed, 'r', alpha=0.2)
        plt.plot(time_span_trimmed, filtered_vid_avg_trimmed, 'r',
                 time_span_trimmed, flow_trimmed, 'b')

        plt.grid()

        plt.subplot(2, 1, 2)
        plt.title('default t1 = {}, t2 = {}'.format(t1, t2))
        plt.plot(time_span, temp_average, 'r', alpha=0.5)
        plt.plot(time_span, temp_flow, 'b', alpha=0.5)
        plt.axvline(x=t1, color='r')
        plt.axvline(x=t2, color='b')
        plt.grid()

        plt.tight_layout()
        plt.gcf().canvas.manager.set_window_title(trim_txt_name.format(trim_index) + '->' + sync_txt_name.format(sync_index))
        plt.show()
        # plt.savefig('data{:02d}.png'.format(index))

    return vid_avg_trimmed, SPIRO

def get_video_roi_max_min(root = data_info.Trimmed, exclude_list = []):
    vid_list = [i for i in list(range(root.count())) if i not in exclude_list]
    
    avgs_list = []
    for i, index in enumerate(vid_list):
        video = v_utils.VideoReader_cv2(root.path + root.FileNames.video_name.format(index) + '.mp4')
        data = File.read(root.path + root.FileNames.data_name.format(index) + '.txt')
        positions = list(zip(np.int32(data['position_y']),np.int32(data['position_x'])))
        avg = []
        j = 0
        while True:
            ret ,frame = video.Read()
            if not ret:
                break
            temp ,cut = v_utils.getAvgOnCircle(frame, positions[j], 8)
            j = j + 1
            avg.append(temp)
        video.cap.release()
        avgs_list.append(avg)
        
    _max = max(avgs_list[0])
    _min = min(avgs_list[0])
    if len(avgs_list) > 1:
        for i in range(1,len(avgs_list)):
            _max = max(_max, max(avgs_list[i]))
            _min = min(_min, min(avgs_list[i]))
    return _max, _min

def get_single_data_info(index = None
                         ,normalize_to_flow = None
                         ,normalize_to_avg = None
                         ,exclude_list = []
                         ,path = data_info.Trimmed.path
                         ,data_name = data_info.Trimmed.FileNames.data_name
                         ,video_name = data_info.Trimmed.FileNames.video_name):   
    if index == None:
        _full_txt_name = path + data_name + '.txt'
        _full_vid_name = path + video_name + '.mp4'
    else:
        _full_txt_name = path + data_name.format(index) + '.txt'
        _full_vid_name = path + video_name.format(index) + '.mp4'
        
    data = File.read(_full_txt_name)
    video = v_utils.VideoReader_cv2(_full_vid_name)
    
    if normalize_to_flow != None:
        data['flow'] = g_utils.normalize(data['flow'],normalize_to_flow[0],normalize_to_flow[1])
    positions = list(zip(np.int32(data['position_y']),np.int32(data['position_x'])))
    
    avg = []
    _min = min(len(positions), video.length)
    for j in range(_min):
        ret ,frame = video.Read()
        if not ret:
            break
        temp ,cut = v_utils.getAvgOnCircle(frame, positions[j], 8)
        avg.append(temp)
    video.cap.release()
    
    if normalize_to_avg != None:
        avg = g_utils.normalize(avg,normalize_to_avg[0],normalize_to_avg[1])
    
    return list(avg), list(data['flow'])

def get_data_max_min(folder_path, dict_key, file_name = '{:02d}data.txt', exclude_list = []):
    n_files = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name)) and name[2:]==file_name[6:]])
    if n_files > 0:
        _max, _min = None, None
        for i in range(n_files):
            if i not in exclude_list:
                data = File.read(folder_path + file_name.format(i))
                if _max == None and _min == None:
                    _max = np.max(data[dict_key])
                    _min = np.min(data[dict_key])
                else:
                    _max = max(np.max(data[dict_key])    ,_max)
                    _min = min(np.min(data[dict_key])    ,_min)
        return _max, _min
    else:
        return 0, 0     

def Read_ROI(folder_name, class_num = 10, RGB = False, return_raw_flow = False, images_format = 'img{:04d}.png'):
    path = data_info.Labeled.path + folder_name + '/'
    images = []
    number_of_images = 0
    for i in range(class_num):
        _path = path + str(i)
        number_of_images = number_of_images + len([name for name in os.listdir(_path) if os.path.isfile(os.path.join(_path, name))])

    statusBar = g_utils.ProgressBar(number_of_images-1, print_with_details=True)
    shape = []
    flow = []
    categorized_flow = None

    for i in range(class_num):
        index = 0
        _len = len([name for name in os.listdir(path + str(i) + '/')])
        
        cat = np.zeros((_len,class_num))
        cat[:,i]=1

        while index < _len:
            filename = path + str(i)+ '/' + images_format.format(index)
            image = Image.open(filename)

            if shape == []:
                shape = np.shape(image)
            else:
                if np.shape(image) != shape:
                    index = index + 1
                    statusBar.update()
                    cat = np.remove(cat,1,0)
                    continue

            if return_raw_flow:
                text = image.text
                flow.append(float(text['flow']))


            images.append(np.array(image))
                
            index = index + 1
            statusBar.update()
        if not return_raw_flow:
            if categorized_flow is None:
                categorized_flow = cat
            else:
                categorized_flow = np.vstack((categorized_flow,cat))
                
    print('\nNumber of All images : {}'.format(number_of_images))

    if not return_raw_flow:
        return shuffle(images, categorized_flow)
    else:
        return shuffle(images, flow)

def get_class(target, n_class, max, min):
    range = max - min
    class_range = range / n_class
    return int(target/class_range)

def get_class_custom_borders(target, borders):
    _class = None
    for j in range(len(borders) - 1):
        if target > borders[j] and target <= borders[j + 1]:
            _class = j
    
            
    if _class == None:
        raise(Exception(
            'error finding class'))
    return _class

def Augment(path, n):
    p = Augmentor.Pipeline(path,"",)
    p.gaussian_distortion(probability=0.5,grid_height=2,grid_width=2,magnitude=2,corner='bell',method='sdy')
    p.random_distortion(grid_height=10, grid_width=10, magnitude=2, probability=1)
    p.random_erasing(probability=0.5,rectangle_area=0.4)
    p.rotate(probability=0.5,max_left_rotation=20,max_right_rotation=20)
    p.flip_random(probability=0.5)
    p.sample(n)

def Equalize_Classes(folder_name, class_num=10, Aug_or_Cut = True, Aug_ratio = None):
    count = []
    root = data_info.Labeled.path + folder_name + '/'
    for i in range(class_num):
        _path = root + "/{:d}/".format(i)
        count.append(len([name for name in os.listdir(_path) if os.path.isfile(os.path.join(_path, name))]))
    _max = max(count)
    _min = min(count)

    if Aug_or_Cut:
        if Aug_ratio == None:
            for i in range(class_num):
                if _max-count[i] != 0:
                    _path = root + "{:d}/".format(i)
                    Augment(_path,_max - count[i])
        else:
            _max = int(Aug_ratio * _max)
            for i in range(class_num):
                _path = root + "{:d}/".format(i)
                if _max-count[i] > 0:
                    Augment(_path,_max - count[i])
                    
                elif _max-count[i] < 0:
                    files = os.listdir(_path)
                    delete_num = count[i] - _max
                    for i in range(delete_num):
                        rnd_file = random.choice(files)
                        os.remove(_path + rnd_file)
                        files.remove(rnd_file)

    else:
        for i in range(class_num):
            _path = root + "{:d}/".format(i)
            files = os.listdir(_path)
            delete_num = count[i] - _min
            for i in range(delete_num):
                rnd_file = random.choice(files)
                os.remove(_path + rnd_file)
                files.remove(rnd_file)

def Numerize_Images_names(folder_name, class_num=10, prefix = '', suffix = '.png', dataset_or_folder = 'dataset'):
    if dataset_or_folder == 'dataset':
        for i in range(class_num):
            _path = data_info.Labeled.path + folder_name + "/{}/".format(str(i))
            
            files = os.listdir(_path)
            for j,filename in enumerate(files):
                _from = _path+filename
                _to = _path+'{}img{:04d}{}'.format(prefix,j, suffix)
                os.rename(_from,_to)
    else:
        files = os.listdir(folder_name)
        for j,filename in enumerate(files):
            _from = folder_name + '/' + filename
            _to = folder_name + '/{}img{:04d}{}'.format(prefix,j, suffix)
            os.rename(_from,_to)
    
def update_K_Fold_Dataset(folder_name, n):
    db_path = f'{folder_name}/'
    data_info = Labeled_data_info.cast(db.get_db(db_path + 'data_info.txt', Labeled_data_info)[0])
    data = db.get_db(db_path + 'frames_info.txt', v_utils.frame_info)
    k = data[0].k_fold ; class_num = len(data_info.test_classes_count)
    if n >= k: raise Exception('n must be smaller than k!')
    for mode in ['test', 'train']:
        shutil.rmtree(db_path + mode)
        os.mkdir(db_path + mode)
        for i in range(class_num):
            os.mkdir(f'{db_path}{mode}/{i}/')
    train_classes_count = [0]*class_num
    test_classes_count = [0]*class_num
    pBar = g_utils.ProgressBar(len(data) + 1, 0, True)
    
    for d in data:        
        if d.n_k_fold == n:
            shutil.copyfile(d.get_full_path(), f'{folder_name}/test/{d.class_number}/{d.image_name}')
            d.val_or_train = True
            test_classes_count[d.class_number] = test_classes_count[d.class_number] + 1
        else:
            shutil.copyfile(d.get_full_path(), f'{folder_name}/train/{d.class_number}/{d.image_name}')
            d.val_or_train = False
            train_classes_count[d.class_number] = train_classes_count[d.class_number] + 1
        
        pBar.update()
        
    db.create(db_path + 'frames_info.txt', v_utils.frame_info, True)
    db.insertOrUpdate(db_path + 'frames_info.txt', data)
    
    total_test_data_percentage = round(100*sum(test_classes_count)/len(data),2)
    total_val_data_percentage = round(100*data_info.val_total_count/len(data),2)
    
    data_info.test_classes_count = test_classes_count
    
    data_info.test_total_count = sum(test_classes_count)
    data_info.train_classes_count = train_classes_count
    data_info.train_total_count = sum(train_classes_count)
    
    data_info.total_test_data_percentage = total_test_data_percentage
    data_info.total_val_data_percentage = total_val_data_percentage
    
    db.create(db_path + 'data_info.txt', Labeled_data_info, True)
    db.insert(db_path + 'data_info.txt', data_info)
    
    pBar.update()

def pre_proccess_data():
    pass

# Data Labeling Functions

def Data_Labeling(frame_tale, class_num=10, folder_name='{:d}ROI', test_data_ratio=0.3, select_test_data_manual=False,  
                      test_data_vids_indexes=[], exclude_list=[], data_folder=data_info.Synced, equal_test_classes=False, 
                      train_video_indexes=[], resize_images = False):
       
    save_path = data_info.Labeled.path + folder_name.format(frame_tale) + '/'
    
    # Making Folders
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    clear_folder_content(save_path)
    os.mkdir(save_path + 'train')
    os.mkdir(save_path + 'test')
    for i in range(class_num):
        _path = save_path + 'train/' + str(i)
        if not os.path.isdir(_path):
            os.mkdir(_path)
            clear_folder_content(_path)
        
        _path = save_path + 'test/' + str(i)
        if not os.path.isdir(_path):
            os.mkdir(_path)
            clear_folder_content(_path)

    # Check if Train Videos are Selected Manually
    if train_video_indexes != []:
        for i in range(data_folder.count()):
            if (i not in train_video_indexes) and (i not in test_data_vids_indexes):
                exclude_list.append(i)

    # Calculating Number of All Data
    all_frames, i = 0, 0
    for i in range(data_folder.count()):
        if not i in exclude_list:
            vidname_read = data_folder.FileNames.video_name.format(i) + '.mp4'
            video = v_utils.VideoReader_cv2(data_folder.path + vidname_read)
            all_frames = all_frames + video.length - frame_tale + 1
            i = i + 1
    
    # Calculating Classes
    class_size = int(all_frames/class_num)
    flow_list = []
    for i in range(data_folder.count()):
        if not i in exclude_list:
            data = File.read(data_folder.path  + data_folder.FileNames.data_name.format(i) + '.txt')
            flow_list = flow_list + data['flow']

    flow_list.sort()
    class_borders = [-1]
    for i in range(1,class_num):
        class_borders.append((flow_list[i*class_size-1]+flow_list[i*class_size])/2)
    class_borders.append(max(flow_list))

    # Selecting Test Data
    if not select_test_data_manual:
        test_vid_count = int((data_folder.count() - len(exclude_list)) * test_data_ratio)
        if test_vid_count == 0:
            test_vid_count = 1
        test_data_vids_indexes = random.sample([i for i in list(range(data_folder.count())) if i not in exclude_list] , test_vid_count)
    
    # Count Test Data
    if equal_test_classes:
        original_test_classes_count = [0]*class_num
        for i in test_data_vids_indexes:
            data = File.read(data_folder.path  + data_folder.FileNames.data_name.format(i) + '.txt')
            for i in range(len(data['flow']) - 1):
                if i >= frame_tale:
                    _class = get_class_custom_borders(data['flow'][i-1], class_borders)
                    original_test_classes_count[_class] = original_test_classes_count[_class] + 1
        
        original_test_classes_count_max = max(original_test_classes_count)
        test_classes_fill_gap_count = [(original_test_classes_count_max-c) for c in original_test_classes_count]
    
    # Initializing Variables
    errors = []
    pBar = g_utils.ProgressBar(all_frames,1, True)
    i = 0
    count_train = 0
    count_test = 0
    
    train_classes_count = [0]*class_num
    test_classes_count = [0]*class_num
    
    frames_info = []
    
    for i in range(data_folder.count()):
        if not i in exclude_list:
            if i == 9:
                vv=0
            data = File.read(data_folder.path  + data_folder.FileNames.data_name.format(i) + '.txt')
            positions = list(zip(np.int32(data['position_y']), np.int32(data['position_x'])))
            
            vidname_read = data_folder.FileNames.video_name.format(i) + '.mp4'
            video = v_utils.VideoReader_cv2(data_folder.path + vidname_read)
            frame_list = []

            j = 0
            while True:
                ret, frame = video.Read()
                if not ret:
                    break
                margin = [10]*4
                result = v_utils.getSquareROI(frame, positions[j], margin)
                if j == 0 or np.shape(frame_list[-1]) == np.shape(result):
                    frame_list.append(result)
                else:
                    all_frames = all_frames - 1
                    pBar.end = pBar.end - 1
                j = j + 1
            video.done()
            cv2.destroyAllWindows()

            for j in range(len(frame_list)):
                if j >= frame_tale-1 :
                    try:
                        _concat = cv2.hconcat(frame_list[j-frame_tale+1:j+1])
                        _flow = data['flow'][j]

                        im = Image.fromarray(_concat)
                        
                        if resize_images:
                            im = im.resize(resize_images)
                            
                        _class = get_class_custom_borders(_flow, class_borders)
                        
                        val_or_train_data = i in test_data_vids_indexes
                            
                                
                        if equal_test_classes and (val_or_train_data == False):
                            if test_classes_fill_gap_count[_class] > 0:
                                val_or_train_data = True
                                test_classes_fill_gap_count[_class] = test_classes_fill_gap_count[_class] - 1
                                
                        
                        if val_or_train_data:
                            img_name= g_utils.get_free_file_name(save_path + 'test/' + str(_class) + '/', '_img{:02d}','.png')
                            test_classes_count[_class] = test_classes_count[_class] + 1
                            count_train = count_train + 1
                        else:
                            img_name= g_utils.get_free_file_name(save_path + 'train/' + str(_class) + '/', '_img{:02d}','.png')
                            train_classes_count[_class] = train_classes_count[_class] + 1
                            count_test = count_test + 1

                        frames_info.append(v_utils.frame_info(i, j, img_name, _class))
                        im.save(img_name)
                        pBar.update()
                        
                    except Exception as e:
                        errors.append('\nvid{}@{} error : {}'.format(i,j,e))
            
            i = i + 1

    # Backup Data
    os.mkdir(save_path + 'backup')
    shutil.copytree(save_path + 'train', save_path + 'backup/train')
    shutil.copytree(save_path + 'test', save_path + 'backup/test')
    
    # Save Data Info
    total_test_data_percentage = round( 100 * ( sum(test_classes_count) / ( sum( test_classes_count ) + sum( train_classes_count ) ) ),2)
    _info = Labeled_data_info(
    total_count = all_frames
    ,train_total_count = count_train
    ,test_total_count = count_test
    ,train_classes_count = train_classes_count
    ,test_classes_count = test_classes_count
    ,classes_borders = class_borders
    ,flow_max = max(flow_list)
    ,flow_min = min(flow_list)
    ,class_size = class_size
    ,excluded_videos_indexes = exclude_list
    ,test_videos_indexes = test_data_vids_indexes
    ,total_test_data_percentage=total_test_data_percentage)
    
    db.insert(save_path + 'data_info.txt', _info)
    db.insert(save_path + 'frames_info.txt', frames_info)
    
    
    
    
    
    

    for e in errors:
        print(e)

def Data_Labeling_Shuffle_Frames(class_num=10, folder_name='{:d}ROI', val_data_ratio=0.3, exclude_list=[],
                               data_folder=data_info.Synced, resize_images = False, k_fold = -1, test_videos = [], crop_size = [10, 10, 10, 10]):
       
    save_path = data_info.Labeled.path + folder_name.format(1) + '/'
    # Making Folders
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    clear_folder_content(save_path)
    os.mkdir(save_path + 'train')
    os.mkdir(save_path + 'test')
    os.mkdir(save_path + 'val')
    for i in range(class_num):
        for mode in ['train', 'test', 'val']:
            _path = f'{save_path}{mode}/{i}'
            if not os.path.isdir(_path):
                os.mkdir(_path)
                clear_folder_content(_path)              
    
    # Getting number of all data
    all_frames_count = 0
    for i in range(data_folder.count()):
        if not i in exclude_list:
            all_frames_count = all_frames_count + len(File.read(data_folder.path  + data_folder.FileNames.data_name.format(i) + '.txt')['flow'])
    
    # Calculating Classes
    class_size = int(all_frames_count/class_num)
    flow_list = []
    for i in range(data_folder.count()):
        if not i in exclude_list:
            data = File.read(data_folder.path  + data_folder.FileNames.data_name.format(i) + '.txt')
            flow_list = flow_list + data['flow']
    flow_list.sort()
    class_borders = [-1]
    # if ddl_labling:
    for i in range(1,class_num):
        class_borders.append((flow_list[i*class_size-1]+flow_list[i*class_size])/2)
    class_borders.append(max(flow_list))
    # else:
        # flow_list.sort()
        # class_borders = [-1]
        # step = (max(flow_list) - min(flow_list)) / class_num
        # class_borders = class_borders + list(np.arange(min(flow_list), max(flow_list) + step, step))[1:]

    # Frames Dict
    frames_cat = []
    frames_cat_dict = []
    for i in range(data_folder.count()):
        frames_cat_dict.append({})
    for i in range(data_folder.count()):
        if not i in exclude_list:
            flow_list = File.read(data_folder.path  + data_folder.FileNames.data_name.format(i) + '.txt')['flow']
            video_len = v_utils.VideoReader_cv2(data_folder.path + data_folder.FileNames.video_name.format(i) + '.mp4').length
            for j in range(video_len):
                _class = get_class_custom_borders(flow_list[j], class_borders)
                cat = 'test' if i in test_videos else 'train'
                fold_state = -1 if cat == 'test' else -2
                frames_cat.append(frame_cat(nvid=i, nframe=j, nclass=_class, cat=cat, fold = fold_state))
                frames_cat_dict[i][j] = frames_cat[-1]
                
                
    # Selecting validation data
    train_frames = [d for d in frames_cat if d.cat == 'train']
    val_data_classes_count = [int(len([f for f in train_frames if f.nclass==c])/k_fold) for c in range(class_num)]
    val_frame_indexes = []
    for i in range(class_num):
        class_data_list = [d for d in train_frames if d.nclass == i]
        
        permuted_indices = np.random.permutation(len(class_data_list))
        for f in range(k_fold):
            if f != k_fold - 1:
                val_indices = permuted_indices[f * val_data_classes_count[i]: (f+1) * val_data_classes_count[i]]
            else:
                val_indices = permuted_indices[f * val_data_classes_count[i]:]
                
            for v in val_indices:
                class_data_list[v].fold = f
                if f == 0:
                    class_data_list[v].cat = 'val'
                    val_frame_indexes.append(class_data_list[v].uid())
                
                frames_cat_dict[class_data_list[v].nvid][class_data_list[v].nframe] = class_data_list[v]
                    
                    
                    
        # selected = random.sample(class_data_list, val_data_classes_count[i])
        # for s in selected:
        #     s.cat = 'val'
        #     val_frame_indexes.append(s.uid())
        #     for f in frames_cat:
        #         if f.uid() == val_frame_indexes[-1]:
        #             f.cat = 'val'
        
    
    db.create(save_path + 'frames_cat.txt', frame_cat, True)
    db.insert(save_path + 'frames_cat.txt', frames_cat)
    
    
    # Initializing Variables
    train_classes_count, val_classes_count, test_classes_count = [0]*class_num, [0]*class_num, [0]*class_num
    pBar = g_utils.ProgressBar(all_frames_count + 2,1, True)
    frames_info = []
    errors = []
    os.mkdir(save_path + 'backup')
    for i in range(class_num):
        os.mkdir(save_path + f'backup/{i}')
    for i in range(data_folder.count()):
        if not i in exclude_list:
            data = File.read(data_folder.path  + data_folder.FileNames.data_name.format(i) + '.txt')
            positions = list(zip(np.int32(data['position_y']), np.int32(data['position_x'])))
            
            vidname_read = data_folder.FileNames.video_name.format(i) + '.mp4'
            video = v_utils.VideoReader_cv2(data_folder.path + vidname_read)
            frame_list = []

            j = 0
            while True:
                ret, frame = video.Read()
                if not ret:
                    break
                result = v_utils.getSquareROI(frame, positions[j], crop_size)
                if j == 0 or np.shape(frame_list[-1]) == np.shape(result):
                    frame_list.append(result)
                else:
                    all_frames_count = all_frames_count - 1
                    pBar.end = pBar.end - 1
                j = j + 1
            video.done()
            cv2.destroyAllWindows()
            for j in range(len(frame_list)):
                try:
                    im = Image.fromarray(frame_list[j])
                    flow = data['flow'][j]
                    if resize_images : im = im.resize(resize_images)
                    frame_class = get_class_custom_borders(flow, class_borders)

                    test_or_val = [False, False];
                    
                    test_or_val[0] = i in test_videos
                    test_or_val[1] = bool(f'{i}_{j}' in val_frame_indexes)
                    

                    if test_or_val[0] and test_or_val[1]:
                        raise Exception('found a frame belonging to test and validation!')

                    img_name = secrets.token_hex(nbytes=16)
                    if i in test_videos:
                        k_fold_state = -1
                    else:
                        k_fold_state = frames_cat_dict[i][j].fold
                    
                    if test_or_val[0]:
                        test_classes_count[frame_class] = test_classes_count[frame_class] + 1
                        mode = 'test'
                    elif test_or_val[1]:
                        val_classes_count[frame_class] = val_classes_count[frame_class] + 1
                        mode = 'val'
                    else:
                        train_classes_count[frame_class] = train_classes_count[frame_class] + 1
                        mode = 'train'
                        
                    img_path = os.path.normpath(f'{save_path}/{mode}/{frame_class}/{img_name}.png')
                    backup_img_path = os.path.normpath(f'{save_path}/backup/')

                    frames_info.append(v_utils.frame_info(i, j, flow, f'{img_name}.png', backup_img_path, frame_class, mode, k_fold_state, k_fold))
                    im.save(img_path)
                    im.save(f'{backup_img_path}/{frame_class}/{img_name}.png')
                    pBar.update()
                    
                except Exception as e:
                    errors.append('\nvid{}@{} error : {}'.format(i,j,e))
              
    
    # K-Fold Labling
    # if k_fold != None:
    #     frames_info_val = []
    #     for frame in [f for f in frames_info if f.mode == 'val']:
    #         frame.n_k_fold = 0
    #         frames_info_val.append(frame)
    #         frames_info.remove(frame)
                       
    #     frames_info = shuffle(frames_info)
    #     for c in range(class_num):
    #         target_frames = [f for f in frames_info if f.class_number == c and f.n_k_fold != -1]
    #         fold_size = int(len(target_frames)/(k_fold-1))
    #         # fold_size = val_data_classes_count[c]
    #         for k in range(0, k_fold-1):
    #             if k != k_fold - 2:
    #                 temp = target_frames[k*fold_size : (k+1)*fold_size]
    #             else:
    #                 temp = target_frames[k*fold_size :]
                    
    #             for f in temp:
    #                 f.n_k_fold = k + 1
          
    #     frames_info = frames_info_val + frames_info
    
    db.insert(save_path + 'frames_info.txt', frames_info)
    pBar.update()
    
    # Save Data Info
    total_test_data_percentage = round( 100 * ( sum(test_classes_count) / ( sum( test_classes_count ) + sum( val_classes_count ) + sum( train_classes_count ) ) ),2)
    total_val_data_percentage = round( 100 * ( sum(val_classes_count) / ( sum( test_classes_count ) + sum( val_classes_count ) + sum( train_classes_count )) ),2)
    _info = Labeled_data_info(
    total_count = sum(train_classes_count) + sum(test_classes_count) + sum(val_classes_count)
    ,train_total_count = sum(train_classes_count)
    ,test_total_count = sum(test_classes_count)
    ,val_total_count = sum(val_classes_count)
    
    ,train_classes_count = train_classes_count
    ,test_classes_count = test_classes_count
    ,val_classes_count = val_classes_count
    
    ,classes_borders = class_borders
    ,flow_max = max(flow_list)
    ,flow_min = min(flow_list)
    ,class_size = class_size
    ,excluded_videos_indexes = exclude_list
    ,test_videos_indexes = test_videos
    
    ,total_test_data_percentage = total_test_data_percentage
    ,total_val_data_percentage = total_val_data_percentage)
    
    db.insert(save_path + 'data_info.txt', _info)
    
    pBar.update()
    
    for e in errors:
        print(e)

def single_vid_Data_Labeling(vid_index, class_num = 10, data_root = data_info.Synced, borders = [], gray_scale = True, resize_image = None):
    
    # Calculating Number of All Data
    vid_name = data_root.FileNames.video_name.format(vid_index) + '.mp4'
    video = v_utils.VideoReader_cv2(data_root.path + vid_name)
    
    # Initializing Variables
    errors = []
    pBar = g_utils.ProgressBar(video.length,1, True)
    i = 0
    
    data = File.read(data_root.path  + data_root.FileNames.data_name.format(vid_index) + '.txt')
    positions = list(zip(np.int32(data['position_y']), np.int32(data['position_x'])))
    frame_list = []

    j = 0
    while True:
        ret, frame = video.Read(gray_scale=gray_scale)
        if not ret:
            break
        
        margin = [10]*4
        result = v_utils.getSquareROI(frame, positions[j], margin)
        
        frame_list.append(result)
        j = j + 1
    video.done()

    data_X = []
    data_Y = []
    real_flow = []

    for j in range(len(frame_list)):
        try:
            
                    
            im = Image.fromarray(frame_list[j])
            if resize_image : im = im.resize(resize_image)
            if not gray_scale : im = im.convert('RGB')
            data_X.append(tf.keras.preprocessing.image.img_to_array(im))
            _flow = data['flow'][j]
            real_flow.append(_flow)

            if borders == []:
                _class = get_class(_flow,class_num,1,0)
                if _class == class_num:
                    _class = class_num - 1
            else:
                _class = get_class_custom_borders(_flow, borders)
                if _class == None:
                    a=1
            data_Y.append(_class)
            
            pBar.update()
        except Exception as e:
            errors.append('\nvid{}@{} error : {}'.format(i,j,e))
    for e in errors:
        print(e)
    return np.array(data_X)/255, data_Y, real_flow

