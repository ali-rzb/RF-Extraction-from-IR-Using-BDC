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

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorflow_addons as tfa
from keras.layers import Activation, Dense, Flatten, Conv2D, Dropout, MaxPooling2D
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras import Sequential
from tensorflow import keras
from keras.utils import np_utils
import Functions.GlobalUtils as g_utils
import Functions.DataUtils as d_utils
import Functions.VideoUtils as v_utils
import Functions.LocalDB as db
import DATA_INFO as data_info
import visualkeras
import matplotlib.pyplot as plt
import os, shutil, copy, re, cv2, glob
import PIL.Image as Image
import numpy as np
import pandas as pd
import random as rnd
import matplotlib.cm as cm
from keras import metrics
from sklearn.utils import class_weight

try:
    device_name = os.environ["COLAB_TPU_ADDR"]
    TPU_ADDRESS = "grpc://" + device_name
    print("Found TPU at: {}".format(TPU_ADDRESS))
except:
    print("TPU not found")
    try:
        device = tf.config.list_physical_devices('GPU')[0]
        tf.config.experimental.set_memory_growth(
            device, True)
        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(log_device_placement=True))
        print(f"using {device}")
    except:
        print("GPU not found")
    


def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def get_gradcam(img, heatmap, alpha=0.4):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    if len(np.shape(img)) == 2 and len(np.shape(jet_heatmap)) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    return np.array(superimposed_img)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

class CNN:
    """_summary_
        Steps For Use : 
            Step 1 : Initialize Parameters 
                -> Inputs to the Class Constructor
                
            Step 2 : Checking Data if needed make it 
                -> Function prepare_data
                
            Step 3 : Making Model with these functions or manually setting self.model to a customized model or Load a Saved Model 
                -> Functions make_model, make_model_VGG16, load_model or manually making model in self.model
                
            Step 4 : Train self.Model With Custom Parameters -
                > Function train_model
                
            Optional : Saving Model 
                -> Function save_model
    """

    # Step 1 : Initialize Parameters
    def __init__(self, frame_tale, class_num, augment_ratio, batch_size, validation_ratio=0.2
                 ,early_stopping_patience=3, equal_test_classes=False, test_video_indexes=[1, 7, 13]
                 ,train_video_indexes=[], exclude_list=[], force_remake_date=False, reset_init_database=False
                 ,dataset_path=None, dataset_folder_name=None, resize_images = False):

        # Data Parameters
        self.augment_ratio = augment_ratio
        self.validation_ratio = validation_ratio
        self.equal_test_classes = equal_test_classes
        self.exclude_list = exclude_list
        self.frame_tale = frame_tale
        self.class_num = class_num
        self.test_video_indexes = test_video_indexes
        self.train_video_indexes = train_video_indexes
        self.resize_images = resize_images
        self.force_remake_date = force_remake_date
        self.reset_init_database = reset_init_database

        if (dataset_path == None) and (dataset_folder_name == None):
            self.dataset_folder_name = data_info.Labeled.FileNames.Folder_Name.format(
                self.frame_tale, self.class_num, int(validation_ratio*100), augment_ratio)
            self.dataset_path = data_info.Labeled.path + self.dataset_folder_name + '/'
        else:
            self.dataset_folder_name = dataset_folder_name
            self.dataset_path = dataset_path + self.dataset_folder_name

        if resize_images:
            self.image_size = self.resize_images
        else:
            self.image_size = (20, self.frame_tale*20)

        

        # CNN Parameters
        self.batch_size = batch_size
        self.early_stop_patience = early_stopping_patience
        self.es_callback = EarlyStopping(
            monitor='val_loss', patience=early_stopping_patience)

        self.history = self.hist(model_info='')



    # Step 2 : Checking Data if needed make it
    def prepare_data(self, color_mode = 'grayscale', shuffle_frames = False, n_k_fold = -1, k_fold = -1, ddl_labeling = True):
        self.ddl_labeling = ddl_labeling
        self.color_mode = color_mode
        self.k_fold = k_fold
        self.n_k_fold = n_k_fold
        
        if  not os.path.isfile(self.dataset_path + '/' + data_info.Labeled.FileNames.data_info_txt) or \
            not os.path.isdir(self.dataset_path) or \
            self.force_remake_date or \
            self.reset_init_database:
            self.__make_data(just_reset_database=self.reset_init_database, shuffle_frames = shuffle_frames, n_k_fold=n_k_fold, k_fold = k_fold, refold = False)
        elif n_k_fold != -1:
            temp = db.get_db(data_info.Labeled.path + self.dataset_folder_name + '/frames_info.txt', v_utils.frame_info)
            temp = list(set([d.n_k_fold for d in temp if d.mode == "val"]))
            if temp[0] != n_k_fold:
                self.__make_data(just_reset_database=self.reset_init_database, shuffle_frames = shuffle_frames, n_k_fold=n_k_fold, k_fold = k_fold, refold = True)
                
        self.image_shape = [75, 75]
        self.force_remake_date = False

    # Step 3 : Making Model with these functions or manually setting self.model to a customized model or Load a Saved Model
    def make_model(self, conv, fully, activation='relu', conv_kernel_sizes=[], conv_strides=[], pooling_kernels=[], pooling_strides=[], drop_out=0.5, dummy_spacing = False):
        self.conv = conv
        self.fully = fully

        model_info = ('{}.CONV' + len(conv) * '.{}' + '.FULLY' + len(fully)
                      * '.{}').format(self.dataset_folder_name, *conv, *fully)
        self.history = self.hist(model_info=model_info)

        self.model = Sequential()
        
        for i, c in enumerate(conv):
            if c != 0:
                kernel_size = (3, 3) if conv_kernel_sizes == [] else (
                    conv_kernel_sizes[i], conv_kernel_sizes[i])
                conv_stride = (1, 1) if conv_strides == [] else (
                    conv_strides[i], conv_strides[i])
                pool_kernel = (1, 1) if pooling_kernels == [] else (
                    pooling_kernels[i], pooling_kernels[i])
                pool_stride = (1, 1) if pooling_strides == [] else (
                    pooling_strides[i], pooling_strides[i])

                if i == 0:
                    self.model.add(Conv2D(c, kernel_size=kernel_size, strides=conv_stride,
                                   padding='same', input_shape=(self.image_shape)))
                else:
                    self.model.add(
                        Conv2D(c, kernel_size=kernel_size, strides=conv_stride, padding='same'))
                self.model.add(BatchNormalization())
                self.model.add(Activation(activation))
                self.model.add(MaxPooling2D(pool_size=pool_kernel,
                               strides=pool_stride, padding='same'))

                if drop_out != 0:
                    self.model.add(Dropout(drop_out))
                if dummy_spacing:
                    self.model.add(visualkeras.SpacingDummyLayer(spacing=dummy_spacing))
        self.model.add(Flatten())
        if dummy_spacing:
            self.model.add(visualkeras.SpacingDummyLayer(spacing=dummy_spacing))
        for f in fully:
            if f != 0:
                self.model.add(Dense(f, activation='relu'))
                self.model.add(BatchNormalization())
                if drop_out != 0:
                    self.model.add(Dropout(drop_out))
            if dummy_spacing:
                self.model.add(visualkeras.SpacingDummyLayer(spacing=dummy_spacing))
            
        self.model.add(Dense(self.class_num, activation='softmax'))

    def load_model(self, path):
        self.model = keras.models.load_model(path)

    def get_class_weights(self, data_suffix = '.png'):
        path = self.dataset_path + '/backup'
        classes = list(range(self.class_num))
        y_train = []
        for i in classes:
            class_count = len(glob.glob(f'{path}/{i}/*{data_suffix}'))
            y_train = y_train + [i]*class_count
        
        return class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

    # Step 4 : Train self.Model With Custom Parameters
    def train_model(self, epochs, save_model_path, learning_rate=None, verbose=True, optimizer=Adam, wheighted_loss = False):
        class_weights = self.get_class_weights() if wheighted_loss is True else None
        optimizer = (Adam(learning_rate=learning_rate) if learning_rate is not None else Adam()) if optimizer == Adam else optimizer
        
        self.train_sampler = self.sampler(self.dataset_path + 'train', self.batch_size, self.class_num)
        self.val_sampler = self.sampler(self.dataset_path + 'val', self.batch_size, self.class_num)
            
        score_name = 'f1_score'
        self.model.compile(loss='categorical_crossentropy', metrics=[g_utils.f1_score()], optimizer=optimizer)

        i = 0
        print('\n')
        try:
            while True:
                print('Epoch {}/{}'.format(i+1, epochs))
                
                hist = self.model.fit(self.train_sampler, batch_size=self.batch_size, epochs=1,validation_data=self.val_sampler, 
                                      verbose=verbose, shuffle=True, class_weight=class_weights)
                
                # self.history.accuracy = self.history.accuracy + hist.history['accuracy']
                # self.history.val_accuracy = self.history.val_accuracy + hist.history['val_accuracy']
                
                
                self.history.score = self.history.score + hist.history[score_name]
                self.history.val_score = self.history.val_score + hist.history[f'val_{score_name}']
                self.history.loss = self.history.loss + hist.history['loss']
                self.history.val_loss = self.history.val_loss + hist.history['val_loss']
                
                
                if i == 0 or self.history.val_score[-1] > max(self.history.val_score[:-1]):
                    self.save_model(save_model_path)
                    print('model saved!')
                if i >= self.early_stop_patience:
                    if max(self.history.val_score[:-self.early_stop_patience]) >= max(self.history.val_score[-self.early_stop_patience:]):
                        break
                if i >= epochs:
                    break
                i = i + 1
            
            if self.k_fold != -1:
                # self.val_ds = self.__get_image_data(data_path=self.dataset_path + 'val', image_size=self.image_size,
                #                       color_mode=self.color_mode, label_mode='categorical', seed=1, batch_size=self.batch_size)
                self.test_sampler = self.sampler(self.dataset_path + 'test', self.batch_size, self.class_num)
                results = self.model.evaluate(self.test_sampler, batch_size = self.batch_size)
                self.history.fold_loss = results[0]
                self.history.fold_score = results[1]
                
        except Exception as e:
            print(f'ERROR : {e}')


    # Optional : Saving Model
    def save_model(self, path, name=None):
        if name == None:
            name = self.history.model_info + '.h5'
        if os.path.isfile(path + '/' + name):
            os.remove(path + '/' + name)
        self.model.save(path + '/' + name)


    # Optional : Heatmap
    def get_heatmap(self, img, layer = 'LastConv'):
        
        if layer == 'LastConv':
            layer = self.get_conv_layer_names()[-1]
            
        last_conv_layer_name = layer
        # Remove last layer's softmax
        self.model.layers[-1].activation = None
        # Print what the top predicted class is
        image = [img]
        image = np.reshape(image, (np.shape(image)[0], np.shape(image)[1],np.shape(image)[2], 1))
        # Generate class activation heatmap
        heatmap = make_gradcam_heatmap(image, self.model, last_conv_layer_name)


        return get_gradcam(img, heatmap, alpha = 0.5)
    
    # Optional : get layer names 
    def get_conv_layer_names(self):
        return [l.input.name[:l.input.name.index('/')] for l in self.model.layers if '/' in l.input.name and 'conv2d' in l.input.name ]
        

    def __make_data(self, just_reset_database=False, shuffle_frames = False, n_k_fold = -1, k_fold = -1, refold = False):
        
        if not refold:
            if self.test_video_indexes != []:
                select_test_data_manual = True

            generate_new_data = True
            if just_reset_database:
                generate_new_data = False
                dir = os.listdir(data_info.Labeled.path)
                def func(n): return '.'.join(n.split('.')[:-1])
                old_datasets = [d for d in dir if func(d) == func(self.dataset_folder_name)]
                if len(old_datasets) != 0:
                    self.reset_database(data_info.Labeled.path + old_datasets[0], data_info.Labeled.path + self.dataset_folder_name)
                else:
                    generate_new_data = True

            if generate_new_data:
                if shuffle_frames:
                    d_utils.Data_Labeling_Shuffle_Frames(class_num=self.class_num, folder_name=self.dataset_folder_name + '/', data_folder=data_info.Synced,
                                                             resize_images=self.resize_images, k_fold=k_fold, test_videos = self.test_video_indexes)
                else:
                    d_utils.Data_Labeling(frame_tale=self.frame_tale, class_num=self.class_num, test_data_ratio=self.validation_ratio
                                                    ,folder_name=self.dataset_folder_name + '/' ,data_folder=data_info.Synced,select_test_data_manual=select_test_data_manual
                                                    ,test_data_vids_indexes=self.test_video_indexes, train_video_indexes=self.train_video_indexes, exclude_list=self.exclude_list
                                                    ,equal_test_classes=self.equal_test_classes, resize_images=self.resize_images)
            if n_k_fold != 0 and n_k_fold != -1:
                d_utils.update_K_Fold_Dataset(data_info.Labeled.path + self.dataset_folder_name, n_k_fold)    
        else:
            d_utils.update_K_Fold_Dataset(data_info.Labeled.path + self.dataset_folder_name, n_k_fold)
        
        if self.augment_ratio != 0:
            d_utils.Equalize_Classes(folder_name=self.dataset_folder_name +
                                    '/train', class_num=self.class_num, Aug_ratio=self.augment_ratio)
            
        d_utils.Numerize_Images_names(folder_name=self.dataset_folder_name + '/train', class_num=self.class_num)
        d_utils.Numerize_Images_names(folder_name=self.dataset_folder_name + '/test', class_num=self.class_num)

    def __get_image_data(self, data_path, color_mode, image_size, label_mode, batch_size, seed=None):
        raw_data_set = keras.preprocessing.image_dataset_from_directory(
            data_path,
            image_size=image_size,
            color_mode=color_mode,
            label_mode=label_mode,
            seed=seed,
            batch_size=batch_size)
        raw_data_set.class_names.sort()
        return {
            "data": raw_data_set.cache().prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE
            ),
            "classNames": raw_data_set.class_names
        }


    class hist(db.data_object_class):
        def __init__(self, model_info: str, 
                     score: list = [], val_score: list = [], fold_score: float = 0,
                     loss: list = [], val_loss: list = [], fold_loss : float = 0, 
                     id=None):
            
            self.score = score
            self.val_score = val_score
            self.fold_score = fold_score
            
            self.loss = loss
            self.val_loss = val_loss
            self.fold_loss = fold_loss
            
            self.id = id
            self.model_info = model_info
            
        
        def class_num(self): return int([s for s in self.model_info.split('.') if s[0] == 'C'][0][1:])
        def validation_ratio(self): return float([s for s in self.model_info.split('.') if s[0] == 'V'][0][1:])/100
        def augmentation_ratio(self): return int([s for s in self.model_info.split('.') if s[0] == 'A'][0][1:])
        def conv(self): return list(np.int64(self.model_info.split('CONV')[1].split('FULLY')[0][1:-1].split('.')))
        def fully(self): return list(np.int64(self.model_info.split('FULLY')[1][1:].split('.')))
        def max_score(self): return round(max(self.val_score), 2)
        def fold_num(self): 
            try:
                res = int([s for s in self.model_info.split('.') if s[0] == 'F'][0][1:])
            except:
                try:
                    res = int([s for s in self.model_info.split('_') if s[0] == 'F'][0][1:])
                except:
                    print(self.model_info)
            return res

    class prediction_data(db.data_object_class):
        def __init__(self, predicted: list, truth_class: list, truth_flow: list, vid_name:str, model_name:str, borders:list, class_num:int, raw_pred:list, f_max : float, f_min : float, id=None):
            self.raw_pred = raw_pred
            self.predicted = predicted
            self.class_num = class_num
            self.truth_class = truth_class
            self.truth_flow = truth_flow
            self.vid_name = vid_name
            self.model_name = model_name
            self.borders = borders
            self.f_max = f_max
            self.f_min = f_min
            self.id = id
    
    class sampler(tf.keras.utils.Sequence):
        def __init__(self, root, batch_size, n_class, data_suffix = '.png'):
            self.root = root
            self.batch_size = batch_size
            self.n_class = n_class

            self.per_class = int(batch_size / n_class)
            
            self.class_count = []
            
            self.data_list_per_class = []
            # self.data_list_per_class_IMAGE = []
            for i in range(n_class):
                data_list = glob.glob(f'{root}/{i}/*{data_suffix}')
                self.class_count.append(len(data_list))
                rnd.shuffle(data_list)
                self.data_list_per_class.append(data_list)
                # self.data_list_per_class_IMAGE.append([])
                # for path in data_list:
                #     image = tf.keras.preprocessing.image.load_img(path)
                #     image_arr = tf.keras.preprocessing.image.img_to_array(image)
                #     self.data_list_per_class_IMAGE[-1].append(image_arr)
            
            self.n = sum(self.class_count)
        
        def __getitem__(self, index):        
            Y = []
            X = []
            for i in range(self.n_class):
                a, b = index*self.per_class, (index+1)*self.per_class
                while a > self.class_count[i] or b > self.class_count[i]:
                    a = a - self.per_class
                    b = b - self.per_class
                    
                for path in self.data_list_per_class[i][a : b]:
                    image = tf.keras.preprocessing.image.load_img(path)
                    image_arr = tf.keras.preprocessing.image.img_to_array(image)
                    X.append(image_arr)
                Y = Y + [i]*self.per_class
                
            Y = tf.keras.utils.to_categorical(Y, self.n_class)
            # X = tf.stack(X)
            X = np.array(X)/255
            
            return X, Y
        
        def __len__(self):
            return self.n // self.batch_size

        def on_epoch_end(self):
            for i in range(self.n_class):
                rnd.shuffle(self.data_list_per_class[i])
       
    
    @staticmethod
    def plot_model_hist(_hist: hist, show=True, save_path=None, plot_loss = False):
        if show:
            plt.figure(figsize=(10, 3))
        if plot_loss:
            plt.subplot(2, 1, 1)
        plt.plot(_hist.score)
        plt.plot(_hist.val_score)
        plt.title('{}\nmodel accuracy'.format(_hist.model_info))
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.plot([_hist.val_score.index(max(_hist.val_score))], [max(_hist.val_score)], 'o', color='#e3655b')
        plt.xticks(range(len(_hist.score)))
        plt.grid()
        plt.legend(['train', 'val', 'max val'])
        
        if plot_loss:
            plt.subplot(2, 1, 2)
            plt.plot(_hist.loss)
            plt.plot(_hist.val_loss)
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.grid()
            plt.legend(['train', 'val'], loc='upper right')

        # plt.tight_layout()
        if save_path != None:
            plt.savefig(save_path, dpi=300)
        if show:
            plt.show()

    @staticmethod
    def plot_models_hist(data, to_excel_path = None):
        def get_key(d):
            if len(re.findall(r'T(\d+)\.C(\d+)\.V(\d+)\.A(\d+)\.CONV\.(.*)\.FULLY\.(.*)', d.model_info)) > 0:
                return '{:02d}'.format(d.frame_tale()) + str(d.conv()) + str(d.fully())
            elif len(re.findall(r'C(\d+)_(.*)', d.model_info)) > 0:
                return d.model_info[4:]
        def get_name(n):
            if len(re.findall(r'(\d+)\[(.*)\]\[(.*)\]', n)) > 0:
                conv = np.int64(n.split('][')[0].split('[')[1].split(','))
                fully = np.int64(n.split('][')[1][:-1].split(','))
                tale = int(n.split('[')[0])
                if len(set(conv)) == len(set(fully)) == 1:
                    return 'Depth_{}_Tale{}'.format(len(conv), tale)
                else:    
                    return 't : {:02d}, c : {}, f : {}'.format(
                        tale ,   
                        conv if len(set(conv)) != 1 else str([conv[0]])+f'*{len(conv)}',
                        fully if len(set(fully)) != 1 else str([fully[0]])+f'*{len(fully)}'
                )
            else :
                return n
        def get_class_num(d):
            if len(re.findall(r'T(\d+)\.C(\d+)\.V(\d+)\.A(\d+)\.CONV\.(.*)\.FULLY\.(.*)', d.model_info)) > 0:
                return d.class_num()
            elif len(re.findall(r'C(\d+)_(.*)', d.model_info)) > 0:
                res = re.findall(r'C(\d+)_(.*)', d.model_info)[0]
                if len(res) == 1:
                    return int(res)
                else:
                    return int(res[0])
            
        temp = []
        for d in data:
            temp.append(get_key(d))
        
        keys = list(set(temp))
        keys.sort(key=lambda x: 100*np.mean([t.max_score() for t in data if get_key(t) == x]), reverse=True)
                
        mean_column = {}
        for k in keys:
            mean_column[k] = int(100*round(np.mean([t.max_score() for t in data if get_key(t) == k]), 2))

        nets = {}
        for n in keys:
            nets[n] = get_name(n)
                
        names_max_len = max([len(v) for v in nets.values()])
        classes = sorted(list(set([get_class_num(g) for g in data])))

        print(' '*(names_max_len + 5), end='')
        [print('C{:02d}\t'.format(c), end='') for c in classes]
        print('mean')

        pd_dict = {}
        pd_dict['model_names'] = nets.values()
        for c in classes:
            pd_dict[c] = []
        pd_dict['mean'] = mean_column.values()
        
        for net in nets.keys():
            print(nets[net], end=' '*(names_max_len - len(nets[net]) + 5))
            trains = [t for t in data if get_key(t) == net]
            for c in classes:
                temp = [t for t in trains if get_class_num(t) == c]
                if len(temp) != 0:
                    num = int(100*max([t.max_score() for t in temp]))
                    print('{:02d}'.format(num), end="\t")
                    pd_dict[c].append(num)
                else:
                    print('\t', end="")
                    pd_dict[c].append(' ')
                    
            print('{:2d}'.format(mean_column[net]))

        if to_excel_path is not None:
            res = pd.DataFrame(pd_dict)
            path = to_excel_path if to_excel_path[-5:] == '.xlsx' else to_excel_path + '.xlsx'
            res.to_excel(path, sheet_name='sheet1', index=False)
    
    @staticmethod
    def plot_models_hist_folds(data, to_excel_path = None):
        def get_key(d):
            if len(re.findall(r'T(\d+)\.C(\d+)\.V(\d+)\.A(\d+)\.CONV\.(.*)\.FULLY\.(.*)', d.model_info)) > 0:
                return '{:02d}'.format(d.frame_tale()) + str(d.conv()) + str(d.fully())
            elif len(re.findall(r'C(\d+)_(.*)', d.model_info)) > 0:
                
                ind = [i for i in range(len(d.model_info)) if d.model_info[i]=='_']
                return d.model_info[ind[1]+1:]
        def get_name(n):
            if len(re.findall(r'(\d+)\[(.*)\]\[(.*)\]', n)) > 0:
                conv = np.int64(n.split('][')[0].split('[')[1].split(','))
                fully = np.int64(n.split('][')[1][:-1].split(','))
                tale = int(n.split('[')[0])
                if len(set(conv)) == len(set(fully)) == 1:
                    return 'Depth_{}_Tale{}'.format(len(conv), tale)
                else:    
                    return 't : {:02d}, c : {}, f : {}'.format(
                        tale ,   
                        conv if len(set(conv)) != 1 else str([conv[0]])+f'*{len(conv)}',
                        fully if len(set(fully)) != 1 else str([fully[0]])+f'*{len(fully)}'
                )
            else :
                return n
        def get_fold_num(d):
            return d.fold_num()

        for i in range(len(data)):
            if data[i].fold_score != 0:
                temp = copy.copy(data[i])
                temp.score = []
                temp.val_score = [temp.fold_score]
                temp.loss = []
                temp.val_loss = [temp.fold_loss]
                temp.model_info = temp.model_info+'_Fold'
                data.append(temp)
                
        temp = []
        for d in data:
            temp.append(get_key(d))
        
        keys = list(set(temp))
        def sort_func(x):
            if x.find('_Fold') != -1:
                x = x.replace('_Fold', '')
                return 100*np.mean([t.max_score() for t in data if get_key(t) == x]) - 0.05
            else:
                return 100*np.mean([t.max_score() for t in data if get_key(t) == x])
                
        keys.sort(key=sort_func , reverse=True)
            
        mean_column = {}
        for k in keys:
            mean_column[k] = int(100*round(np.mean([t.max_score() for t in data if get_key(t) == k]), 2))

        nets = {}
        for n in keys:
            nets[n] = get_name(n)
                
        names_max_len = max([len(v) for v in nets.values()])
        classes = sorted(list(set([get_fold_num(g) for g in data])))

        print(' '*(names_max_len + 5), end='')
        [print('F{:02d}\t'.format(c), end='') for c in classes]
        print('mean')

        pd_dict = {}
        pd_dict['model_names'] = nets.values()
        for c in classes:
            pd_dict[c] = []
        pd_dict['mean'] = mean_column.values()
        
        for net in nets.keys():
            print(nets[net], end=' '*(names_max_len - len(nets[net]) + 5))
            trains = [t for t in data if get_key(t) == net]
            for c in classes:
                temp = [t for t in trains if get_fold_num(t) == c]
                if len(temp) != 0:
                    num = int(100*max([t.max_score() for t in temp]))
                    print('{:02d}'.format(num), end="\t")
                    pd_dict[c].append(num)
                else:
                    print('\t', end="")
                    pd_dict[c].append(' ')
                    
            print('{:2d}'.format(mean_column[net]))
            if nets[net].find('_Fold') != -1:
                print('')

        if to_excel_path is not None:
            res = pd.DataFrame(pd_dict)
            path = to_excel_path if to_excel_path[-5:] == '.xlsx' else to_excel_path + '.xlsx'
            res.to_excel(path, sheet_name='sheet1', index=False)
    
    @staticmethod
    def reset_database(path_old, path_new):
        if os.path.isdir(path_new):
            shutil.rmtree(path_new)

        os.mkdir(path_new)
        shutil.copyfile(path_old + '/' + data_info.Labeled.FileNames.data_info_txt,
                        path_new + '/' + data_info.Labeled.FileNames.data_info_txt)
        shutil.copytree(path_old + '/backup', path_new + '/backup')

        shutil.copytree(path_old + '/backup/train', path_new + '/train')
        shutil.copytree(path_old + '/backup/test', path_new + '/test')

    @staticmethod
    def make_small_data(path, save_path, n_each_class, test_ratio=-1, clear_start = False, resize_images = None):
        """_summary_

        Args:
            path (str): source data path
            save_path (str): destenation save path
            n_each_class (int): how many images load for each class
            test_ratio (float, optional): test data to train data ratio. if set to -1 all test data from source data would be selected. Defaults to -1.
        """
        save_path = save_path + '/' + path.split('/')[-1] + '.SMALL'
        if os.path.isdir(save_path):
            if clear_start:
                shutil.rmtree(save_path)
                os.mkdir(save_path)
            else:
                return
        else:
            os.mkdir(save_path)
            
        if not os.path.isdir(save_path+'/test'):
            os.mkdir(save_path+'/test')
        if not os.path.isdir(save_path+'/train'):
            os.mkdir(save_path+'/train')

        if not os.path.isdir(path) or \
                not os.path.isfile(path+'/'+data_info.Labeled.FileNames.data_info_txt):
            raise Exception('incorrect database! (path : {})'.format(path))
        
        try:
            info = db.get_db(path+'/'+data_info.Labeled.FileNames.data_info_txt, g_utils.Labeled_data_info)[0]
        except:
            raise Exception('incorrect database! (path : {})'.format(path))
        
        database_info = db.get_db(path+'/'+data_info.Labeled.FileNames.data_info_txt, g_utils.Labeled_data_info)
        class_num = len(database_info[0].train_classes_count)

        if test_ratio != -1:
            n_each_class_test = int(n_each_class*test_ratio)
        else:
            
            n_each_class_test = max(info.test_classes_count)
            
        train_count = [0] * class_num
        test_count  = [0] * class_num
        
        for c in range(class_num):
            
            train_old = path + '/backup/train/'+str(c)+'/'
            test_old = path + '/backup/test/'+str(c)+'/'

            train_new = save_path + '/train/'+str(c)+'/'
            test_new = save_path + '/test/'+str(c)+'/'

            
            for _n_each_class, path_old, path_new, count_class in \
                [[n_each_class      , train_old , train_new ,train_count],
                 [n_each_class_test , test_old  , test_new  ,test_count]]:
            
                data_list = os.listdir(path_old)
                if len(data_list) <= _n_each_class:
                    count_class[c] = len(os.listdir(path_old))
                    shutil.copytree(path_old, path_new)
                else:
                    if not os.path.isdir(path_new):
                        os.mkdir(path_new)
                    i = 0
                    selected = []
                    data_list_copy = data_list.copy()
                    go_flag = True
                    while go_flag:
                        step = int(len(data_list)/_n_each_class)
                        for i in range(0, len(data_list), step):
                            selected.append(data_list[i])
                            data_list_copy.remove(data_list[i])
                            if len(selected) >= _n_each_class:
                                go_flag = False
                                break
                        data_list = data_list_copy.copy()

                    for i, file in enumerate(selected):
                        count_class[c] = count_class[c] + 1
                        if not resize_images:
                            shutil.copyfile(path_old + file, path_new + 'img{:4d}.png'.format(i))
                        else:                    
                            im = Image.open(train_old + file)
                            im = im.resize(resize_images)
                            im.save(train_new + 'img{:4d}.png'.format(i))

        _info = g_utils.Labeled_data_info(
        total_count = np.sum(train_count) + np.sum(test_count)
        ,train_total_count = np.sum(train_count)
        ,test_total_count = np.sum(test_count)
        ,train_classes_count = train_count
        ,test_classes_count = test_count
        ,classes_borders = info.classes_borders
        ,flow_max = info.flow_max
        ,flow_min = info.flow_min
        ,class_size = info.class_size
        ,excluded_videos_indexes = info.excluded_videos_indexes
        ,test_videos_indexes = info.test_videos_indexes
        ,total_test_data_percentage=info.total_test_data_percentage)
    
        db.insert(save_path + '/data_info.txt', _info)

    @staticmethod
    def freeze_model(model_name, model_path, save_path):
            model = keras.models.load_model(model_path + model_name)

            # Convert Keras model to ConcreteFunction
            full_model = tf.function(lambda x: model(x))
            full_model = full_model.get_concrete_function(
                x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

            # Get frozen ConcreteFunction
            frozen_func = convert_variables_to_constants_v2(full_model)
            frozen_func.graph.as_graph_def()

            # inspect the layers operations inside your frozen graph definition and see the name of its input and output tensors
            layers = [op.name for op in frozen_func.graph.get_operations()]
            print("-" * 50)
            print("Frozen model layers: ")
            for layer in layers:
                print(layer)

            print("-" * 50)
            print("Frozen model inputs: ")
            print(frozen_func.inputs)
            print("Frozen model outputs: ")
            print(frozen_func.outputs)

            # Save frozen graph from frozen ConcreteFunction to hard drive
            # serialize the frozen graph and its text representation to disk.
            tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                            logdir="./" + save_path,
                            name=model_name[:-3] + ".pb",
                            as_text=False)

            #Optional
            tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                            logdir="./" + save_path,
                            name=model_name[:-3] + ".pbtxt",
                            as_text=True)

            model.summary()



