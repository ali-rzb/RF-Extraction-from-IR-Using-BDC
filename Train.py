from keras.layers import Dense, Flatten
from keras import Sequential
import Functions.LocalDB as db
from Functions.CNN import CNN
import Functions.DataUtils as d_utils
from keras.applications.efficientnet import EfficientNetB0
from keras.applications.densenet import DenseNet121
from keras.applications.inception_v3 import InceptionV3
import os

frame_tale = 1
augment_ratio = 2
batch_size = 32
stp_patience = 10
class_num = 10
epochs = 100
folds = 5


models = [DenseNet121]
for arch in models:
    try:
        os.mkdir(f'models/{arch.__name__}')
    except:
        pass

cnn = CNN(class_num=class_num, frame_tale=frame_tale, augment_ratio=augment_ratio, batch_size=batch_size, 
            early_stopping_patience=stp_patience, resize_images=(75, 75))
cnn.prepare_data(shuffle_frames=True, color_mode='rgb', k_fold=folds)
del cnn
for f in range(folds):
    cnn = CNN(class_num=class_num, frame_tale=frame_tale, augment_ratio=augment_ratio, batch_size=batch_size, 
            early_stopping_patience=stp_patience, resize_images=(75, 75))
    cnn.prepare_data(shuffle_frames=True, color_mode='rgb', n_k_fold=f, k_fold=folds)
    for arch in models:
        path = f'models/{arch.__name__}'
        
        cnn.model = Sequential()
        cnn.model.add(arch(include_top=False, input_shape=(cnn.image_size[0], cnn.image_size[1], 3), classes=class_num, weights=None))
        cnn.model.add(Flatten())
        cnn.model.add(Dense(64, activation='relu'))
        cnn.model.add(Dense(class_num, activation='softmax'))

        cnn.history = CNN.hist(model_info=f'F{f}_C{class_num:02d}_{arch.__name__}')
        cnn.train_model(epochs=epochs, save_model_path=path)
        db.insert(f'{path}/log.txt', cnn.history)


