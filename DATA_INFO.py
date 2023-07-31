import os


class Main:
    path = 'Data/'


class Raw:
    path = Main.path + '0_Raw/'

    def count():
        return len([name for name in os.listdir(Raw.path) if os.path.isdir(os.path.join(Raw.path, name))])

    class FileNames:
        subject_info = '0_subject_info.txt'
        spiro_data_file = '1_spiro.txt'
        spiro_timer = '2_spiro_timer.txt'
        vid_timer = '3_vid_timer.txt'
        video_positions_flow = '4_flow_position'
        video_name = 'vid'
        

class Trimmed:
    path = Main.path + '1_Trimmed/'

    def count():
        return len([name for name in os.listdir(Trimmed.path) if os.path.isfile(os.path.join(Trimmed.path, name)) and (name[-4:] == '.mp4')])

    class FileNames:
        video_name = '{:02d}vid'
        data_name = '{:02d}data'
        shifts_file = 'shifts.txt'


class Synced:
    path_first_sync = Main.path + '2_Synced/first/'
    path = Main.path + '2_Synced/'

    def count():
        return len([name for name in os.listdir(Synced.path) if os.path.isfile(os.path.join(Synced.path, name)) and (name[-4:] == '.mp4')])

    class FileNames:
        video_name = '{:02d}vid'
        data_name = '{:02d}data'


class Labeled:
    path = Main.path + '3_Labeled/'    
    class FileNames:
        Folder_Name = 'T{:d}.C{:d}.V{:d}.A{}'
        data_info_txt = 'data_info.txt'
        frames_info = 'frames_info.txt'
    

if not os.path.isdir(Main.path):
    os.mkdir(Main.path)

if not os.path.isdir(Raw.path):
    os.mkdir(Raw.path)

if not os.path.isdir(Trimmed.path):
    os.mkdir(Trimmed.path)

if not os.path.isdir(Synced.path):
    os.mkdir(Synced.path)
    
if not os.path.isdir(Synced.path_first_sync):
    os.mkdir(Synced.path_first_sync)



    
