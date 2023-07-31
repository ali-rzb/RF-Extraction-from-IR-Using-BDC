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
import Functions.DataUtils as d_utils
import Functions.LocalDB as db
import DATA_INFO as data_info
import cv2, os, math, ffmpeg
import numpy as np

class VideoReader_cv2:
    def __init__(self, VideoFullPath):
        if not os.path.isfile(VideoFullPath):
            raise Exception(
                'File Does Not Exist! (path : {})'.format(VideoFullPath))
        self.cap = cv2.VideoCapture(VideoFullPath)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.video_shape = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        self.time_length = float(self.length)/float(self.fps)
        self.video_dir = os.path.dirname(VideoFullPath)
        self.video_name = os.path.basename(VideoFullPath)

    def ReadFrame(self, FrameNumber):
        """Reads a certain frame.

        NOTE: The returned frame is assumed to be with `RGB` channel order.

        Args:
        position: Optional. If set, the reader will read frames from the exact
            position. Otherwise, the reader will read next frames. (default: None)
        """
        if FrameNumber is not None and FrameNumber < self.length:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, FrameNumber)
            self.position = FrameNumber
        elif FrameNumber is None:
            raise Exception('Please enter a frame number to fetch!')
        elif FrameNumber >= self.length:
            raise Exception(f'Video length is {self.length}, frame number can\'t be bigger or equal to that!\n(frame number : {FrameNumber})')

        success, frame = self.cap.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame if success else None

    def Read(self, gray_scale = True):
        ret, frame = self.cap.read()
        if ret:
            if gray_scale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.position = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return ret, frame

    def Trim(self, t1, t2, path, name, second_or_frame_number=True):
        fourcc = cv2.VideoWriter_fourcc(*'XMP4')
        fileFullPath = path + name + '.mp4'
        out = cv2.VideoWriter(fileFullPath, fourcc, self.fps, self.video_shape)
        T = 1/self.fps

        if second_or_frame_number:
            n_t1 = int(t1/T)
            n_t2 = int(t2/T)
        else:
            n_t1 = t1
            n_t2 = t2

        j = 0
        i = 0
        while i <= n_t2:
            ret, frame = self.Read()
            if not ret:
                break

            if i >= n_t1:
                out.write(frame)
                j = j + 1
            i = i + 1
        out.release()

    def TrimFromFile(self, path,
                     timer=data_info.Raw.FileNames.vid_timer,
                     save_path=data_info.Trimmed.path,
                     save_name=data_info.Trimmed.FileNames.video_name):

        file = open(path + timer, 'r')
        lines = file.readlines()
        start_times = list(np.int32(lines))
        stop_times = list(np.int32(lines)+20)

        self.save_path = save_path
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        dt = 1/self.fps
        time = 0
        fourcc = cv2.VideoWriter_fourcc(*'XMP4')
        n_trims = len(start_times)
        current_trim = None
        while(True):
            ret, frame = self.Read()
            if not ret:
                break
            for i in range(n_trims):
                if (time >= start_times[i]) and (current_trim == None) and (time <= stop_times[i]):
                    current_trim = i
                    fileFullPath = g_utils.get_free_file_name(
                        save_path, save_name, '.mp4')
                    out = cv2.VideoWriter(
                        fileFullPath, fourcc, self.fps, self.video_shape)
                    print('\t\t', start_times[i], fileFullPath)
                elif (time >= stop_times[i]) and current_trim == i:
                    current_trim = None
                    out.release()
                    cv2.destroyAllWindows()
            if current_trim != None:
                out.write(frame)
                # cv2.imshow(fileFullPath, frame)
            # key = cv2.waitKeyEx(0)
            # if key & 0xFF == 27:
            #     break
            time = time + dt

    def done(self):
        self.cap.release()

class VideoWriter_cv2:
    def __init__(self, VideoFullPath, fps, shape):
        fourcc = cv2.VideoWriter_fourcc(*'XMP4')
        self.out = cv2.VideoWriter(VideoFullPath, fourcc, fps, shape)

    def write(self, frame):
        self.out.write(frame)

    def done(self):
        self.out.release()

class VideoReader_ffmpeg:
    def __init__(self, VideoPath):
        self.video_path = VideoPath
        try:
            self.probe = ffmpeg.probe(VideoPath)
            video_stream = next(
                (stream for stream in self.probe['streams'] if stream['codec_type'] == 'video'), None)
            if video_stream is None:
                raise Exception('No video stream found')
            self.video_info = video_stream

        except ffmpeg.Error as err:
            raise Exception(str(err.stderr, encoding='utf8'))

    def read_frame(self, frame_num):
        """
        Read any frame with specified number of frames
        """
        out, err = (
            ffmpeg.input(self.video_path)
            .filter('select', 'gte(n,{})'.format(frame_num))
            .output('pipe:', vframes=1, format='image2')
            .global_args('-loglevel', 'quiet')
            .run(capture_stdout=True)
        )
        image_array = np.asarray(bytearray(out), dtype="uint8")
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return image

class frame_info(db.data_object_class):
    def __init__(self, video_number : int, frame_number : int, flow : float, image_name : str, image_path : str, class_number : int, mode : str, n_k_fold : int, k_fold : int, id = None):
        self.video_number = video_number
        self.frame_number = frame_number
        self.flow = flow
        self.class_number = class_number
        self.mode = mode
        
        self.image_name = image_name
        self.image_path = image_path        
        
        self.n_k_fold = n_k_fold
        self.k_fold = k_fold
        
        self.id = id
        
    def get_full_path(self):
        return os.path.normpath(f'{self.image_path}/{self.class_number}/{self.image_name}')

def get_ROI_avg(video_path, data_path, 
                show_frames = False):
    data = d_utils.File.read(data_path)
    video = VideoReader_cv2(video_path)

    positions = list(
        zip(np.int32(data['position_y']), np.int32(data['position_x'])))
    j = 0
    avg = []
    while True:
        ret, frame = video.Read()
        if not ret:
            break
        temp, cut = getAvgOnCircle(frame, positions[j], 8)
        j = j + 1
        avg.append(temp)
        if show_frames:
            cv2.imshow('cut',cut)
            key = cv2.waitKey(1)
            if key == 27:
                break
    video.cap.release()
    time = np.arange(len(avg))*(1/20)

    return time, avg

def getSquareROI(img, pos, margin):
    width = np.shape(img)[0]
    height = np.shape(img)[1]
    circle = (pos[1], pos[0])
    margin_top = margin[0]
    margin_right = margin[1]
    margin_bottom = margin[2]
    margin_left = margin[3]
    x_left, x_right, y_top, y_bottom = circle[1] - margin_left, circle[1] + \
        margin_right, circle[0] - margin_top, circle[0] + margin_bottom
    [x_left, x_right] = g_utils.clamp([x_left, x_right], 0, height-1)
    [y_top, y_bottom] = g_utils.clamp([y_top, y_bottom], 0, width-1)

    if len(np.shape(img)) == 2 :
        result = img[y_top: y_bottom, x_left: x_right].copy()
    else:
        result = img[y_top: y_bottom, x_left: x_right,:].copy()
    # preview = img.copy()
    # for x in range(x_left, x_right):
    #     for y in range(y_top, y_bottom):
    #         preview[(y, x)] = preview[(y,x)] + 20

    return result

def houghTransform(img, min_d, param1, param2, min_r, max_r):
    circles = None
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.5, min_d, circles, param1, param2, min_r, max_r)
    ret = False
    if circles is not None:
        ret = False
        circles = np.uint16(np.around(circles))
        for pt in circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            cv2.circle(img, (a, b), r, (0, 255, 0), 2)
    
    return ret, img

def getAvgOnCircle(img, pos, radius):
    _img = img.copy()
    sum_of_pixles = []
    width = np.shape(img)[0]
    height = np.shape(img)[1]
    circle = (pos[0], pos[1])
    indices = []
    x_min, x_max, y_min, y_max = circle[1] - \
        radius, circle[1]+radius, circle[0]-radius, circle[0]+radius
    [x_min, x_max] = np.int16(g_utils.clamp([x_min, x_max], 0, width-1))
    [y_min, y_max] = np.int16(g_utils.clamp([y_min, y_max], 0, height-1))
    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            dx = x - circle[1]
            dy = y - circle[0]
            distanceSquared = dx * dx + dy * dy
            if distanceSquared <= radius*radius:
                indices.append((x, y))
                sum_of_pixles.append(img[(x, y)])
                _img[(x, y)] = 0

    return np.mean(sum_of_pixles), _img

def getCircle(img, first_point, last_circle, check_radius):
    width = np.shape(img)[0]
    height = np.shape(img)[1]
    min_r, max_r = 6, 11
    min_d = 100
    param1 = 100
    param2 = 23

    circles = None
    circles = cv2.HoughCircles(
        img, cv2.HOUGH_GRADIENT, 1.5, min_d, circles, param1, param2, min_r, max_r)

    sum_of_pixles = []

    position = (math.nan, math.nan)
    radius = math.nan
    if circles is None or \
        not g_utils.isInRange(circles[0, 0][0], circles[0, 0][1],
                            first_point[0]-check_radius,
                            first_point[0]+check_radius,
                            first_point[1]-check_radius,
                            first_point[1]+check_radius):

        circle = last_circle
        if circle is not None:
            radius = circle[2]
    else:
        circles = np.uint16(np.around(circles))
        circle = circles[0, 0]
        radius = circle[2]-2

    if circle is not None:
        position = (circle[1], circle[0])
        indices = []
        x_min, x_max, y_min, y_max = circle[1] - \
            radius, circle[1]+radius, circle[0]-radius, circle[0]+radius
        [x_min, x_max] = g_utils.clamp([x_min, x_max], 0, width-1)
        [y_min, y_max] = g_utils.clamp([y_min, y_max], 0, height-1)
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                dx = x - circle[1]
                dy = y - circle[0]
                distanceSquared = dx * dx + dy * dy
                if distanceSquared <= radius*radius:
                    indices.append((x, y))
                    sum_of_pixles.append(img[(x, y)])
                    img[(x, y)] = 0

    avg = math.nan
    if len(sum_of_pixles) != 0:
        avg = round(np.average(sum_of_pixles), 2)

    return avg, img, position, radius

first_point = None
def click_event(event, x, y, flags, params):
    global first_point
    if event == cv2.EVENT_LBUTTONUP:
        first_point = (x, y)
        cv2.destroyWindow("select starting point")

def getVidCircleAvg(path, name, save_path, save_name,
                    save_or_not=True):
    video = VideoReader_cv2(path+'/'+name+'.mp4')
    global first_point
    first_point = None
    list = []
    positions_list = []
    radius_list = []

    nan_list = []
    i = 0
    print('\n\n'+name)
    print('PRESS → Jump Frame // Esc to Jump video // r to Reset Processing {}.mp4'.format(name))
    jump_over_the_video = False

    last_circle = None
    while(True):
        ret, frame = video.Read()
        if not ret:
            break

        if first_point is None:
            cv2.imshow('select starting point', frame)
            cv2.setMouseCallback('select starting point', click_event)
            key = cv2.waitKeyEx(0)
            if key == 114:          # r
                video = VideoReader_cv2(path+'/'+name+'.mp4')
                first_point = None
                list = []
                nan_list = []
                i = 0
                print("\tProcessing Reseted on {}.mp4".format(name))
            elif key == 27:         # Esc
                jump_over_the_video = True
                print("\t{}.mp4 ignored!".format(name))
                break
            elif key == 2555904:    # →
                cv2.destroyWindow("select starting point")
                print("\tframe number {} ignored!".format(i))
                i = i + 1
            continue
        else:
            avg, image, position, radius = getCircle(
                frame, first_point, last_circle, 7)
            if not math.isnan(radius):
                last_circle = (position[1], position[0], radius)

            if math.isnan(avg):
                nan_list.append(i)

            list.append(avg)
            positions_list.append(position)
            radius_list.append(radius)
            i = i + 1
            cv2.imshow('Processing '+name, image)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    video.cap.release()

    cv2.destroyAllWindows()

    if not jump_over_the_video:
        # filling nan cells with near cells
        if len(nan_list) != 0:
            if nan_list[0] == 0:
                # finding first not nan
                for i in range(len(nan_list)):
                    if nan_list[i] != i:
                        end_of_nans = i-1
                        break
                    else:
                        end_of_nans = i

                if end_of_nans == 0:
                    list[0] = list[1]
                    nan_list.pop(0)
                else:
                    for i in range(end_of_nans, -1, -1):
                        list[i] = list[i+1]
                        nan_list.pop(i)

            if nan_list != []:
                for i in range(len(nan_list)):
                    list[nan_list[i]] = list[nan_list[i]-1]

        list = np.array(list)
        list = (list - min(list))
        list = list / max(list)

        if save_or_not:
            path = save_path + save_name + '.txt'
            positions_list = np.array(positions_list)
            try:
                d_utils.File.add_or_update(
                    zip(np.round(list, 5),
                        positions_list[:, 0], positions_list[:, 1], radius_list),
                    path, ['average', 'position_x', 'position_y', 'radius'])
            except:
                d_utils.File.save(
                    data = zip(np.round(list, 5),positions_list[:, 0], positions_list[:, 1], radius_list),
                    path=path, column_names=['average', 'position_x', 'position_y', 'radius'])
