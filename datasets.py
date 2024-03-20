import numpy as np
from numpy.random import randint
import os 
import cv2
import tensorflow as tf
import pandas as pd

class DatasetLoader():
    def __init__(self, image_shape, csv, csv_test=None, num_segments = 3):
        self.__shape__ = image_shape
        self.__csv__ = csv
        self.num_segments = num_segments
        self.__csv_test__ = csv_test



    def _sample_indices(self, cap, length = 1, num_segments = 3):
        """

        :param record: VideoRecord
        :return: list
        """

        num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        average_duration = (num_frames - length + 1) // num_segments
        # if average_duration == 0:
        #     print(1000)
        # print(num_frames)
        if average_duration > 0:
            offsets = np.multiply(list(range(num_segments)), average_duration) + randint(max(1, average_duration - length + 1), size=num_segments)
        elif num_frames > num_segments:
            offsets = np.sort(randint(num_frames - length + 1, size=num_segments))
        else:
            offsets = np.zeros((num_segments,))
        return (offsets).astype(np.int64)
    
    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
  
        return (offsets).astype(np.int64)
    
    # generate dataset index from videos
    @staticmethod
    def index(folder, flow_folder, output_file_path, headers=['file', 'flow_file', 'length', 'width', 'height', 'label']):
        with open(output_file_path, "w") as my_file:
            my_file.write(",".join(headers) + "\n")
            for i, c in enumerate(os.listdir(folder)):
                for f in os.listdir(os.path.join(folder, c)):
                    file_path = os.path.join(folder, c, f)
                    flow_file_path = os.path.join(flow_folder, c, f)
                    cap = cv2.VideoCapture(file_path)
                    
                    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`

                    formnatted_line = f'{file_path},{flow_file_path},{video_length},{width},{height},{i}'

                    my_file.write(formnatted_line + "\n")
                
    
    def get_dataset(self):
        
        part_1 = pd.read_csv(self.__csv__)
        dataset = tf.data.Dataset.from_tensor_slices(dict(part_1)).shuffle(1000000)
        
        # dataset = tf.data.TFRecordDataset(
        #     self.get_files_list(self.train_folder),
        #     num_parallel_reads=tf.data.experimental.AUTOTUNE,
        #     compression_type="GZIP",
        # )

        dataset = dataset.map(self.flatten,)
        dataset = dataset.map(self.construct_decode_fn(),)
        dataset = dataset.batch(16)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        if(self.__csv_test__ is not None):
            part_2 = pd.read_csv(self.__csv_test__)
            dataset_ts = tf.data.Dataset.from_tensor_slices(dict(part_2)).shuffle(1000000)
            
            # dataset = tf.data.TFRecordDataset(
            #     self.get_files_list(self.train_folder),
            #     num_parallel_reads=tf.data.experimental.AUTOTUNE,
            #     compression_type="GZIP",
            # )

            dataset_ts = dataset_ts.map(self.flatten,)
            dataset_ts = dataset_ts.map(self.construct_decode_fn(test=True),)
            dataset_ts = dataset_ts.batch(16)
            dataset_ts = dataset_ts.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            return dataset, dataset_ts
        return dataset

    @staticmethod
    def flatten(x):
        return [x['file'], x['flow_file'], x['label']]
    
    def get_images(self, cap, indices, n = 1):
        segments = []
        for frame_number in indices:
            # get total number of frames
            totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_number-2)

            for _ in range(n):
                 # check for valid frame number
                if frame_number >= 0 & frame_number <= totalFrames:
                    # set frame position
                    ret, frame = cap.read()
                    if not ret:
                        # print('no ret', totalFrames, frame_number)
                        segments.append(np.zeros(segments[0].shape))
                        continue
                        
                    segments.append(frame)
                    # print(totalFrames, frame_number)
        return np.float32(segments)
    
    @staticmethod
    def transform_image(images, resize):
        (N,H,W,C) = images.shape
        assert N*C <= 512, 'batch size to large'


        instack = np.transpose(images, (1,2,3,0)).reshape((H,W,C*N))
        outstack = cv2.resize(instack, resize)
        out_images = outstack.reshape((resize[0],resize[1],C,N)).transpose((3,0,1,2))
        return out_images

    def construct_decode_fn(self, test = False):
        @tf.numpy_function( Tout=[tf.float32, tf.float32])
        def decode_fn(*args):
            file, flow_file, label = args
            cap = cv2.VideoCapture(file.decode("utf-8"))

            indices = self._sample_indices(cap, 1, num_segments = 15 if test else self.num_segments)

            rgb = self.get_images(cap, indices)
            
            # cap = cv2.VideoCapture(flow_file.decode("utf-8"))

            # fps = cap.get(cv2.CAP_PROP_FPS)
            # indices = self._sample_indices(cap, 5, num_segments = 15 if test else self.num_segments)
            # flow = self.get_images(cap, indices, 5)
            rgb = DatasetLoader.transform_image(rgb, self.__shape__)
            # flow = DatasetLoader.transform_image(flow, self.__shape__)
            return rgb, tf.cast(label, tf.float32)
        
        return decode_fn
    
    @staticmethod
    def load_fake_dataset(W, H, B, S, C):
        return tf.zeros((B, S, H, W, C))
        

dl = DatasetLoader(None, './dataset/hmdb51.csv')
# ds = dl.get_dataset()
# import matplotlib.pyplot as plt
# for i in ds.take(1):
#     plt.imshow(np.uint8(i[1][0]),)
#     plt.show()
#     pass

# dl.index('/mnt/d/Datasets/UCF101/UCF-101/', '/mnt/d/Datasets/UCF101_Flow/', './dataset/UCF101.csv')