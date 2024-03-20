import cv2
import os
from pathlib import Path
import time
import numpy as np

class VideoToFlow():

    def get_list(self)->list:
        raise NotImplementedError

    def __init__(self, shape, input_loc) -> None:
        self.__shape__ = shape
        self.__input_loc__ = input_loc

    def __call__(self, output_loc):
        for file in self.get_list():
            self.video_to_frames(file['full_path'], os.path.join(output_loc, file['output_dir']))

    def dense_optical_flow(method, video_path, params=[], to_gray=False):
        # Read the video and first frame
        cap = cv2.VideoCapture(video_path)
        ret, old_frame = cap.read()
    
        # crate HSV & make Value a constant
        hsv = np.zeros_like(old_frame)
        hsv[..., 1] = 255
    
        # Preprocessing for exact method
        if to_gray:
            old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        
    def create_video_writer(self, size, fps, output_file):
        print(output_file)
        return cv2.VideoWriter(output_file, -1, fps, size)
    
    def get_flow(self, old_frame, new_frame, mask_shape):
        # Calculates dense optical flow by Farneback method 
        flow = cv2.calcOpticalFlowFarneback(old_frame, new_frame,  
                                        None, 
                                        0.5, 3, 15, 3, 5, 1.2, 0) 
        
        # Computes the magnitude and angle of the 2D vectors 
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1]) 

        mask = np.zeros(mask_shape, dtype=np.uint8)
        mask[..., 1] = 255
        
        # Sets image hue according to the optical flow  
        # direction 
        mask[..., 0] = angle * 180 / np.pi / 2
        
        # Sets image value according to the optical flow 
        # magnitude (normalized) 
        mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX) 
        return cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)


    def video_to_frames(self, input_file, output_loc):
        """Function to extract frames from input video file
        and save them as separate frames in an output directory.
        Args:
            input_loc: Input video file.
            output_loc: Output directory to save the frames.
        Returns:
            None
        """
        try:
            os.makedirs(os.path.dirname(output_loc))
        except OSError as e:
            pass
        # Log the time
        time_start = time.time()
        # Start capturing the feed
        cap = cv2.VideoCapture(input_file)
        # Find the number of frames
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
        print("Running file: ", input_file)
        print ("Number of frames: ", video_length)
        count = 0
        print ("Converting video..\n")
        # Start converting the video

        old_frame = None


        SIZE = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        FPS =  cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(output_loc, cv2.VideoWriter_fourcc(*'MJPG'), FPS, SIZE)

        

        while cap.isOpened():
            # Extract the frame
            ret, frame = cap.read()
            if not ret:
                continue
            shape = np.shape(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Write the results back to output location.
            if old_frame is None:
                old_frame = frame

            out.write(self.get_flow(old_frame, frame, shape))
            old_frame = frame
            
            count = count + 1

            
            # If there are no more frames left
            if (count > (video_length-1)):
                # Log the time again
                time_end = time.time()
                # Release the feed
                cap.release()
                out.release()
                # Print stats
                print ("Done extracting frames.\n%d frames extracted" % count)
                print ("It took %d seconds forconversion." % (time_end-time_start))
                break
        

class HMDB51Transform(VideoToFlow):
    def get_list(self) -> list:
        o = []
        for c in os.listdir(self.__input_loc__):
            for f in os.listdir(os.path.join(self.__input_loc__, c)):
                o.append({
                    'full_path' : os.path.join(self.__input_loc__, c, f),
                    'output_dir' : os.path.join(c, f)
                })
        return o
    
HMDB51Transform(None, '/mnt/d/Datasets/UCF101/UCF-101/')('/mnt/d/Datasets/UCF101_Flow/')