from keras import Model
from keras.layers import Bidirectional, LSTM, Dense, InputLayer
from keras.applications import ResNet50, InceptionV3
from datasets import DatasetLoader
import tensorflow as tf

class DB_BiLSTM(Model):
    def __init__(self, image_shape, n_class, is_training = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_shape = image_shape
        self.backbone = self.construct_backbone()
        self.lstm = Bidirectional(LSTM(512))
        self.fc = Dense(n_class, activation=None if is_training else 'softmax')
        

    def construct_backbone(self):
        return InceptionV3(include_top=False, input_shape=self.image_shape, pooling='avg')
    

    
    def call(self, inputs, training=False):
        # segments = tf.unstack(inputs, axis=1)
        # Xs = []
        # for segment in segments:
        #     Xs.append(self.backbone(segment))
        
        in_shape = tf.shape(inputs) #(B,S,H,W,C)


        x = tf.reshape(inputs, (in_shape[0]*in_shape[1], in_shape[2], in_shape[3], in_shape[4]))
        x = self.backbone(x)
        new_shape = tf.shape(x)
        x = tf.reshape(x, (in_shape[0],in_shape[1], new_shape[1]))
        x = self.lstm(x)
        x = self.fc(x)
        # x = self.backbone(inputs)
        # print(x.shape)
        return x


class BiLSTM_Temporal(Model):
    def __init__(self, image_shape, n_class, is_training = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_shape = image_shape
        self.backbone = self.construct_backbone()
        self.lstm = Bidirectional(LSTM(512))
        self.temporal_lstm = Bidirectional(LSTM(512))
        self.fc = Dense(n_class, activation=None if is_training else 'softmax')
        

    def construct_backbone(self):
        return InceptionV3(include_top=False, input_shape=self.image_shape, pooling='avg')
    

    
    def call(self, inputs, training=False):
        # segments = tf.unstack(inputs, axis=1)
        # Xs = []
        # for segment in segments:
        #     Xs.append(self.backbone(segment))
        
        in_shape = tf.shape(inputs) #(B,S,H,W,C)


        x = tf.reshape(inputs, (in_shape[0]*in_shape[1], in_shape[2], in_shape[3], in_shape[4]))
        x = self.backbone(x)
        new_shape = tf.shape(x)
        x = tf.reshape(x, (in_shape[0],in_shape[1], new_shape[1]))
        x = self.lstm(x)
        x = self.fc(x)
        # x = self.backbone(inputs)
        # print(x.shape)
        return x
# dl = DatasetLoader(None, './dataset/hmdb51.csv', num_segments=5)
# ds = dl.get_dataset()
# for i in ds.take(1):
#     DB_BiLSTM(image_shape=(299, 299, 3), n_class=30)(i[0])
    
# input = InputLayer((299, 299, 3), 1)
# for 