from models import DB_BiLSTM
import keras
import tensorflow as tf
from datasets import DatasetLoader
import pandas as pd
import os 

if os.path.exists('/mnt/d/TMP/logs/'):
    os.rmdir('/mnt/d/TMP/logs/')
for i in range(1, 4):
    dl = DatasetLoader(image_shape=(299, 299), csv=f'./dataset/ucf101_train_{i}.csv', csv_test=f'./dataset/ucf101_test_{i}.csv', num_segments=3)
    ds, ts = dl.get_dataset()
    # for i in ds:
    #     # print(i)
    #     pass

    checkpoint_filepath = "/mnt/d/TMP/checkpoint"
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor="val_accuracy",
        save_best_only=True,
    )
    tensorboard = keras.callbacks.TensorBoard(f'/mnt/d/TMP/logs/{i}')

    net = DB_BiLSTM(image_shape=(299, 299, 3), n_class=101)
    adam = keras.optimizers.Adam(learning_rate=0.0001)
    train_accuracy = keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    net.compile(
        optimizer=adam,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=train_accuracy,
    )
    net.fit(
        ds,
        validation_data=ts,
        epochs=20,
        callbacks=[tensorboard],
    )

# from sklearn.model_selection import KFold

# kfold = KFold()
        
# df = pd.read_csv('./dataset/hmdb51.csv')
# for i, (train_index, test_index) in enumerate(kfold.split(df)):
#     df.iloc[train_index].to_csv(f'./dataset/hmdb51_train_{i}.csv')
#     df.iloc[test_index].to_csv(f'./dataset/hmdb51_test_{i}.csv')