import argparse
import tensorflow as tf

from datasets import ECGSequence
from utils import ecg_feature_extractor
# from densenet1d import _DenseBlock, _TransitionBlock

# from clr.learningratefinder import LearningRateFinder
from clr.clr_callback import CyclicLR
from clr import config

if __name__ == "__main__":
    # Get data and train
    parser = argparse.ArgumentParser(description='Train neural network.')

    parser.add_argument('--path_to_hdf5', type=str,
                        help='path to hdf5 file containing tracings')
    parser.add_argument('--path_to_csv', type=str,
                        help='path to csv file containing annotations')
    parser.add_argument('--val_split', type=float, default=0.02,
                        help='number between 0 and 1 determining how much of '
                             'the data is to be used for validation. The remaining '
                             'is used for validation')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    parser.add_argument('--epochs', type=int, default=70,
                        help='maximum number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    args = parser.parse_args()

    # Creating training and validation sets
    train_seq, valid_seq, train_size = ECGSequence.get_train_and_val(args.path_to_hdf5, args.dataset_name,
                                                                     args.path_to_csv, args.batch_size, args.val_split)

    # Creating model
    # If you are continuing an interrupted section, uncomment line bellow:
    # model = tf.keras.models.load_model('backup_model_last.hdf5',
    #                                    custom_objects={'_DenseBlock': _DenseBlock,
    #                                                    '_TransitionBlock': _TransitionBlock},
    #                                    compile=False)

    inputs = tf.keras.layers.Input(shape=train_seq[0][0].shape[1:], dtype=train_seq[0][0].dtype)
    backbone_model = ecg_feature_extractor(input_layer=inputs)
    x = tf.keras.layers.GlobalMaxPooling1D()(backbone_model.output)
    x = tf.keras.layers.Dense(units=train_seq.n_classes, activation='sigmoid', kernel_initializer='he_normal')(x)
    model = tf.keras.models.Model(inputs=backbone_model.input, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(args.lr))

    # Optimization settings
    cycle_rate = CyclicLR(mode=config.CLR_METHOD,
                          base_lr=config.MIN_LR,
                          max_lr=config.MAX_LR,
                          step_size=config.STEP_SIZE * train_size // args.batch_size)

    callbacks = [tf.keras.callbacks.EarlyStopping(patience=9,
                                                  min_delta=0.001,
                                                  verbose=1),
                 cycle_rate]

    # Create log
    callbacks += [tf.keras.callbacks.TensorBoard(log_dir='/content/drive/MyDrive/ECG12Dataset/logs', write_graph=False),
                  tf.keras.callbacks.CSVLogger('training.log',
                                               append=False)]  # Change append to true if continuing training

    # Save the BEST and LAST model
    callbacks += [tf.keras.callbacks.ModelCheckpoint('/content/drive/MyDrive/ECG12Dataset/backup_model_last.hdf5'),
                  tf.keras.callbacks.ModelCheckpoint('/content/drive/MyDrive/ECG12Dataset/backup_model_best.hdf5',
                                                     save_best_only=True)]

    # Train neural network
    history = model.fit(train_seq,
                        epochs=args.epochs,
                        initial_epoch=0,  # If you are continuing a interrupted section change here
                        callbacks=callbacks,
                        validation_data=valid_seq,
                        verbose=1)

    # Save final result
    model.save("/content/drive/MyDrive/ECG12Dataset/final_model.hdf5")
