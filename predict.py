import numpy as np
import warnings
import argparse
import tensorflow as tf

warnings.filterwarnings("ignore")
from datasets import ECGSequence
from densenet1d import _DenseBlock, _TransitionBlock

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get performance on test set from hdf5')
    parser.add_argument('--path_to_hdf5', type=str,
                        help='path to hdf5 file containing tracings')
    parser.add_argument('--path_to_model',  # or model_date_order.hdf5
                        help='file containing training model.')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    parser.add_argument('--output_file', default="./dnn_output.npy",  # or predictions_date_order.csv
                        help='output csv file.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size.')

    args, unk = parser.parse_known_args()
    if unk:
        warnings.warn("Unknown arguments:" + str(unk) + ".")

    # Import data
    seq = ECGSequence(args.path_to_hdf5, args.dataset_name, batch_size=args.batch_size)

    # Import model
    model = tf.keras.models.load_model(args.path_to_model, custom_objects={'_DenseBlock': _DenseBlock,
                                                                           '_TransitionBlock': _TransitionBlock},
                                       compile=False)
    model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam())
    y_score = model.predict(seq, verbose=1)

    # Generate dataframe
    np.save(args.output_file, y_score)

    print("Output predictions saved")
