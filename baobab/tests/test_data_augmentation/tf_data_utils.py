import tensorflow as tf
import numpy as np
__all__  = ['generate_simple_tf_record', 'parse_example', 'tf_img_size', 'tf_y_names', 'tf_data_size']

tf_img_size = 3
tf_y_names = ['y0', 'y1']
tf_data_size = 17

def generate_simple_tf_record(tf_record_path, y_names):
    """Generate a simple TFRecord file

    Note
    ----
    Made for demonstration and testing purposes

    Parameters
    ----------
    y_names : list of str
        names of the Y labels
    tf_record_path : str or os.path object
        where the TFRecord file will be saved

    """
    # Initialize the writer object and write the lens data
    with tf.io.TFRecordWriter(tf_record_path) as writer:
        for idx in range(tf_data_size):
            # The image must be converted to a tf string feature
            image_feature = tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[np.random.randn(tf_img_size, tf_img_size).astype(np.float32).tostring()]))
            # Initialize a feature dictionary with the image, height, width
            feature = {
                    'image' : image_feature,
                    'index': tf.train.Feature(int64_list=tf.train.Int64List(value=[idx]))
                    }
            # Add all of the lens parameters to the feature dictionary
            for y_name in y_names:
                feature[y_name] = tf.train.Feature(float_list=tf.train.FloatList(value=[np.random.randn()]))
            # Create the tf example object
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # Write out the example to the TFRecord file
            writer.write(example.SerializeToString())

def parse_example(example):
    """Parse the image and Y values
    
    Parameters
    ----------
    example : tf.train.Example
        a single training example

    Returns
    -------
    decoded example

    """
    metadata = {
                'image' : tf.io.FixedLenFeature([], tf.string),
                'index' : tf.io.FixedLenFeature([], tf.int64),
                }
    for y_name in tf_y_names:
        metadata[y_name] = tf.io.FixedLenFeature([], tf.float32)
    parsed_example = tf.io.parse_single_example(example, metadata)
    image = tf.io.decode_raw(parsed_example['image'], out_type=float)
    image = tf.reshape(image, (tf_img_size, tf_img_size, 1))
    y_vals = tf.stack([parsed_example[y_name] for y_name 
        in tf_y_names])
    return image, y_vals
