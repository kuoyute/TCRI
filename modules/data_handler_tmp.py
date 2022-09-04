import tensorflow as tf
from functools import partial
import tensorflow_addons as tfa
from modules.tfrecord_generator import get_or_generate_tfrecord


def ascii_array_to_string(ascii_array):
    string = ''
    for ascii_code in ascii_array:
        string += chr(ascii_code)
    return string


def deserialize(serialized_TC_history):
    features = {
        'history_len': tf.io.FixedLenFeature([], tf.int64),
        'images': tf.io.FixedLenFeature([], tf.string),
        'intensity': tf.io.FixedLenFeature([], tf.string),
        'frame_ID': tf.io.FixedLenFeature([], tf.string),
        'lon': tf.io.FixedLenFeature([], tf.string),
        'lat': tf.io.FixedLenFeature([], tf.string),
        'env_feature': tf.io.FixedLenFeature([], tf.string),
        'SHTD': tf.io.FixedLenFeature([], tf.string)
    }

    example = tf.io.parse_single_example(serialized_TC_history, features)
    history_len = tf.cast(example['history_len'], tf.int32)

    images = tf.reshape(
        tf.io.decode_raw(example['images'], tf.float32),
        [history_len, 64, 64, 4]
    )
    intensity = tf.reshape(
        tf.io.decode_raw(example['intensity'], tf.float64),
        [history_len]
    )
    intensity = tf.cast(intensity, tf.float32)
    
    lon = tf.reshape(
        tf.io.decode_raw(example['lon'], tf.float64),
        [history_len]
    )    
    lon = tf.cast(lon, tf.float32)
    
    lat = tf.reshape(
        tf.io.decode_raw(example['lat'], tf.float64),
        [history_len]
    )    
    lat = tf.cast(lat, tf.float32)

    env_feature = tf.reshape(
        tf.io.decode_raw(example['env_feature'], tf.float64),
        [-1 ,history_len]
    )    
    env_feature = tf.cast(env_feature, tf.float32)
    
    SHTD = tf.reshape(
        tf.io.decode_raw(example['SHTD'], tf.float64),
        [history_len]
    )    
    SHTD = tf.cast(SHTD, tf.float32)   
    
    frame_ID_ascii = tf.reshape(
        tf.io.decode_raw(example['frame_ID'], tf.uint8),
        [history_len, -1]
    )

    return images, intensity, lon, lat, env_feature, history_len, frame_ID_ascii, SHTD



def translation(starting_lon, starting_lat, ending_lon, ending_lat):
    west = 360. * tf.cast((ending_lon < 0), tf.float32)
    ending_lon = tf.cast((ending_lon), tf.float32) + west

    west = 360. * tf.cast((starting_lon < 0), tf.float32)
    starting_lon = tf.cast((starting_lon), tf.float32) + west

    lon_dif = tf.math.abs((ending_lon-starting_lon))
    cross_0 = 360. * tf.cast((lon_dif > 100), tf.float32)
    lon_dif = cross_0 - lon_dif
    
    lat_dif= tf.cast(ending_lat-starting_lat, tf.float32)
    
    speed = 108/6*tf.sqrt(tf.square(lon_dif) + tf.square(lat_dif))    # in  km/hr
    return speed


def breakdown_into_sequence(
    images, intensity, lon, lat, env_feature, history_len, frame_ID_ascii, encode_length, estimate_distance
):
    sequence_num = history_len - (encode_length + estimate_distance) + 1
    starting_index = tf.range(2,sequence_num)

    image_sequences = tf.map_fn(
        lambda start: images[start: start+encode_length],
        starting_index, fn_output_signature=tf.float32
    )
    
    starting_frame_ID_ascii = frame_ID_ascii[encode_length - 1+2:-estimate_distance]

    starting_intensity = intensity[encode_length - 1+2: -estimate_distance]
    starting_intensity = tf.cast(starting_intensity, tf.float32)                 
    ending_intensity = intensity[encode_length + estimate_distance - 1+2:]
    ending_intensity = tf.cast(ending_intensity, tf.float32)
    RI_labels = ending_intensity
    intensity_change = ending_intensity-starting_intensity
        
    starting_lon = lon[encode_length - 1+2:-estimate_distance]
    previous_6hr_lon = lon[encode_length - 1: -estimate_distance -2]
    starting_lat = lat[encode_length - 1+2:-estimate_distance]    
    previous_6hr_lat = lat[encode_length - 1: -estimate_distance -2]
    
    past_translation_speed  = translation(starting_lon, starting_lat, previous_6hr_lon, previous_6hr_lat)

    starting_lat = tf.math.abs((starting_lat))
#    ending_lat = tf.math.abs((ending_lat))     
    
    starting_env_feature = env_feature[:, encode_length - 1+2:-estimate_distance]
    ending_land_dis = env_feature[0, encode_length + estimate_distance - 1+2:] 
#    ending_env_feature = env_feature[:, encode_length + estimate_distance - 1+2:]  
    
    
    feature = tf.concat([[starting_lat], [past_translation_speed], starting_env_feature, [ending_land_dis]], 0) 
    feature = tf.transpose(feature)

    return tf.data.Dataset.from_tensor_slices((image_sequences, RI_labels, feature, starting_frame_ID_ascii, intensity_change))



def image_preprocessing(images, intensity, lon, lat, env_feature, history_len, frame_ID_ascii, SHTD, rotate_type,input_image_type):
    images_channels = tf.gather(images, input_image_type, axis=-1)
    if rotate_type == 'single':
        angles = tf.random.uniform([history_len], maxval=360)
        rotated_images = tfa.image.rotate(images_channels, angles=angles)
    elif rotate_type == 'series':
        angles = tf.ones([history_len]) * tf.random.uniform([1], maxval=360)
        rotated_images = tfa.image.rotate(images_channels, angles=angles)
    elif rotate_type == 'shear':
        print('this is the shear rotation run')
        rotated_images = tfa.image.rotate(images_channels, angles=-SHTD*0.01745329252)    
    else:
        rotated_images = images_channels
    #images_64x64 = tf.image.central_crop(rotated_images, 0.7)

    return rotated_images, intensity, lon, lat, env_feature, history_len, frame_ID_ascii


def get_tensorflow_datasets(
    data_folder, batch_size, encode_length,
    estimate_distance, rotate_type,input_image_type
):
    tfrecord_paths = get_or_generate_tfrecord(data_folder)
    datasets = dict()
    for phase, record_path in tfrecord_paths.items():
        serialized_TC_histories = tf.data.TFRecordDataset(
            [record_path], num_parallel_reads=8
        )
        TC_histories = serialized_TC_histories.map(
            deserialize, num_parallel_calls=tf.data.AUTOTUNE
        )

        min_history_len = encode_length + estimate_distance+7
        long_enough_histories = TC_histories.filter(
            lambda a, b, c, d, e, f, g, h: f >= min_history_len
        )

        preprocessed_histories = long_enough_histories.map(
            partial(
                image_preprocessing,
                rotate_type=rotate_type,
                input_image_type=input_image_type
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
        TC_sequence = preprocessed_histories.interleave(
                partial(
                breakdown_into_sequence,
                encode_length=encode_length,
                estimate_distance=estimate_distance
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        dataset = TC_sequence.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(4)
        datasets[phase] = dataset

    return datasets
