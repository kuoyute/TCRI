import tensorflow as tf
import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from modules.data_downloader import download_data
from mpl_toolkits.basemap import Basemap
import math
from datetime import timedelta
pd.options.mode.chained_assignment = None

def remove_outlier_and_nan(numpy_array, upper_bound=1000):
    numpy_array = np.nan_to_num(numpy_array, copy=False)
    numpy_array[numpy_array > upper_bound] = 0
    VIS = numpy_array[:, :, :, 2]
    VIS[VIS > 1] = 1  # VIS channel ranged from 0 to 1
    return numpy_array

def remove_no_ships(image_matrix, info_df):
    noships = np.argwhere(np.isnan(list(info_df.U200)))
    image_matrix  = np.delete(image_matrix, noships, axis=0)
    info_df = info_df.dropna(subset = ['U200']).reset_index(drop=True)
    return image_matrix, info_df
    
    
def flip_SH_images(image_matrix, info_df):
    SH_idx = info_df.index[info_df.region == 'SH']
    image_matrix[SH_idx] = np.flip(image_matrix[SH_idx], 1)
    return image_matrix


def data_cleaning_and_organizing(image_matrix, info_df):
    image_matrix, info_df = remove_no_ships(image_matrix, info_df)    
    image_matrix = remove_outlier_and_nan(image_matrix)
#    image_matrix = flip_SH_images(image_matrix, info_df)
    return image_matrix, info_df


def data_split(image_matrix, info_df, phase):
    if phase == 'train':
        target_index = info_df.index[info_df.ID < '2015000']
    elif phase == 'valid':
        target_index = info_df.index[(info_df.ID > '2015000') & (info_df.ID < '2017000')]
    elif phase == 'test':
        target_index = info_df.index[info_df.ID > '2017000']

    new_image_matrix = image_matrix[target_index]
    new_info_df = info_df.loc[target_index].reset_index(drop=True)
    return new_image_matrix, new_info_df


def group_by_id(image_matrix, info_df):

    id2indices_group = info_df.groupby('ID', sort=False).groups
    indices_groups = list(id2indices_group.values())

    image_matrix = [image_matrix[indices] for indices in indices_groups]
    info_df = [info_df.iloc[indices] for indices in indices_groups]

    return image_matrix, info_df


def land_distance(lon, lat, m_map, maptxt):
    is_land = m_map.is_land(lon, lat)
    if lon <= 5:
        lon = lon+360.
    near_land = maptxt[(abs(maptxt.m_lon - lon) <= 5) | (abs(maptxt.m_lat - lat) <= 5)]
    dis = 110 * min(((near_land.m_lon - lon)**2 + (near_land.m_lat - lat)**2)** 0.5) if len(near_land) != 0 else 550.
    dis = 550. if dis>550 else dis
    dis = -dis if is_land else dis

    return dis

def coastline():
    df = pd.read_csv('./coastline.csv')
    for i in range(len(df)):
        if df.m_lon[i] < 0:
            df.m_lon[i] = df.m_lon[i] + 360. 
        if df.m_lon[i] <= 10:
            df = df.append(df.loc[df.index[i]], ignore_index=True)
            df.m_lon[len(df)-1] = df.m_lon[len(df)-1] + 360.     
    return df

def write_tfrecord(image_matrix, info_df, tfrecord_path, m_map, maptxt):

    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _encode_tfexample(single_TC_images, single_TC_info, m_map, maptxt):            
        history_len = single_TC_info.shape[0]
        frame_ID = single_TC_info.ID + '_' + single_TC_info.time
        
        lon = single_TC_info.lon.to_numpy()
        lat = single_TC_info.lat.to_numpy()
        
        land_dis = np.ones(len(lon))
        for i in range(len(land_dis)):
            land_dis[i] = land_distance(lon[i], lat[i], m_map, maptxt)     
            
        region_one_hot = {'WP':1., 'EP':2. , 'AL':3., 'SH':4., 'CP':5., 'IO':6.}
        region_string = list(single_TC_info.region)
        region = []
        for i in range(len(region_string)):
            region.append(region_one_hot[region_string[i]])
        region = np.array(region)      

        single_TC_info['global_lon'] = (single_TC_info.lon+180) % 360 - 180  # calibrate longitude, ex: 190 -> -170
        # --- time feature ---
        single_TC_info['GMT_time'] = pd.to_datetime(single_TC_info.time, format='%Y%m%d%H')
        single_TC_info['local_time'] = single_TC_info.GMT_time + single_TC_info.apply(lambda x: timedelta(hours=x.global_lon/15), axis=1)
        single_TC_info['hour_transform'] = single_TC_info.apply(lambda x: x.local_time.hour / 24 * 2 * math.pi, axis=1)
        single_TC_info['hour_sin'] = single_TC_info.hour_transform.apply(lambda x: math.sin(x))    
        single_TC_info['hour_cos'] = single_TC_info.hour_transform.apply(lambda x: math.cos(x))    
        
        local_time_sin = single_TC_info.hour_sin.to_numpy(dtype='float')
        local_time_cos = single_TC_info.hour_cos.to_numpy(dtype='float')
        
        D200 = single_TC_info.D200.to_numpy(dtype='float')
        Vmax = single_TC_info.Vmax.to_numpy(dtype='float')
        POTraw = single_TC_info.POT.to_numpy(dtype='float')
        RHLO = single_TC_info.RHLO.to_numpy(dtype='float')
        SHRD = single_TC_info.SHRD.to_numpy(dtype='float')
        SHTD = single_TC_info.SHTS.to_numpy(dtype='float')
        if region_string[0] == 'SH':
            SHTD = (540. - SHTD) % 360.
        POT = []
        SHR_x = []
        SHR_y = []
        for i in range(len(region_string)):
            POTtmp = Vmax[i] - POTraw[i]
            POTtmp = 170. if POTtmp > 170. else POTtmp
            if POTtmp < 0.:
                if i ==0:
                    POTtmp2 =  Vmax[i+2] - POTraw[i+2]
                    POTtmp1 =  Vmax[i+1] - POTraw[i+1]
                    POTtmp = 2*POTtmp1 - POTtmp2
                elif i == len(region_string)-1:
                    POTtmp2 =  Vmax[i-2] - POTraw[i-2]
                    POTtmp1 =  Vmax[i-1] - POTraw[i-1]
                    POTtmp = 2*POTtmp1 - POTtmp2
                else:                
                    POTtmp = (Vmax[i-1] - POTraw[i-1] + Vmax[i+1] - POTraw[i+1])/2
            POT.append(POTtmp)
            SHR_x.append(SHRD[i]*math.cos(math.radians(SHTD[i])))
            SHR_y.append(SHRD[i]*math.sin(math.radians(SHTD[i])))

        SHRG = single_TC_info.SHRG.to_numpy(dtype='float')
        RSST = single_TC_info.RSST.to_numpy(dtype='float')
        env_feature = [land_dis, region, local_time_sin, local_time_cos, D200, POT, RHLO, SHRD, SHR_x, SHR_y, SHRG, RSST]
        env_feature = np.array(env_feature)
                           
        features = {
            'history_len': _int64_feature(history_len),
            'images': _bytes_feature(np.ndarray.tobytes(single_TC_images)),
            'intensity': _bytes_feature(np.ndarray.tobytes(single_TC_info.Vmax.to_numpy())),
            'frame_ID': _bytes_feature(np.ndarray.tobytes(frame_ID.to_numpy('bytes'))),
            'lon': _bytes_feature(np.ndarray.tobytes(lon)),
            'lat': _bytes_feature(np.ndarray.tobytes(lat)),
            'env_feature': _bytes_feature(np.ndarray.tobytes(env_feature)),
            'SHTD': _bytes_feature(np.ndarray.tobytes(SHTD)),
        }
        return tf.train.Example(features=tf.train.Features(feature=features))
    
    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
        assert(len(image_matrix) == len(info_df))
        for single_TC_images, single_TC_info in zip(image_matrix, info_df):
            example = _encode_tfexample(single_TC_images, single_TC_info, m_map, maptxt)
            serialized = example.SerializeToString()
            writer.write(serialized)


def generate_tfrecord(data_folder):
    file_path = Path(data_folder, 'TCSA_GAN.h5')
    if not file_path.exists():
        print(f'file {file_path} not found! try to download it!')
        download_data(data_folder)
    with h5py.File(file_path, 'r') as hf:
        image_matrix = hf['matrix'][:]
    # collect info from every file in the list
    info_df = pd.read_hdf(file_path, key='info', mode='r')
    image_matrix, info_df = data_cleaning_and_organizing(image_matrix, info_df)

    phase_data = {
        phase: data_split(image_matrix, info_df, phase)
        for phase in ['train', 'valid', 'test']
    }
    del image_matrix, info_df
    
    maptxt = coastline()
    m_map = Basemap(projection='cyl', resolution='i', area_thresh = 5000.)
    
    for phase, (image_matrix, info_df) in phase_data.items():
        image_matrix, info_df = group_by_id(image_matrix, info_df)
        phase_path = Path(data_folder, f'TCSA.tfrecord.{phase}')
        write_tfrecord(image_matrix, info_df, phase_path, m_map, maptxt)


def get_or_generate_tfrecord(data_folder):
    tfrecord_path = {}
    for phase in ['train', 'valid', 'test']:
        phase_path = Path(data_folder, f'TCSA.tfrecord.{phase}')
        if not phase_path.exists():
            print(f'tfrecord {phase_path} not found! try to generate it!')
            generate_tfrecord(data_folder)
        tfrecord_path[phase] = phase_path

    return tfrecord_path
