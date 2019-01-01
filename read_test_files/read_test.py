import numpy as np
import pandas as pd
import os.path
import time

COLUMN_TO_TYPE = {
    'object_id': np.int32,
    'mjd': np.float32,
    'passband': np.int8,
    'flux': np.float32,
    'flux_err': np.float32,
    'detected': np.int8
}

part1_directory = r'test_set_columns_part_1'
part2_directory = r'test_set_columns_part_2'

COLUMN_TO_FOLDER = {
    'object_id': part2_directory,
    'mjd': part2_directory,
    'passband': part2_directory,
    'flux': part1_directory,
    'flux_err': part1_directory,
    'detected': part1_directory
}


def init_reading():
    info = {}
    object_range_file_path = os.path.join(COLUMN_TO_FOLDER['object_id'], 'object_id_range.h5')
    print('reading {}'.format(object_range_file_path))
    object_id_to_range = pd.read_hdf(object_range_file_path, 'data')
    info['object_id_to_range'] = object_id_to_range
    id_to_range = object_id_to_range.set_index('object_id')
    info['object_id_start'] = id_to_range['start'].to_dict()
    info['object_id_end'] = id_to_range['end'].to_dict()

    records_number = object_id_to_range['end'].max()

    mmaps = {}
    for column, dtype in COLUMN_TO_TYPE.items():
        directory = COLUMN_TO_FOLDER[column]
        file_path = os.path.join(directory, 'test_set_{}.bin'.format(column))
        mmap = np.memmap(file_path, dtype=COLUMN_TO_TYPE[column], mode='r', shape=(records_number,))
        mmaps[column] = mmap

    info['mmaps'] = mmaps

    return info


def read_object_info(info, object_id, as_pandas=True, columns=None):
    start = info['object_id_start'][object_id]
    end = info['object_id_end'][object_id]

    data = read_object_by_index_range(info, start, end, as_pandas, columns)
    return data


def read_object_by_index_range(info, start, end, as_pandas=True, columns=None):
    data = {}
    for column, mmap in info['mmaps'].items():
        if columns is None or column in columns:
            data[column] = mmap[start: end]

    if as_pandas:
        data = pd.DataFrame(data)

    return data


def get_chunks(info, chunk_size=1000):
    object_id_to_range = info['object_id_to_range']
    end_of_file_offset = object_id_to_range['end'].max()
    start_offsets = object_id_to_range['start'].values[::chunk_size]
    end_offsets = object_id_to_range['end'].values[(chunk_size - 1)::chunk_size]

    end_offsets = list(end_offsets) + [end_of_file_offset]

    chunks = pd.DataFrame({'start': start_offsets, 'end': end_offsets})
    chunks = chunks.values.tolist()

    return chunks