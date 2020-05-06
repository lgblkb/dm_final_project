import datetime
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns

from diskcache import Cache
from lgblkb_tools import logger, Folder
import numpy as np
import pandas as pd
import geopandas as gpd
from pandas import SparseArray

from db_utils import engine
from scipy.interpolate import griddata

this_folder = Folder(__file__)
data_folder = this_folder.parent()['data']
cache = Cache(this_folder['cache'])

remote_sensing_dhf_path = data_folder['remote_sensing_data.h5']

sns.set(rc={'figure.figsize': (11, 4)})


@cache.memoize(name='get_target_fields')
def get_target_fields():
    target_fields = gpd.read_postgis('select * from target_fields;', engine, geom_col='geometry')
    logger.debug("target_fields:\n%s", target_fields)
    target_fields = target_fields[target_fields.caption.map(lambda caption: caption is None or ('пар' not in caption))]
    # logger.debug("target_fields.head(20).to_string():\n%s", target_fields.head(20).to_string())
    target_fields.drop(columns=['class_number', 'flregion'], inplace=True)
    logger.debug("target_fields.head(20).to_string():\n%s", target_fields.head(20).to_string())
    logger.debug("target_fields.shape: %s", target_fields.shape)
    # for column in target_fields.columns:
    #     unique_vals = target_fields[column].unique()
    #     logger.debug("%s: %s", column, len(unique_vals))
    #     if len(unique_vals) < 20:
    #         for unique_val in unique_vals:
    #             logger.debug("unique_val: %s", unique_val)
    return target_fields


@cache.memoize(name='get_fields_data_2')
def get_fields_data():
    fields_data = gpd.read_postgis('select * from geometry_yield_by_year;', engine, geom_col='geometry')
    logger.debug("fields_data:\n%s", fields_data.head(20).to_string())
    fields_data = fields_data[(fields_data.year >= 2016) & (fields_data.year <= 2019)].copy()
    fields_data = fields_data[(fields_data.cult == 'Пшеница мягкая яровая (Triticum aestivum L)') |
                              (fields_data.cult == 'Пшеница твердая яровая (Triticum durum Desf)')].copy()
    fields_data: pd.DataFrame = fields_data[(fields_data.region == 'Костанайская область') |
                                            (fields_data.region == 'Северо-Казахстанская область') |
                                            (fields_data.region == 'Акмолинская область')
                                            ].copy()
    # describe(fields_data)
    fields_data.dropna(subset=['yield'], inplace=True)
    logger.debug("fields_data:\n%s", fields_data)
    return fields_data


def describe(df, max_unique_vals=20):
    logger.debug("df.shape: %s", df.shape)
    for column in df.columns:
        unique_vals = df[column].unique()
        logger.debug("%s: %s", column, len(unique_vals))
        if len(unique_vals) < max_unique_vals:
            for unique_val in unique_vals:
                logger.debug("unique_val: %s", unique_val)


def get_ml_veg_ind_results():
    remote_sensing_data = pd.read_sql_table('ml_veg_ind_results', engine, index_col='id', parse_dates=True, )
    logger.debug("remote_sensing_data.shape: %s", remote_sensing_data.shape)
    return remote_sensing_data


def process_remote_sensing_item(row):
    # logger.debug("row:\n%s", row.to_string())
    for index_name in ['ndvi', 'ndmi', 'clgreen', 'gndvi']:
        row["_".join([index_name, 'min'])] = row[index_name]['min']
        row["_".join([index_name, 'mean'])] = row[index_name]['mean']
        row["_".join([index_name, 'max'])] = row[index_name]['max']
    row.actual_date = str(row.actual_date.date())
    
    # logger.debug("row:\n%s", row.to_string())
    return row


def process_remote_sensing_fields(row):
    # logger.debug("row:\n%s", row)
    row.field = row.results_dir.split('/')[-1]
    # logger.debug("row:\n%s", row.to_string())
    # raise NotImplementedError
    return row


veg_inds_store = pd.HDFStore(remote_sensing_dhf_path)
fields_veg_data_store = pd.HDFStore(data_folder['fields_veg_data.h5'])


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(sequence.shape[0]):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > sequence.shape[0] - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


@logger.trace()
def main():
    X = list()
    y = list()
    
    for field_key in fields_veg_data_store.keys():
        # field_key = 'field_000115c74eeb8e1a44e5a4b1a606b62f'
        logger.debug("field_key: %s", field_key)
        field_df: pd.DataFrame = fields_veg_data_store[field_key].drop(
            columns=['id', 'divided_cadastre_user_id', 'results_dir', 'field'])
        
        field_df.sort_values('actual_date', inplace=True)
        field_df.actual_date = pd.DatetimeIndex(field_df.actual_date, )
        field_df.set_index('actual_date', inplace=True)
        field_df = field_df.drop(columns=field_df.columns.drop([c for c in field_df.columns if 'mean' in c]))
        
        # field_id = field_df['divided_cadastre_user_id'].iloc[0]
        monthly_data: pd.DataFrame = field_df.resample('M', ).mean()
        # logger.debug("field_df:\n%s", field_df.to_string())
        # logger.debug("monthly_data:\n%s", monthly_data.to_string())
        # logger.debug("monthly_data.shape: %s", monthly_data.shape)
        # return
        
        for year in [2016, 2017, 2018, 2019]:
            # try:
            # logger.info("year: %s", year)
            annual_data: pd.DataFrame = monthly_data.loc[f'{date(year, 4, 1)}':f'{datetime.date(year, 11, 1)}'].copy()
            # logger.debug("annual_data:\n%s", annual_data)
            if annual_data.dropna().shape[0] < 6:
                # logger.debug("annual_data.dropna().shape[0]: %s", annual_data.dropna().shape[0])
                continue
            sequence = annual_data.dropna().values
            # logger.debug("sequence: %s", sequence)
            X_i, y_i = split_sequence(sequence, 4)
            X.append(X_i)
            y.append(y_i)
    
    logger.debug("len(y): %s", len(y))
    X = np.concatenate(X)
    y = np.concatenate(y)
    np.save(data_folder['X.npy'], X)
    np.save(data_folder['y.npy'], y)
    
    return
    df = veg_inds_store['remote_sensing_data'] = pd.read_sql_table('remote_sensing_data_updated', engine)
    df['field'] = df.results_dir.map(lambda rd: rd.split('/')[-1])
    veg_inds_store['remote_sensing_data'] = df
    
    # df: pd.DataFrame = veg_inds_store['remote_sensing_data']
    logger.debug("df:\n%s", df.head(20).to_string())
    logger.debug("df.shape: %s", df.shape)
    for field, group in df.groupby('field'):
        logger.debug("field: %s", field)
        fields_veg_data_store[f'field_{field}'] = group
    
    veg_inds_store.close()
    fields_veg_data_store.close()
    
    # temporal_range = group.shape[0]
    # logger.debug("temporal_range: %s", temporal_range)
    # temporal_ranges.append(temporal_range)
    # plt.hist(temporal_ranges, bins=100, range=(0, 150))
    # plt.show()
    
    # df['field'] = df.results_dir.map(lambda rd: rd.split('/')[-1])
    # # df = df.apply(process_remote_sensing_fields, axis=1)
    # logger.debug("df:\n%s", df.head(20).to_string())
    # veg_inds_store['remote_sensing_data'] = df
    # for column in df.columns:
    #     logger.debug("column: %s", column)
    
    # df.drop(columns=['ndsi', 'temperature_dir', 'is_layer_created'], inplace=True)
    # for index_name in ['ndvi', 'ndmi', 'clgreen', 'gndvi']:
    #     logger.debug("df.shape: %s", df.shape)
    #     df["_".join([index_name, 'min'])] = None
    #     df["_".join([index_name, 'mean'])] = None
    #     df["_".join([index_name, 'max'])] = None
    # df = df.apply(process_remote_sensing_item, axis=1)
    # df.drop(columns=['ndvi', 'ndmi', 'clgreen', 'gndvi'], inplace=True)
    # veg_inds_store['remote_sensing_data_updated'] = df
    
    pass


if __name__ == '__main__':
    main()
