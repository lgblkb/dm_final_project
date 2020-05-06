import matplotlib.pyplot as plt
from collections import defaultdict
from functools import reduce

from odo import odo
import logging

from dateutil.parser import parse
import datetime
import missingno
from box import Box
from lgblkb_tools import logger, Folder
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import tarfile
from diskcache import Cache
from lgblkb_tools.common.utils import get_md5
from lgblkb_tools.pathify import get_name
import seaborn as sns

this_folder = Folder(__file__)
data_folder = this_folder.parent(2)['data']
weather_data_folder = data_folder['weather_data']

cache = Cache(data_folder['cache'], size_limit=2 ** 36)


def untar(tar_filepath, out_filepath):
    tar = tarfile.open(tar_filepath)
    tar.extractall(out_filepath)
    tar.close()


def untar_everything():
    tar_filepaths = weather_data_folder.glob_search('**/*.tar')
    for tar_filepath in tar_filepaths:
        untar(tar_filepath, weather_data_folder['untarred'])


# logger.setLevel(logging.WARNING)


def read_grib_file(target_grib_filepath):
    # logger.debug("target_grib_filepath: %s", target_grib_filepath)
    ds = xr.open_dataset(target_grib_filepath, engine='cfgrib')
    df = ds.to_dataframe()
    # logger.debug("df input:\n%s", df.head(20).to_string())
    
    time = df.time.values.copy()
    # logger.info("time.shape: %s", time.shape)
    df.drop(columns=['valid_time', 'step', 'time'], inplace=True)
    
    # counts = df.count()
    # logger.debug("counts:\n%s", counts)
    target_vars = list()
    for var in sorted(ds.variables):
        # logger.info("%s.shape: %s", var, ds.variables[var].values.shape)
        if len(ds.variables[var].values.shape) > 1:
            target_vars.append(var)
    
    # logger.debug("df:\n%s", df.head().to_string())
    # logger.debug("df.shape: %s", df.shape)
    # logger.debug("df.step.unique():\n%s", df.step.unique())
    # logger.debug("df.heightAboveGround.unique(): %s", df.heightAboveGround.unique())
    # logger.debug("df.index:\n%s", df.index.names)
    # logger.debug("len(df.index.names): %s", len(df.index.names))
    if len(df.index.names) > 2:
        res = df.unstack(0)
        res.columns = ['_'.join(map(str, col)).strip() for col in res.columns.values]
        
        # logger.debug("res:\n%s", res.head(20).to_string())
        # logger.debug("res.shape: %s", res.shape)
        df = res
    else:
        pass
    
    for column in df.columns:
        val_counts = df[column].value_counts()
        # logger.debug("val_counts:\n%s", val_counts)
        # logger.info("len(val_counts): %s", len(val_counts))
        if len(val_counts) == 1:
            if column in target_vars: continue
            df.drop(columns=[column], inplace=True)
    # logger.info("df.shape: %s", df.shape)
    # if time.shape[0] == df.shape[0]:
    #     df['time'] = time.dt.date
    # else:
    ratio = df.shape[0] / time.shape[0]
    # logger.debug("ratio: %s", ratio)
    # logger.debug("int(time.shape[0] * ratio): %s", int(time.shape[0] * ratio))
    df['time'] = time[:int(time.shape[0] * ratio)]
    df.time = df.time.dt.date.map(str)
    # logger.debug("df output:\n%s", df.head(20).to_string())
    
    # logger.debug("var: %s", var)
    #     res = ds.to_array()
    #     logger.debug("res.shape: %s",res.shape)
    #     # logger.debug("res:\n%s", res)
    #
    #     # logger.debug("ds.variables[var].values: %s",ds.variables[var].values)
    #     # logger.debug("ds.variables[var].values.shape: %s", ds.variables[var].values.shape)
    # input('Next')
    out = df.reset_index()
    # logger.debug("out:\n%s", out.head(20).to_string())
    # logger.debug("df:\n%s", df.head(20).to_string())
    
    # input('')
    return out


weather_data_hdf_path = data_folder['weather_data.h5']


def get_grib_data():
    grib_folder = weather_data_folder['untarred']
    target_grib_filepaths = grib_folder.glob_search('*f000*')
    store = pd.HDFStore(weather_data_hdf_path)
    logger.debug("len(target_grib_filepaths): %s", len(target_grib_filepaths))
    dfs = list()
    for i, target_grib_filepath in enumerate(target_grib_filepaths):
        if target_grib_filepath.endswith('.idx'): continue
        if i % 100 == 0:
            logger.debug("i: %s", i)
        if len(dfs) == 10:
            return dfs
        # if i != 3: continue
        # try:
        df = read_grib_file(target_grib_filepath)
        key = 'data_' + get_md5(get_name(target_grib_filepath))
        # logger.debug("key: %s", key)
        store[key] = df
    
    store.close()


@logger.trace()
def main():
    weather_data_store = pd.HDFStore(weather_data_hdf_path)
    combined_df = pd.DataFrame()
    structured_data = defaultdict(list)
    df_keys = [
        'sdwe',
        'fldcp',
        'r2',
        'sp',
        'SUNSD',
        'soilw_0_soilw_1',
        'gust',
        'hindex',
        'orog',
    ]
    for i, key in enumerate(weather_data_store.keys()[:200]):
        df: pd.DataFrame = weather_data_store[key]
        # logger.debug("df:\n%s", df)
        df.time = pd.DatetimeIndex(df.time.map(lambda str_time: parse(str_time)))
        df.set_index(['time'], inplace=True)
        # logger.debug("df:\n%s", df)
        # logger.debug("df.index: %s", df.index)
        
        # return
        # df_key = "_".join(df.columns.drop(['latitude', 'longitude']).tolist())
        # delim = '_QWE_'
        df_key = '_'.join(df.columns.tolist())
        # df_keys.add(df_key)
        structured_data[df_key].append(df)
        # if len(structured_data.get(df_key, [])) == 2:
        #     hm_dfs = structured_data[df_key]
        #     for hm_df in hm_dfs:
        #         logger.debug("hm_df:\n%s", hm_df)
        
        # df_merged = reduce(lambda left, right: pd.merge(left, right, on=['latitude', 'longitude', 'time'],
        #                                                 how='outer'),
        #                    structured_data[df_key])
        # df_merged = pd.concat(hm_dfs)
        # logger.debug("df_merged:\n%s", df_merged.head(20).to_string())
        # logger.debug("df_merged.shape: %s", df_merged.shape)
        # return
        # return
        # logger.debug("df:\n%s", df.head().to_string())
        # if combined_df.empty:
        #     combined_df = df
        # else:
        #     combined_df = pd.merge(combined_df, df, how='outer', on=['latitude', 'longitude'],
        #                            left_index=True, right_index=True)
        # # logger.debug("df:\n%s", df)
        # if len(gusts) % 100 == 0:
        #     logger.info("i: %s", i)
        # logger.debug("combined_df:\n%s", combined_df.head(200).to_string())
        # logger.debug("combined_df.shape: %s", combined_df.shape)
    # for df_key in df_keys:
    #     logger.debug("df_key: %s", df_key)
    # return
    themed_dfs = list()
    for df_key, dfs in structured_data.items():
        # df_merged = reduce(lambda left, right: pd.merge(left, right, on=['latitude', 'longitude'],
        #                                                 how='outer', left_index=True, right_index=True), dfs)
        themed_df = pd.concat(dfs)
        # logger.debug("themed_df:\n%s", themed_df.head(100).to_string())
        
        # return
        themed_dfs.append(themed_df)
    
    combined_df = pd.concat(themed_dfs)
    logger.debug("combined_df:\n%s", combined_df)
    for column in combined_df.columns:
        logger.debug("column: %s", column)
    logger.debug("combined_df.shape: %s", combined_df.shape)
    logger.debug("combined_df.head(200).to_string():\n%s", combined_df.head(200).to_string())
    logger.debug("combined_df.info():\n%s", combined_df.info())
    sns.heatmap(combined_df.isnull(), cbar=False)
    plt.show()
    return
    
    ds = xr.open_dataset(grib_filename, engine='cfgrib')
    # logger.debug("ds:\n%s", ds)
    logger.debug("ds.variables:\n%s", sorted(ds.variables))
    data = Box()
    for var in sorted(ds.variables):
        logger.debug("var: %s", var)
        var_data = ds.variables[var]
        data[var] = var_data.values
        logger.debug("var_data.values.shape: %s", var_data.values.shape)
        # logger.debug("var_data:\n%s", var_data)
    return
    df = pd.DataFrame(data)
    logger.debug("df:\n%s", df)
    
    # with GribFile(grib_filename) as grib:
    #     logger.debug("len(grib): %s", len(grib))
    #     for msg in grib:
    #         logger.debug("msg: %s", msg)
    #         return
    
    # f = open('myfields.grib', 'rb')
    pass


if __name__ == '__main__':
    main()
