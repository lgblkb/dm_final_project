import datetime
import os
import shutil
import subprocess
import tempfile
import time
import re
# import grib
import json
from pathlib import Path

from box import Box
from lgblkb_tools import Folder, logger
from lgblkb_tools.common.utils import ParallelTasker


def get_meteo(start_date, end_date, directory, param, levels):
    """Downloads GFS meteo data

    Parameters
    ----------
    start_date : str
        Download meteo data starting from this date. Format is '20190101'.
    end_date : str
        Download meteo data until this date. Format is '20190101'.
    directory : str
        Download meteo data to this path.
    param : str
        Which meteo data's parameter to include. Only ONE parameter!
    levels : dict
        Which levels and their values for the ONLY param to download.
        See meteo_archive_info.json for correct dictionary format.

    Returns
    -------
    str
        path to the downloaded file
    """
    print("start_date; {}".format(start_date))
    print("end_date; {}".format(end_date))
    print("directory; {}".format(directory))
    print("param; {}".format(param))
    print("levels {}".format(levels))
    # levels = {
    #     "SFC": {
    #         "descr": "Ground or water surface",
    #         "values": [
    #             "0"
    #         ]
    #     }
    # }
    
    formatted_levels = ""
    
    # Constructing a subset request
    with open('ds084.1_control_file', 'w') as control_file:
        print("dataset=ds084.1", file=control_file)
        print("date={}0000/to/{}0000".format(start_date, end_date), file=control_file)
        print("datetype=init", file=control_file)
        print("param={}".format(param), file=control_file)
        
        for level in levels.keys():
            formatted_levels += level + ":"
            formatted_levels += '/'.join(map(str, levels[level]["values"]))
            formatted_levels += ";"
        
        print("level={}".format(formatted_levels), file=control_file)
        print("#oformat=netCDF", file=control_file)
        print("nlat=60.7", file=control_file)
        print("slat=45.2", file=control_file)
        print("wlon=54.7", file=control_file)
        print("elon=78.46", file=control_file)
        print("product=6-hour Forecast/3-hour Forecast/18-hour Forecast/Analysis", file=control_file)
        print("targetdir=/glade/scratch", file=control_file)
    # return
    output1 = subprocess.Popen(['python3', 'rdams-client.py', '-submit', 'ds084.1_control_file'],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output1.wait()
    stdout1, stderr1 = output1.communicate()
    if stderr1:
        raise ValueError(stderr1)
    stdout1 = stdout1.decode("utf-8")
    
    # print(stdout1)
    start_indx = stdout1.find("Request Index") + 14
    rqst_indx = stdout1[start_indx:start_indx + 6]
    
    if re.match('^[0-9]{6}$', rqst_indx) is None:
        # raise ValueError(stdout1)
        raise ValueError('Request Index must be entirely numeric. But it is: {}'.format(rqst_indx))
    
    file_ready = False
    
    while not file_ready:
        time.sleep(30)
        output = subprocess.Popen(['python3', 'rdams-client.py', '-get_status', rqst_indx], stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT)
        output.wait()
        stdout, stderr = output.communicate()
        if stderr:
            raise ValueError(stderr)
        stdout = stdout.decode('utf-8')
        
        if "Q - building" in stdout:
            print("Building data... for the request {}".format(rqst_indx))
        elif "Q -  queued" in stdout:
            print("Request in queue... for the request {}".format(rqst_indx))
        elif "O - Online" in stdout:
            file_ready = True
            print("Downloading data... for the request {}".format(rqst_indx))
        elif "E - Error" in stdout:
            raise ValueError(
                "The data server could not prepare the requested meteo data. The -get_status output: \n".format(stdout))
            # raise ValueError(stdout)
        output.stdout.close()
    
    output1 = subprocess.Popen(['python3', 'rdams-client.py', '-download', rqst_indx, directory],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output1.wait()
    stdout1, stderr1 = output1.communicate()
    if stderr1:
        raise ValueError(stderr1)
    stdout1 = stdout1.decode('utf-8')
    
    # print("STDOUT1 : %s", stdout1)
    
    path_indx = stdout1.find("Request to ") + 12
    path_end_indx = stdout1.find(" directory.")
    data_path = stdout1[path_indx:path_end_indx - 1]
    
    print("DATA_PATH: %s", data_path)
    
    print("Successfully downloaded meteo data into this directory: {}".format(data_path))
    print("Time period is from {} to {} with these parameter {} for these levels: {}:\n".format(start_date, end_date,
                                                                                                param, levels))
    # if param == "TMP":
    #     grib.convert_to_raster(data_path)


this_folder = Folder(__file__)
data_folder = this_folder.parent()['data']

archive_info = Box.from_json(filename=this_folder['meteo_archive_info.json'])


# berths_folder = data_folder['berths']


# levels = {
#     "SFC": {
#         "descr": "Ground or water surface",
#         "values": [
#             "0"
#         ]
#     }
# }

@logger.trace()
def download_meteo_data(param, param_levels, daterange, level_values=()):
    # this_berth = Folder(berths_folder.get_filepath(param=param, param_levels="_AND_".join(map(str, param_levels))))
    # os.chdir(this_berth)
    with tempfile.TemporaryDirectory() as tempdir:
        os.chdir(tempdir)
        shutil.copyfile(this_folder['rdamspw.txt'], Folder(tempdir)['rdamspw.txt'])
        shutil.copyfile(this_folder['rdams-client.py'], Folder(tempdir)['rdams-client.py'])
        levels = Box()
        for param_level in param_levels:
            levels[param_level] = archive_info[param].levels[param_level]
            if level_values: levels[param_level]['values'] = level_values
        start_date, end_date = daterange
        kwargs = {"start_date": start_date,
                  "end_date": end_date,
                  "directory": this_folder['weather_data'][param].get_filepath(start_date, 'QWE', end_date),
                  "param": param,
                  "levels": levels
                  }
        get_meteo(**kwargs)


@logger.trace()
def main():
    inp_data = Box(param=list(), param_levels=list(), level_values=list())
    
    def add_input_data(param, levels, values=()):
        inp_data.param.append(param)
        inp_data.param_levels.append(levels)
        inp_data.level_values.append(values)
    
    start_date = datetime.date(2016, 1, 1).strftime("%Y%m%d")
    end_date = datetime.date(2020, 1, 1).strftime("%Y%m%d")
    add_input_data('R H', ['HTGL'], )
    add_input_data('WEASD', ['SFC'], )
    add_input_data('PRATE', ['SFC'], )
    add_input_data('GUST', ['SFC'], )
    add_input_data('T CDC', ['ISBL'], values=[650])
    add_input_data('SUNSD', ['SFC'], )
    add_input_data('SOILW', ['DBLL'], )
    add_input_data('TSOIL', ['DBLL'], )
    add_input_data('FLDCP', ['SFC'], )
    add_input_data('FLDCP', ['SFC'], )
    add_input_data('HINDEX', ['SFC'], )
    add_input_data('HGT', ['SFC'], )
    
    ParallelTasker(download_meteo_data, daterange=(start_date, end_date)) \
        .set_run_params(**inp_data).run()
    
    pass


if __name__ == '__main__':
    main()
