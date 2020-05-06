import mimetypes
import os
import tempfile
from pathlib import Path
from zipfile import ZipFile

from lgblkb_tools import logger, Folder
import numpy as np
import pandas as pd
import geopandas as gpd
from lgblkb_tools.common.utils import get_md5
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive

import pysftp


#
# def get_gauth():
#     gauth = GoogleAuth()
#     # Try to load saved client credentials
#     gauth.LoadCredentialsFile("mycreds.txt")
#     if gauth.credentials is None:
#         # Authenticate if they're not there
#         gauth.LocalWebserverAuth()
#     elif gauth.access_token_expired:
#         # Refresh them if expired
#         gauth.Refresh()
#     else:
#         # Initialize the saved creds
#         gauth.Authorize()
#     # Save the current credentials to a file
#     gauth.SaveCredentialsFile("mycreds.txt")
#
#     return gauth
#
#
# class GoogleDriveBackupCreator(object):
#     def __init__(self):
#         self.drive = GoogleDrive(get_gauth())
#
#     def backup(self, directory):
#         dir_name = directory.split("/")[-1]
#         gdrive_folder = self.drive.CreateFile({
#             'title': dir_name,
#             "mimeType": "application/vnd.google-apps.folder"
#         })
#         gdrive_folder.Upload()
#         gdrive_folder_id = gdrive_folder['id']
#         files = os.listdir(directory)
#
#         for file in files:
#             full_name = os.path.join(directory, file)
#             print("Backing up {}".format(full_name))
#             mime_type = mimetypes.guess_type(full_name)[0]
#             if mime_type:
#                 # continue
#                 f = self.drive.CreateFile({
#                     'title': file,
#                     "parents": [{"id": gdrive_folder_id}],
#                     'mimeType': mime_type})
#                 f.SetContentFile(full_name)
#                 f.Upload()
#             else:
#                 if os.path.isdir(full_name):
#                     temp_folder = Folder(tempfile.mkdtemp())
#                     zip_filepath = Folder(full_name).zip(temp_folder[get_md5(full_name) + '.zip'])
#                     logger.debug("zip_filepath: %s", zip_filepath)
#
#                     # zip_filename = "{}.zip".format(file)
#                     # with ZipFile(zip_filename, "w") as zip_obj:
#                     #     for folder, sub_folders, filenames in os.walk(full_name):
#                     #         for filename in filenames:
#                     #             file_path = os.path.join(folder, filename)
#                     #             zip_obj.write(file_path)
#                     f = self.drive.CreateFile({
#                         'title': Path(zip_filepath).name,
#                         "parents": [{"id": gdrive_folder_id}]
#                     })
#                     f.SetContentFile(zip_filepath)
#                     f.Upload()
#                     temp_folder.delete()
#                 else:
#                     print("Mime type for {} could not be determined. Skipping".format(file))
#
#         print("Directory: {} backed up successfully".format(directory))


@logger.trace()
def main():
    # gdrive_backup_creator = GoogleDriveBackupCreator()
    # gdrive_backup_creator.backup('/home/lgblkb/Documents')
    
    with pysftp.Connection(host="94.247.135.91", username="egistic_db",
                           password="UxeiJ5ree2riVoi", ) as sftp:
        sftp.cwd('/nfs/storage/ml_data')
        sftp.put('/home/lgblkb/PycharmProjects/dm_final_project')
    
    # drive = GoogleDrive(get_gauth())
    # directory = '/home/lgblkb/PycharmProjects/dm_final_project'
    # dir_name = directory.split("/")[-1]
    # gdrive_folder = drive.CreateFile({
    #     'title': dir_name,
    #     "mimeType": "application/vnd.google-apps.folder"
    # })
    # gdrive_folder.Upload()
    # gdrive_folder_id = gdrive_folder['id']
    # files = os.listdir(directory)
    #
    # for file in files:
    #     logger.debug("file: %s", file)
    #     full_name = os.path.join(directory, file)
    #     print("Backing up {}".format(full_name))
    #     mime_type = mimetypes.guess_type(full_name)[0]
    #     if mime_type:
    #         f = drive.CreateFile({
    #             'title': file,
    #             "parents": [{"id": gdrive_folder_id}],
    #             'mimeType': mime_type})
    #         f.SetContentFile(full_name)
    #         f.Upload()
    #     else:
    #         if os.path.isdir(full_name):
    #             with ZipFile("{}.zip".format(file), "w") as zip_obj:
    #                 for folder, sub_folders, filenames in os.walk(full_name):
    #                     for filename in filenames:
    #                         file_path = os.path.join(folder, filename)
    #                         zip_obj.write(file_path)
    #                 f = drive.CreateFile({
    #                     'title': "{}".format(zip_obj.filename),
    #                     "parents": [{"id": gdrive_folder_id}]
    #                 })
    #                 f.SetContentFile(zip_obj.open())
    #                 f.Upload()
    #         else:
    #             print("Mime type for {} could not be determined. Skipping".format(file))
    #
    # print("Directory: {} backed up successfully".format(directory))
    
    # drive.CreateFile({'id': textfile['id']}).GetContentFile('eng-dl.txt')
    pass


if __name__ == '__main__':
    main()
