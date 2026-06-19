import os
from os import listdir
from os.path import isfile, join
import pathlib
from src.definitions import ASSETS_PATH


class Files:
  def __init__(self):
    self.assets_folder = pathlib.Path(ASSETS_PATH)

  @property
  def ASSETS_PATH(self):
    return self.assets_folder

  def get_assets_folder_path(self):
    return self.assets_folder

  def join(self, *p):
    return join(*p)

  def file_name(self, file_paths: str):
    return os.path.splitext(file_paths)[0]

  def file_extension(self, file_paths: str):
    return os.path.splitext(file_paths)[1]

  def get_only_folders(self, path):
    return [f for f in listdir(path) if isfile(join(path, f)) is False]

  def get_only_files(self, path):
    return [f for f in listdir(path) if isfile(join(path, f)) and f != '.DS_Store']

  def create_folder(self, directory: str):
    if not os.path.exists(directory):
      os.makedirs(directory)

  def remove(self, path):
    os.remove(path)

  def is_exist(self, path):
    return os.path.exists(path)
