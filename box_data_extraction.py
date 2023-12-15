from pathlib import Path

from boxsdk import Client, OAuth2
from tqdm import tqdm

DATA_LOC = "CEUS_images_TJU_Stanford_Calgary_essentials_20230901"
TYPE_LOC = "CE_MCv7_ROI200_pickle"
DESTINATION = "HCC_pkl_data"

from logging.handlers import logging

logging.basicConfig(level=logging.CRITICAL)

def format_size(size):
  return f"{round(size/10**9,1)} GB"

def access_folder_by_name(client, name, root=None):
  if root is None: root = client.root_folder().get()
  
  folder = None
  for item in root.get_items():
    if item.name == name: folder = item
  
  if folder is None: raise NameError(f"Name \"{name}\" not found in directory \"{root}\".")
  return folder.get()

def get_folders(client, root=None):
  if root is None: root = client.root_folder().get()
  
  folders = []
  for item in root.get_items():
    if item.type == "folder": folders.append(item.get())
  
  return folders

def get_files(client, root=None):
  if root is None: root = client.root_folder().get()
  
  files = []
  for item in root.get_items():
    if item.type == "file": files.append(item.get())
  
  return files

def download_folder(client, destination, root=None, name=None):
  if root is None: root = client.root_folder().get()
  
  files = get_files(client, root)
  subdirectory = (name, root.name)[name is None]
  for file in tqdm(files, desc=subdirectory):
    directory = f"{destination}/{subdirectory}/"
    Path(directory).mkdir(parents=True, exist_ok=True)
    filename = directory + file.name
    with open(filename, "wb") as stream:
      file.download_to(stream)

def get_authenticated_client():
  client_id = input("Enter client ID: ")
  client_secret = input("Enter client secret: ")
  dev_token = input("Enter refreshed access token: ")
  client = Client(OAuth2(client_id, client_secret, access_token=dev_token))
  
  return client

def collect_patient_data(data_loc=DATA_LOC, type_loc=TYPE_LOC):
  client = get_authenticated_client()

  patient_count = 0
  patient_names = []

  image_data_count = 0
  image_folders = []

  total_size = 0
  data_root = access_folder_by_name(client, data_loc)

  for batch in tqdm(get_folders(client, data_root)):
    for patient in get_folders(client, batch):
      try:
        pkl_folder = access_folder_by_name(client, type_loc, patient)
        patient_names.append(patient.name)
        image_folders.append(pkl_folder)
        for image_data in tqdm(get_files(client, pkl_folder), desc=patient.name):
          image_data_count += 1
          total_size += image_data.size
        patient_count += 1
      except NameError: pass

  print(f"# of Patients: {patient_count}")
  print(f"# of Image Data Files: {image_data_count}")
  print(f"Total Size of Dataset: {format_size(total_size)}")

  return image_folders, patient_names

def extract_patient_data(destination=DESTINATION, data_roots=None, collected_names=None):
  client = get_authenticated_client()

  if len(data_roots) != len(collected_names):
    raise ValueError("Lengths of \"data_roots\" and \"collected_names\" do not match.")
  
  for i, data_root in enumerate(tqdm(data_roots)):
    download_folder(client, destination, data_root, collected_names[i])

if __name__ == "__main__":
  image_folders, patient_names = collect_patient_data()
  extract_patient_data(data_roots=image_folders, collected_names=patient_names)
