# Python Standard Library
import os
from pathlib import Path
import requests
from typing import Callable
import zipfile

# Third Party Libraries
from PIL import Image
from torch.utils.data import Dataset

class EuroSATDataset(Dataset):
  @staticmethod
  def download(download_dir: Path = Path('/workspace/code/data')):
    url = 'https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1'
    save_path = download_dir / 'rgb.zip'
    extract_path = download_dir / 'rgb/'

    os.makedirs(download_dir, exist_ok=True)

    if save_path.exists():
      # Zip file already downloaded and saved on local machine
      print(f"File {save_path} exists already - not overwriting")
    else:
      # Zip file is not downloaded yet
      print(f"Starting download from {url}")
      response = requests.get(url, allow_redirects=True)
      with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                _ = file.write(chunk)
      print(f"File downloaded and saved to {save_path}")

    if extract_path.exists():
      # Zip file already extracted to local storage
      print(f"File {extract_path} exists already - not overwriting")
    else:
      # Zip file not extracted yet
      print(f"Starting extracting {extract_path}")
      with zipfile.ZipFile(save_path, "r") as zip_ref:
          zip_ref.extractall(extract_path)
      print(f"File extracted to {extract_path}")

  def __init__(self, root_dir, transform: Callable | None = None,):
    self.root_dir = root_dir
    self.transform = transform

    self.observations = list()
    self.classes = os.listdir(root_dir)
    for label in self.classes:
        class_dir = os.path.join(root_dir, label)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if img_path.lower().endswith('.jpg'):
                self.observations.append({"image_path": img_path, "label": label})

  def __len__(self):
    return len(self.observations)

  def __getitem__(self, idx):
    image_path = self.observations[idx]["image_path"]
    image = Image.open(image_path)
    label = self.observations[idx]["label"]
    label_num = self.classes.index(label)

    if self.transform:
      image = self.transform(image)

    return {"image": image, "label": label_num}

  @classmethod
  def from_subset(cls, original, indices: list[int], transform: Callable | None = None):
      # Create a new instance with the same properties as the original
      subset = cls(root_dir=original.root_dir, transform=original.transform if transform is None else transform,)

      # Filter the observations based on the subset indices
      subset.observations = [original.observations[i] for i in indices]
      subset.classes = original.classes  # Keep class list consistent

      print(f'Created a subset with {len(subset.observations):_} of {len(original.observations):_} images')

      return subset