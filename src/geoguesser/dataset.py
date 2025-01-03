# Python Standard Library
import os
from pathlib import Path
import requests
from typing import Callable
from typing import Literal
import zipfile

# Third Party Libraries
from PIL import Image
import tifffile
from torch.utils.data import Dataset
from torchvision import transforms

class EuroSATDataset(Dataset):
  def __init__(self, root_dir, transform: Callable | None = None,):
    self.root_dir = root_dir
    self.transform = transform

    self.observations = list()
    self.classes = os.listdir(root_dir)
    for label in self.classes:
        class_dir = os.path.join(root_dir, label)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if any(img_path.lower().endswith(ext) for ext in ['.jpg', '.tif']):
                self.observations.append({"image_path": img_path, "label": label})

  def __len__(self):
    return len(self.observations)

  def __getitem__(self, idx):
    image_path = self.observations[idx]["image_path"]
    _ , ext = os.path.splitext(image_path)
    if ext == '.jpg':
      image = Image.open(image_path)
      image = transforms.ToTensor()(image)
    else:
      image = tifffile.imread(image_path)
      image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

    # Normalize each channel
    for channel in range(image.shape[0]):
      image[channel] = (image[channel] - image[channel].min()) / (image[channel].max() - image[channel].min())

    label = self.observations[idx]["label"]
    label_num = self.classes.index(label)

    if self.transform:
      image = self.transform(image)

    return {"image": image, "label": label_num}

  @classmethod
  def download(cls, dataset: Literal['RGB', 'MS'] = 'MS', download_dir: Path = Path('/workspace/code/data')):
    if dataset == 'RGB':
      url = 'https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1'
    else:
      # Multi-spectral dataset selected
      url = 'https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1'
    
    save_path = download_dir / f'{dataset}.zip'
    extract_path = download_dir / f'{dataset}/'

    os.makedirs(download_dir, exist_ok=True)

    if save_path.exists():
      # Zip file already downloaded and saved on local machine
      print(f"Dataset already saved in {save_path} - not overwriting")
    else:
      # Zip file is not downloaded yet
      print(f"Starting download from {url}")
      response = requests.get(url, allow_redirects=True)
      with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
          file.write(chunk)
      print(f"Dataset downloaded and saved to {save_path}")

    if extract_path.exists():
      # Zip file already extracted to local storage
      print(f"Dataset already extracted to {extract_path} - not overwriting")
    else:
      # Zip file not extracted yet
      print(f"Starting extracting dataset to {extract_path}")
      with zipfile.ZipFile(save_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
      print(f"Dataset extracted to {extract_path}")

    if dataset == 'RGB':
      return cls(root_dir=extract_path / 'EuroSAT_RGB/')
    # Else: Multi-spectral dataset selected
    return cls(root_dir=extract_path / 'EuroSAT_MS/')

  @classmethod
  def from_subset(cls, original, indices: list[int], transform: Callable | None = None):
      # Create a new instance with the same properties as the original
      subset = cls(root_dir=original.root_dir, transform=original.transform if transform is None else transform,)

      # Filter the observations based on the subset indices
      subset.observations = [original.observations[i] for i in indices]
      subset.classes = original.classes  # Keep class list consistent

      print(f'Created a subset with {len(subset.observations):_} of {len(original.observations):_} images')

      return subset