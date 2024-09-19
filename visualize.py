import numpy as np 
import matplotlib.pyplot as plt
import argparse
from PIL import Image
import os
from skimage import io
from nnunetv2.paths import nnUNet_raw, nnUNet_results
from tqdm import tqdm
join = os.path.join

def show_mask(mask, ax):
    masked1 = np.ma.masked_where(mask == 1, mask)
    masked2 = np.ma.masked_where(mask == 2, mask)
    h, w = mask.shape[-2:]
    color1 = np.array([0, 0, 255, 0.3]).reshape(1, 1, -1)
    color2 = np.array([0, 255, 0, 0.5]).reshape(1, 1, -1)
    mask_image = mask.reshape(h, w, 1) * color1
    ax.imshow(mask_image)

def visualize(img_path: str,
              label_path: str,
              save_path: str):
    os.makedirs(save_path, exist_ok=True)
    name_ext_dict = {}
    for filename in tqdm(sorted(os.listdir(img_path))):
      name, ext = os.path.splitext(filename)
      name_ext_dict[name] = ext      
    print(list(name_ext_dict.keys())[0])
    for mask_name in tqdm(sorted(os.listdir(label_path))):
      img_name = mask_name.split('.')[0][:-6] + name_ext_dict[mask_name.split('.')[0][:-6]]
      img = np.array(Image.open(join(img_path, img_name)), dtype=np.uint8)
      mask = np.uint8(io.imread(join(label_path, mask_name)))
      _, axs = plt.subplots(1, 2, figsize=(10, 10))
      axs[0].imshow(img)
      show_mask(mask, axs[0])
      axs[0].axis("off")

      axs[1].imshow(mask)
      axs[1].axis("off")
      plt.subplots_adjust(wspace=0.01, hspace=0)
      plt.savefig(join(save_path, mask_name), bbox_inches="tight", dpi=300)
      plt.close()   
      

if __name__ == "__main__":
  internal_test_dir = '/content/drive/MyDrive/nnunet/Original_data/imagesTs-Internal/'
  external_test_dir = '/content/drive/MyDrive/nnunet/Original_data/imagesTs-External/'
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--img_path",
    type=str,
    default=external_test_dir,
  )
  parser.add_argument(
    "--label_path",
    type=str,
    default=os.path.join(nnUNet_results, "Dataset019_Hema/seg-External-Trans/postprocessed/"),
  )
  parser.add_argument(
    "--save_path",
    type=str,
    default=os.path.join(nnUNet_results, "Dataset019_Hema/visualized-seg-External-Trans/"),
  )

  args = parser.parse_args()
  visualize(**vars(args))  
