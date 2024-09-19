import os
import json
import numpy as np
from skimage import io
from skimage.transform import resize
import argparse
join = os.path.join
from tqdm import tqdm
from PIL import Image
join = os.path.join

def preprocessing(img_dir: str):
    fixed_size = (360,360)
    folder_name = os.path.basename(img_dir)
    """Preprocess all images to have the same file format (.png) and shape (360,360) as 
    training images"""
    save_dir = join(os.path.dirname(img_dir), 'preprocessed_'+folder_name)
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    parent_folder = os.path.dirname(img_dir)
    json_dict = {}
    count = 0
    for img_name in sorted(tqdm(os.listdir(img_dir))):
      img = np.uint8(io.imread(join(img_dir, img_name)))
      img_id = img_name.split(".")[0]
      json_dict[img_id] = img.shape[:2]
      img = resize(img, fixed_size, order=0, preserve_range=True)
      im = Image.fromarray(img)
      if img.shape[-1] == 4:
        im = im.convert('RGB')
      assert np.array(im).shape == (360,360,3)
      
      im.save(join(save_dir, img_id+'_0000.png'))

    with open(join(os.path.dirname(img_dir), folder_name +".json"), 'w') as outfile:
      json.dump(json_dict, outfile)
    print(json_dict)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--img_dir",
    type=str,
    default="/content/drive/MyDrive/nnunet/Original_data/imagesTs-External",
    help="path to directory of input images"
  )
  args = parser.parse_args()
  preprocessing(args.img_dir)