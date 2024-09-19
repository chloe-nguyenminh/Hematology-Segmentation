from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import os
from PIL import Image
from multiprocessing import freeze_support
from tqdm import tqdm
import numpy as np
import json
from skimage import io
from skimage.transform import resize
import argparse


join = os.path.join

def postprocessing(img_dir: str,
                    label_dir: str):
    """Resize the segmentation mask to match the original image's dimension"""
    save_dir = join(label_dir, 'postprocessed')
    os.makedirs(save_dir, exist_ok=True)
    json_dir = join(os.path.dirname(img_dir), os.path.basename(img_dir).split(".")[0] + '.json')
    with open(json_dir) as json_file:
      data_dict = json.load(json_file)
      for img_name in sorted(tqdm(os.listdir(img_dir))):
        if img_name.endswith('png'):
          np_label = np.uint8(io.imread(join(label_dir, img_name)))
          ori_size = data_dict[img_name.split(".")[0]]
          np_label = resize(np_label, ori_size, order=0, preserve_range=True)
          io.imsave(join(save_dir, img_name.split('.')[0]+'_label.png'), np_label)

def inference(img_dir, predictor, output_path):
    """Perform inference using the nnUNet predictor."""
    internal_test_outputs = []
    internal_test_inputs = []
    for fn in sorted(os.listdir(img_dir)):
        input_path = join(img_dir, fn)
        assert os.path.isfile(input_path)
        internal_test_inputs.append([input_path])
        output_dir = join(output_path, fn.split('.')[0]+'_label')
        internal_test_outputs.append(output_path)

    predictor.predict_from_files(internal_test_inputs,
                                internal_test_outputs,
                                save_probabilities=False, overwrite=False,
                                num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)


if __name__ == "__main__":
  freeze_support()
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--input_path",
    type=str,
    default="/content/drive/MyDrive/nnunet/Original_data/imagesTs-Internal",
    help="path to directory of input images"
  )
  parser.add_argument(
    "--output_path",
    type=str,
    default="/content/drive/MyDrive/nnunet/dataset/nnUNet_results/Dataset019_Hema/seg-Internal-Trans",
    help="path to preferred directory for output segmentations"
  )

  args = parser.parse_args()
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  # instantiate the nnUNetPredictor
  predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                    perform_everything_on_device=True, device=device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
  inference(args.input_path, predictor, args.output_path)
  postprocessing(args.input_path, args.output_path)
  
  
