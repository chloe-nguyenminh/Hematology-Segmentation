import multiprocessing
import shutil
from multiprocessing import Pool
import os
from tqdm import tqdm

from batchgenerators.utilities.file_and_folder_operations import *

from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from skimage import io
from acvl_utils.morphology.morphology_helper import generic_filter_components
from scipy.ndimage import binary_fill_holes

join = os.path.join

def load_and_convert_case(imagesTr_image: str, imagesTr_seg: str, labelsTr_image: str, labelsTr_seg: str,
                          min_component_size: int = 50):
    try:
        seg = io.imread(imagesTr_seg)
        seg[seg == 255] = 1
        image = io.imread(imagesTr_image)
        image = image.sum(2)
        mask = image == (3 * 255)
        
        smask = generic_filter_components(mask, filter_fn=lambda ids, sizes: [i for j, i in enumerate(ids) if
                                                                            sizes[j] > min_component_size])
        mask = binary_fill_holes(mask)
        seg[mask] = 0
        io.imsave(labelsTr_seg, seg, check_contrast=False)
        shutil.copy(imagesTr_image, labelsTr_image)
    except Exception as e:
        print(e, imagesTr_image)


if __name__ == "__main__":
    # extracted archive from https://www.kaggle.com/datasets/insaff/massachusetts-roads-dataset?resource=download
    source = '/content/drive/MyDrive/nnunet/data'
    dataset_name = 'Dataset019_Hema'
    imagestr = join(nnUNet_raw, dataset_name, 'imagesTr')
    labelstr = join(nnUNet_raw, dataset_name, 'labelsTr')
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    with multiprocessing.get_context("spawn").Pool(8) as p:

        # not all training images have a segmentation
        valid_ids = subfiles(join(source, 'labelsTr'), join=False, suffix='.png')
        r = []
        for v in tqdm(valid_ids):
                r.append(
                    p.starmap_async(
                        load_and_convert_case,
                        ((
                            join(source, 'imagesTr', v),
                            join(source, 'labelsTr', v),
                            join(imagestr, v[:-4] + '_0000.png'),
                            join(labelstr, v),
                            50
                        ),)
                    )
                )

        _ = [i.get() for i in r]
        num_train = len(os.listdir(join(source, "imagesTr")))

    generate_dataset_json(output_folder=join(nnUNet_raw, dataset_name), 
                          channel_names={0: 'R', 1:'G', 2:'B'}, 
                          labels={'background': 0, 'cytoplasm': 1, 'nucleus': 2},
                          num_training_cases=num_train, 
                          file_ending='.png',
                          dataset_name=dataset_name,
                          overwrite_image_reader_write="SwinUNetrNaturalImage2DIO")
