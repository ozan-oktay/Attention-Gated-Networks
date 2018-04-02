import numpy as np
import os
import torch

from torch.utils.data import DataLoader
from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj, determine_crop_size
from dataio.loader.utils import write_nifti_img

from models import get_model
from utils.metrics import dice_score, distance_metric, precision_and_recall
from utils.error_logger import StatLogger

def inference(json_filename):

    # Load options
    json_opts = json_file_to_pyobj(json_filename)

    # Setup the NN Model
    arch_type = 'test_sax'
    model = get_model(json_opts.model)

    # Output save directory
    save_directory = os.path.join(model.save_dir, arch_type)

    # Use post-processing
    use_crf = False

    # Setup Dataset and Augmentation
    dataset_class = get_dataset(arch_type)
    dataset_path = get_dataset_path(arch_type, json_opts.data_path)
    dataset_transform = get_dataset_transformation(arch_type, json_opts.augmentation)

    # Setup Data Loader
    test_dataset = dataset_class(dataset_path, transform=dataset_transform['test'])
    test_loader = DataLoader(dataset=test_dataset, num_workers=1, batch_size=1, shuffle=False)

    # Setup stats logger
    stat_logger = StatLogger()

    # testing loop
    for iteration, (input_arr, input_meta, label_arr) in enumerate(test_loader, 1):
        pixel_size  = input_meta['pixdim'][0][1:4].cpu().numpy()
        input_shape = input_meta['dim'][0][1:4].cpu().numpy()
        output = np.zeros(input_shape, dtype=np.int8)

        # Determine the crop area
        pre_pad, _ = determine_crop_size(input_shape, json_opts.augmentation.test_sax.division_factor)

        # Run the model
        model.set_input(input_arr)
        model.test()

        # Use Conditional Random Field to post-process the segmentations
        if use_crf:
            from utils.post_process_crf import apply_crf
            prob = model.logits.data.squeeze().permute(1, 2, 3, 0).cpu().float().numpy()
            inp = input_arr.squeeze().cpu().numpy()
            pred_np = apply_crf(inp, prob, theta_a=0.5, theta_b=1, theta_r=1, mu1=1, mu2=2)
        else:
            pred_np = np.squeeze(model.pred_seg.cpu().byte().numpy())

        # Remove the padded area
        output[:, :, :] = pred_np[pre_pad[0]: pre_pad[0] + input_shape[0],
                                  pre_pad[1]: pre_pad[1] + input_shape[1],
                                  pre_pad[2]: pre_pad[2] + input_shape[2]]

        # write the images
        write_nifti_img(output, input_meta, save_directory)

        # If there is a label image - compute statistics
        if torch.is_tensor(label_arr):
            label_arr = label_arr[0].cpu().numpy()
            dice_vals = dice_score(label_arr, output, n_class=int(4))
            md, hd = distance_metric(label_arr, output, dx=pixel_size[0], k=1)
            precision, recall = precision_and_recall(label_arr, output, n_class=int(4))
            stat_logger.update(split='test', input_dict={'img_name':input_meta['name'][0],
                                                         'dice_LV': dice_vals[1],
                                                         'dice_MY': dice_vals[2],
                                                         'dice_RV': dice_vals[3],
                                                         'prec_MYO': precision[1],
                                                         'reca_MYO': recall[1],
                                                         'md_MYO': md,
                                                         'hd_MYO': hd
                                                          })
        if iteration==100:
            break

    stat_logger.statlogger2csv(split='test', out_csv_name=os.path.join(save_directory,'stats.csv'))
    for key, (mean_val, std_val) in stat_logger.get_errors(split='test').items():
        print('-',key,': \t{0:.3f}+-{1:.3f}'.format(mean_val, std_val),'-')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg Inference Function')

    parser.add_argument('-c', '--config', help='testing config file', required=True)
    args = parser.parse_args()

    inference(args.config)
