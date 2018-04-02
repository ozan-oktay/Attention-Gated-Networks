from torch.utils.data import DataLoader

from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj

from models import get_model
import numpy as np
import os
from utils.metrics import dice_score, distance_metric, precision_and_recall
from utils.error_logger import StatLogger


def mkdirfun(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def validation(json_name):
    # Load options
    json_opts = json_file_to_pyobj(json_name)
    train_opts = json_opts.training

    # Setup the NN Model
    model = get_model(json_opts.model)
    save_directory = os.path.join(model.save_dir, train_opts.arch_type); mkdirfun(save_directory)

    # Setup Dataset and Augmentation
    dataset_class = get_dataset(train_opts.arch_type)
    dataset_path = get_dataset_path(train_opts.arch_type, json_opts.data_path)
    dataset_transform = get_dataset_transformation(train_opts.arch_type, opts=json_opts.augmentation)

    # Setup Data Loader
    dataset = dataset_class(dataset_path, split='validation', transform=dataset_transform['valid'])
    data_loader = DataLoader(dataset=dataset, num_workers=8, batch_size=1, shuffle=False)

    # Visualisation Parameters
    #visualizer = Visualiser(json_opts.visualisation, save_dir=model.save_dir)

    # Setup stats logger
    stat_logger = StatLogger()

    # test
    for iteration, data in enumerate(data_loader, 1):
        model.set_input(data[0], data[1])
        model.test()

        input_arr  = np.squeeze(data[0].cpu().numpy()).astype(np.float32)
        label_arr  = np.squeeze(data[1].cpu().numpy()).astype(np.int16)
        output_arr = np.squeeze(model.pred_seg.cpu().byte().numpy()).astype(np.int16)

        # If there is a label image - compute statistics
        dice_vals = dice_score(label_arr, output_arr, n_class=int(4))
        md, hd = distance_metric(label_arr, output_arr, dx=2.00, k=2)
        precision, recall = precision_and_recall(label_arr, output_arr, n_class=int(4))
        stat_logger.update(split='test', input_dict={'img_name': '',
                                                     'dice_LV': dice_vals[1],
                                                     'dice_MY': dice_vals[2],
                                                     'dice_RV': dice_vals[3],
                                                     'prec_MYO':precision[2],
                                                     'reca_MYO':recall[2],
                                                     'md_MYO': md,
                                                     'hd_MYO': hd
                                                      })

        # Write a nifti image
        import SimpleITK as sitk
        input_img = sitk.GetImageFromArray(np.transpose(input_arr, (2, 1, 0))); input_img.SetDirection([-1,0,0,0,-1,0,0,0,1])
        label_img = sitk.GetImageFromArray(np.transpose(label_arr, (2, 1, 0))); label_img.SetDirection([-1,0,0,0,-1,0,0,0,1])
        predi_img = sitk.GetImageFromArray(np.transpose(output_arr,(2, 1, 0))); predi_img.SetDirection([-1,0,0,0,-1,0,0,0,1])

        sitk.WriteImage(input_img, os.path.join(save_directory,'{}_img.nii.gz'.format(iteration)))
        sitk.WriteImage(label_img, os.path.join(save_directory,'{}_lbl.nii.gz'.format(iteration)))
        sitk.WriteImage(predi_img, os.path.join(save_directory,'{}_pred.nii.gz'.format(iteration)))

    stat_logger.statlogger2csv(split='test', out_csv_name=os.path.join(save_directory,'stats.csv'))
    for key, (mean_val, std_val) in stat_logger.get_errors(split='test').items():
        print('-',key,': \t{0:.3f}+-{1:.3f}'.format(mean_val, std_val),'-')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg Validation Function')

    parser.add_argument('-c', '--config', help='testing config file', required=True)
    args = parser.parse_args()

    validation(args.config)
