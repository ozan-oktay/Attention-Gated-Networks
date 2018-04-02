import os, argparse
import numpy as np, nibabel as nib

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian


def apply_crf(input_image, input_prob, theta_a, theta_b, theta_r, mu1, mu2):
    n_slices = input_image.shape[2]
    output = np.zeros(input_image.shape)
    for slice_id in range(n_slices):
        image = input_image[:,:,slice_id]
        prob = input_prob[:,:,slice_id,:]

        n_pixel = image.shape[0] * image.shape[1]
        n_class = prob.shape[-1]

        P = np.transpose(prob, axes=(2, 0, 1))

        # Setup the CRF model
        d = dcrf.DenseCRF(n_pixel, n_class)

        # Set unary potentials (negative log probability)
        U = - np.log(P + 1e-10)
        U = np.ascontiguousarray(U.reshape((n_class, n_pixel)))
        d.setUnaryEnergy(U)

        # Set edge potential
        # This creates the color-dependent features and then add them to the CRF
        feats = create_pairwise_bilateral(sdims=(theta_a, theta_a), schan=(theta_b,), img=image, chdim=-1)
        d.addPairwiseEnergy(feats, compat=mu1, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This creates the color-independent features and then add them to the CRF
        feats = create_pairwise_gaussian(sdims=(theta_r, theta_r), shape=image.shape)
        d.addPairwiseEnergy(feats, compat=mu2, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

        # Perform the inference
        Q = d.inference(5)
        res = np.argmax(Q, axis=0).astype('float32')
        res = np.reshape(res, image.shape).astype(dtype='int8')
        output[:,:,slice_id] = res

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train', metavar='int', nargs=1, default=['80'], help='number of training subjects')
    args = parser.parse_args()

    # Data path
    data_path = '/vol/medic02/users/wbai/data/cardiac_atlas/Biobank'
    data_list = sorted(os.listdir(data_path))
    dest_path = '/vol/bitbucket/wbai/cardiac_cnn/Biobank/seg'

    # Model name
    size = 224
    n_train = int(args.n_train[0])
    model_name = 'FCN_VGG16_sz{0}_n{1}_2d'.format(size, n_train)
    epoch = 500

    # model_name = 'FCN_VGG16_sz{0}_prob_atlas_stepwise'.format(size)
    # epoch = 200

    # model_name = 'FCN_VGG16_sz{0}_auto_context_stepwise'.format(size)
    # epoch = 200

    for data in data_list:
        print(data)
        data_dir = os.path.join(data_path, data)
        dest_dir = os.path.join(dest_path, data)

        # tune_dir = os.path.join(dest_dir, 'tune')
        # if not os.path.exists(tune_dir):
        #     os.mkdir(tune_dir)

        for fr in ['ED', 'ES']:
            # Read image
            nim = nib.load(os.path.join(data_dir, 'image_{0}.nii.gz'.format(fr)))
            image = np.squeeze(nim.get_data())

            # Scale the intensity to be [0, 1] so that we can set a consistent intensity parameter for CRF
            #image = intensity_rescaling(image, 1, 99)

            # Read probability map
            nim = nib.load(os.path.join(dest_dir, 'prob_{0}_{1}_epoch{2:03d}.nii.gz'.format(fr, model_name, epoch)))
            prob = nim.get_data()

            # Apply CRF
            mu1 = 1
            theta_a = 0.5
            theta_b = 1
            mu2 = 2
            theta_r = 1
            seg = apply_crf(image, prob, theta_a, theta_b, theta_r, mu1, mu2)

            # Save the CRF segmentation
            seg_name = os.path.join(dest_dir, 'seg_{0}_{1}_epoch{2:03d}_crf.nii.gz'.format(fr, model_name, epoch))
            nib.save(nib.Nifti1Image(seg, nim.affine), seg_name)

            # For parameter tuning
            # nib.save(nib.Nifti1Image(seg, nim.affine), os.path.join(tune_dir, 'seg_{0}_crf_mu2{1}_sr{2}.nii.gz'.format(fr, mu2, theta_r)))
            # nib.save(nib.Nifti1Image(seg, nim.affine), os.path.join(tune_dir, 'seg_{0}_crf_mu1{1}_sa{2:.1f}_sb{1}.nii.gz'.format(fr, mu1, theta_a, theta_b)))

            # # Fit to the template
            # template_dir = '/vol/medic02/users/wbai/data/imperial_atlas/template'
            # par_dir = '/vol/vipdata/data/biobank/cardiac/Application_18545/par'
            # out_name = os.path.join(dest_dir, 'seg_{0}_{1}_epoch{2:03d}_crf_fit.nii.gz'.format(fr, model_name, epoch))
            # fit_to_template(seg_name, fr, template_dir, par_dir, out_name)