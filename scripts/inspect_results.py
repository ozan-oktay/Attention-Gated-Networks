import os
import subprocess

def callmyfunction(mycmd):
    cmd = subprocess.Popen(mycmd, shell=True, executable="/bin/bash", stdout=subprocess.PIPE)
    stdoutput = cmd.communicate()[0].decode("utf-8") .strip('\n')
    print(stdoutput)
    return stdoutput

source_img_dir = '/vol/biomedic2/oo2113/dataset/ken_abdominal_ct_pancreas/validation/image'
source_lbl_dir = '/vol/biomedic2/oo2113/dataset/ken_abdominal_ct_pancreas/validation/label'

output_1_dir = '/vol/biomedic2/oo2113/projects/syntAI/ukbb_pytorch/checkpoints/experiment_unet_ct_pancreas/test_sax'
output_2_dir = '/vol/biomedic2/oo2113/projects/syntAI/ukbb_pytorch/checkpoints/experiment_unet_ct_att_pancreas/test_sax'

query_name = 'nusurgery333.512.nii.gz'

# Prepare the arguments
source_img = os.path.join(source_img_dir, query_name)
output1 = os.path.join(output_1_dir, query_name)
output2 = os.path.join(output_2_dir, query_name)
source_lbl = os.path.join(source_lbl_dir, query_name)

# Run the command
callmyfunction('rview -target {0} -source {1} {2} {3} -tmax 2200 -tmin 1800 -scontour'.format(source_img, output1, output2, source_lbl))