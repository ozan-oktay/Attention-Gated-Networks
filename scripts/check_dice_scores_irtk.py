import os, re
import subprocess
import numpy as np


def callmyfunction(mycmd):
    cmd = subprocess.Popen(mycmd, shell=True, executable="/bin/bash", stdout=subprocess.PIPE)
    stdoutput = cmd.communicate()[0].decode("utf-8") .strip('\n')
    print(stdoutput)
    return stdoutput

gt_dir = '/vol/biomedic2/oo2113/dataset/UKBB_2964/sax/validation/label'
pred_dir = '/vol/biomedic2/oo2113/projects/syntAI/ukbb_pytorch/results_validation'

gt_names = sorted(next(os.walk(gt_dir))[2])
pred_names = sorted(next(os.walk(pred_dir))[2])

assert len(gt_names) == len(pred_names)

endo_dice = []
myo_dice  = []
rv_dice   = []

for gt_name, pred_name in zip(gt_names, pred_names):
    subject1 = gt_name.split('_')[0]
    subject2 = pred_name.split('_')[0]
    assert subject1 == subject2

    # Get the fullpath of prediction and ground-truth labels
    gt_full_name = os.path.join(gt_dir, gt_name)
    pred_full_name = os.path.join(pred_dir, pred_name)

    # Compute the dice scores
    cmd_output = callmyfunction('labelStats {0} {1} -diceRow'.format(pred_full_name, gt_full_name))
    dice_scores = re.findall("\d+\.\d+", cmd_output)

    endo_dice.append(float(dice_scores[0]))
    myo_dice.append(float(dice_scores[1]))
    rv_dice.append(float(dice_scores[2]))

# Report the results
print('LV BP dice score: {0} +- {1}'.format(np.mean(endo_dice), np.std(endo_dice)))
print('Myoca dice score: {0} +- {1}'.format(np.mean(myo_dice), np.std(myo_dice)))
print('RV BP dice score: {0} +- {1}'.format(np.mean(rv_dice), np.std(rv_dice)))