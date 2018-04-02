import os
import re
import csv
import numpy
import subprocess
from nyul_normalise import normalise_image
import nibabel as nib

def mkdirfun(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def callmyfunction(mycmd):
    cmd = subprocess.Popen(mycmd, shell=True, executable="/bin/bash", stdout=subprocess.PIPE)
    stdoutput = cmd.communicate()[0].decode("utf-8") .strip('\n')
    print(stdoutput)
    return stdoutput

def flipimage(inputimg,interp):
    tmpimg = '/homes/oo2113/Desktop/template.nii.gz'
    callmyfunction('/vol/medic02/users/oo2113/Build/irtk/bin/headertool {0} {1} -reset'.format(inputimg,tmpimg))
    callmyfunction('/vol/medic02/users/oo2113/Build/irtk/bin/headertool {0} {1} -orientation 1 0 0 0 -1 0 0 0 -1'.format(tmpimg,inputimg))
    callmyfunction('/vol/medic02/users/oo2113/Build/irtk/bin/transformation {0} {1} -target {2} -{3}'.format(inputimg,inputimg,tmpimg,interp))
    os.remove(tmpimg)

def getResolution(image_full_name):
    import SimpleITK as sitk
    import numpy as np
    image     = sitk.ReadImage(image_full_name)
    pixelsize = image.GetSpacing()
    return np.array(pixelsize)

def write_dict_2_csv(inp_dict,csvfilename):
    with open(csvfilename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in sorted(inp_dict.items()):
            value = value.tolist()
            value.insert(0,key)
            writer.writerow(value)

challenge_data_dir = '/vol/bitbucket/oo2113/ACDC_Challenge_Training'; subjects = sorted(next(os.walk(challenge_data_dir))[1])
target_dir         = '/vol/biomedic2/oo2113/dataset/ACDC_data_structured'; mkdirfun(target_dir)
target_image_dir   = os.path.join(target_dir,'image'); mkdirfun(target_image_dir)
target_ground_dir  = os.path.join(target_dir,'label'); mkdirfun(target_ground_dir)
target_textname    = os.path.join(target_dir,'subject_metainfo.csv')
metainfo_dict      = {}

for subject in subjects:
    if subject == 'LM': continue
    infofilename = os.path.join(challenge_data_dir,subject,'Info.cfg')
    infofile = open(infofilename)
    subjectinfo = infofile.readlines()

    # Parse the textfile
    ED = int(re.findall(r'\d+', subjectinfo[0])[0])
    ES = int(re.findall(r'\d+', subjectinfo[1])[0])
    Height = float(re.findall(r'\d+', subjectinfo[3])[0])
    Weight = float(re.findall(r'\d+', subjectinfo[5])[0])

    # Identify the pathology group
    if 'NOR' in subjectinfo[2]:
        pathology = int(1)
    elif 'HCM' in subjectinfo[2]:
        pathology = int(2)
    elif 'DCM' in subjectinfo[2]:
        pathology = int(3)
    elif 'RV' in subjectinfo[2]:
        pathology = int(4)
    elif 'MINF' in subjectinfo[2]:
        pathology = int(5)
    else:
        continue
    metainfo_dict[subject] = numpy.array([Height, Weight, pathology])

    # Copy the image
    sourceimagename_ED  = os.path.join(challenge_data_dir, subject, subject + '_frame{0:02d}.nii.gz'.format(ED))
    sourcegroundname_ED = os.path.join(challenge_data_dir, subject, subject + '_frame{0:02d}_gt.nii.gz'.format(ED))
    sourceimagename_ES  = os.path.join(challenge_data_dir, subject, subject + '_frame{0:02d}.nii.gz'.format(ES))
    sourcegroundname_ES = os.path.join(challenge_data_dir, subject, subject + '_frame{0:02d}_gt.nii.gz'.format(ES))
    sourceimages = [sourceimagename_ED,sourcegroundname_ED,sourceimagename_ES,sourcegroundname_ES]

    targetimagename_ED    = os.path.join(target_image_dir, subject)  + '_ED.nii.gz'
    targetimagename_ED_gt = os.path.join(target_ground_dir, subject) + '_ED_gt.nii.gz'
    targetimagename_ES    = os.path.join(target_image_dir, subject)  + '_ES.nii.gz'
    targetimagename_ES_gt = os.path.join(target_ground_dir, subject) + '_ES_gt.nii.gz'
    targetimages = [targetimagename_ED,targetimagename_ED_gt,targetimagename_ES,targetimagename_ES_gt]
    interps = ['linear','nn','linear','nn']

    for source,target,interp in zip(sourceimages,targetimages,interps):
        callmyfunction('/vol/medic02/users/oo2113/Build/other/c3d {0} -type short -o {1}'.format(source, target))
        callmyfunction('/vol/medic02/users/oo2113/Build/irtk/bin/headertool {0} {0} -reset'.format(target))
        flipimage(target, interp)
        callmyfunction('resample {0} {0} -{1} -size 1.82 1.82 {2}'.format(target, interp, getResolution(target)[2]))
        if interp == 'nn':
            callmyfunction('/vol/medic02/users/oo2113/Build/python_scripts/changeLabelId.py '
                           '--inputname {0} --outputname {0} -o 0 3 2 1 -n 0 4 5 6'.format(target))
            callmyfunction('/vol/medic02/users/oo2113/Build/python_scripts/changeLabelId.py '
                           '--inputname {0} --outputname {0} -o 0 4 5 6 -n 0 1 2 3'.format(target))
        else:
            normalise_image(target, target)

write_dict_2_csv(metainfo_dict,target_textname)


