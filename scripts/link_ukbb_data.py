import os

root_dir = '/vol/medic02/users/wbai/dataio/cardiac_atlas/UKBB_2964/sa'
save_dir = '/vol/biomedic2/oo2113/dataset/UKBB_2964/sax'

targetnames = ['sa_ED.nii.gz','sa_ES.nii.gz','label_sa_ED.nii.gz','label_sa_ES.nii.gz']
targetcategories = ['image', 'image', 'label', 'label']



def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def mkdirfunc(directory):
    if not (os.path.exists(directory)):
        os.makedirs(directory)


splitdirs = get_immediate_subdirectories(root_dir)
for split in splitdirs:
    if split in ['train','test','validation']:
        mkdirfunc(os.path.join(save_dir, split))
        [mkdirfunc(os.path.join(save_dir, split, cat)) for cat in targetcategories]
        subjectdirs = get_immediate_subdirectories(os.path.join(root_dir,split))
        for subject in subjectdirs:
            for (dirpath, dirnames, filenames) in os.walk(os.path.join(root_dir,split,subject)):
                for filename in sorted(filenames):
                    if filename in targetnames:
                        category     = targetcategories[targetnames.index(filename)]
                        filefullname = os.path.join(root_dir,split,subject,filename)
                        newfullname  = os.path.join(save_dir, split, category, subject + '_' + filename)
                        filefullname = os.path.realpath(filefullname)
                        if not os.path.isfile(newfullname):
                            os.symlink(filefullname, newfullname)
                            print('creating a symblink: ', newfullname)



