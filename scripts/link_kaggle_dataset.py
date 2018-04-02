import os

targetnames = ['image_ED.nii.gz','image_ES.nii.gz','label_ED.nii.gz','label_ES.nii.gz']
targetcategories = ['image', 'image', 'label', 'label']


def filter_subdirectories(subdirs):
    restricted_words = ['SC','HF', 'NI', 'HYP']
    output = []
    for subdir in subdirs:
        for word in restricted_words:
            if word in subdir:
                break
            else:
                output.append(subdir)
    return output

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

def mkdirfunc(directory):
    if not (os.path.exists(directory)):
        os.makedirs(directory)

root_dir = '/vol/medic02/users/wbai/data/kaggle_atlas'
save_dir = '/vol/biomedic2/oo2113/dataset/kaggle_atlas'; mkdirfunc(save_dir)

subjectdirs = get_immediate_subdirectories(root_dir)
subjectdirs = filter_subdirectories(subjectdirs)

for subject in subjectdirs:
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(root_dir,subject)):
        for filename in sorted(filenames):
            if filename in targetnames:
                category     = targetcategories[targetnames.index(filename)]
                mkdirfunc(os.path.join(save_dir, category))
                filefullname = os.path.join(root_dir,subject,filename)
                newfullname  = os.path.join(save_dir,category, subject + '_' + filename)
                filefullname = os.path.realpath(filefullname)
                if not os.path.isfile(newfullname):
                    os.symlink(filefullname, newfullname)
                    print('creating a symblink: ', newfullname)



