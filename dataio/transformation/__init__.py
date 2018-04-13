import json
from dataio.transformation.transforms import Transformations


def get_dataset_transformation(name, opts=None):
    '''
    :param opts: augmentation parameters
    :return:
    '''
    # Build the transformation object and initialise the augmentation parameters
    trans_obj = Transformations(name)
    if opts: trans_obj.initialise(opts)

    # Print the input options
    trans_obj.print()

    # Returns a dictionary of transformations
    return trans_obj.get_transformation()
