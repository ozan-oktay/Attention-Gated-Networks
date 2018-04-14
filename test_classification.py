import os, sys, numpy as np
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm


from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
from utils.visualiser import Visualiser
from utils.error_logger import ErrorLogger
from models.networks_other import adjust_learning_rate

from models import get_model

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

class StratifiedSampler(object):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.class_vector = class_vector
        self.batch_size = batch_size
        self.num_iter = len(class_vector) // 52
        self.n_class = 14
        self.sample_n = 2
        # create pool of each vectors
        indices = {}
        for i in range(self.n_class):
            indices[i] = np.where(self.class_vector == i)[0]

        self.indices = indices
        self.background_index = np.argmax([ len(indices[i]) for i in range(self.n_class)])


    def gen_sample_array(self):
        # sample 2 from each class
        sample_array = []
        for i in range(self.num_iter):
            arrs = []
            for i in range(self.n_class):
                n = self.sample_n
                if i == self.background_index:
                    n = self.sample_n * (self.n_class-1)
                arr = np.random.choice(self.indices[i], n)
                arrs.append(arr)

            sample_array.append(np.hstack(arrs))
        return np.hstack(sample_array)

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


def test(arguments):

    # Parse input arguments
    json_filename = arguments.config
    network_debug = arguments.debug

    # Load options
    json_opts = json_file_to_pyobj(json_filename)
    train_opts = json_opts.training

    # Architecture type
    arch_type = train_opts.arch_type

    # Setup Dataset and Augmentation
    ds_class = get_dataset(arch_type)
    ds_path  = get_dataset_path(arch_type, json_opts.data_path)
    ds_transform = get_dataset_transformation(arch_type, opts=json_opts.augmentation)

    # Setup the NN Model
    with HiddenPrints():
        model = get_model(json_opts.model)

    if network_debug:
        print('# of pars: ', model.get_number_parameters())
        print('fp time: {0:.8f} sec\tbp time: {1:.8f} sec per sample'.format(*model.get_fp_bp_time2((1,1,224,288))))
        exit()

    # Setup Data Loader
    num_workers = train_opts.num_workers if hasattr(train_opts, 'num_workers') else 16
    
    valid_dataset = ds_class(ds_path, split='val',   transform=ds_transform['valid'], preload_data=train_opts.preloadData)
    test_dataset  = ds_class(ds_path, split='test',  transform=ds_transform['valid'], preload_data=train_opts.preloadData)
   # loader
    batch_size = train_opts.batchSize
    valid_loader = DataLoader(dataset=valid_dataset, num_workers=num_workers, batch_size=train_opts.batchSize, shuffle=False)
    test_loader  = DataLoader(dataset=test_dataset,  num_workers=0, batch_size=train_opts.batchSize, shuffle=False)

    # Visualisation Parameters
    filename = 'test_loss_log.txt'
    visualizer = Visualiser(json_opts.visualisation, save_dir=model.save_dir,
                            filename=filename)
    error_logger = ErrorLogger()

    # Training Function
    track_labels = np.arange(len(valid_dataset.label_names))
    model.set_labels(track_labels)
    model.set_scheduler(train_opts)

    if hasattr(model.net, 'deep_supervised'):
        model.net.deep_supervised = False

    # Validation and Testing Iterations
    pr_lbls = []
    gt_lbls = []
    for loader, split in zip([test_loader], ['test']):
    #for loader, split in zip([valid_loader, test_loader], ['validation', 'test']):
        model.reset_results()

        for epoch_iter, (images, labels) in tqdm(enumerate(loader, 1), total=len(loader)):

            # Make a forward pass with the model
            model.set_input(images, labels)
            model.validate()

        # Error visualisation
        errors = model.get_accumulated_errors()
        stats = model.get_classification_stats()
        error_logger.update({**errors, **stats}, split=split)

    # Update the plots
    # for split in ['train', 'validation', 'test']:
    for split in ['test']:
        # exclude bckground
        #track_labels = np.delete(track_labels, 3)
        #show_labels = train_dataset.label_names[:3] + train_dataset.label_names[4:]
        show_labels = valid_dataset.label_names
        visualizer.plot_current_errors(300, error_logger.get_errors(split), split_name=split, labels=show_labels)
        visualizer.print_current_errors(300, error_logger.get_errors(split), split_name=split)

        import pickle as pkl
        dst_file = os.path.join(model.save_dir, 'test_result.pkl')
        with open(dst_file, 'wb') as f:
            d = error_logger.get_errors(split)
            d['labels'] = valid_dataset.label_names
            d['pr_lbls'] = np.hstack(model.pr_lbls)
            d['gt_lbls'] = np.hstack(model.gt_lbls)
            pkl.dump(d, f)

    error_logger.reset()

    if arguments.time:
        print('# of pars: ', model.get_number_parameters())
        print('fp time: {0:.8f} sec\tbp time: {1:.8f} sec per sample'.format(*model.get_fp_bp_time2((1,1,224,288))))
        

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg Training Function')

    parser.add_argument('-c', '--config',  help='training config file', required=True)
    parser.add_argument('-d', '--debug',   help='returns number of parameters and bp/fp runtime', action='store_true')
    parser.add_argument('-t', '--time',   help='returns number of parameters and bp/fp runtime', action='store_true')
    args = parser.parse_args()

    test(args)
