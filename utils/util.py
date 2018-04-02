from __future__ import print_function
import torch
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
import json
import csv
from skimage.exposure import rescale_intensity

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imgtype='img', datatype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.ndim == 4:# image_numpy (C x W x H x S)
        mid_slice = image_numpy.shape[-1]//2
        image_numpy = image_numpy[:,:,:,mid_slice]
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    if imgtype == 'img':
        image_numpy = (image_numpy + 8) / 16.0 * 255.0
    if np.unique(image_numpy).size == int(1):
        return image_numpy.astype(datatype)
    return rescale_intensity(image_numpy.astype(datatype))


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def json_file_to_pyobj(filename):
    def _json_object_hook(d): return collections.namedtuple('X', d.keys())(*d.values())
    def json2obj(data): return json.loads(data, object_hook=_json_object_hook)
    return json2obj(open(filename).read())


def determine_crop_size(inp_shape, div_factor):
    div_factor= np.array(div_factor, dtype=np.float32)
    new_shape = np.ceil(np.divide(inp_shape, div_factor)) * div_factor
    pre_pad = np.round((new_shape - inp_shape) / 2.0).astype(np.int16)
    post_pad = ((new_shape - inp_shape) - pre_pad).astype(np.int16)
    return pre_pad, post_pad


def csv_write(out_filename, in_header_list, in_val_list):
    with open(out_filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(in_header_list)
        writer.writerows(zip(*in_val_list))
