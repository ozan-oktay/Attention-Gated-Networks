import numpy as np
from .util import csv_write


class BaseMeter(object):
    """Just a place holderb"""

    def __init__(self, name):
        self.reset()
        self.name = name

    def reset(self):
        pass

    def update(self, val):
        self.val = val

    def get_value(self):
        return self.val


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.reset()
        self.name = name

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1.0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_value(self):
        return self.avg

class StatMeter(object):
    """Computes and stores the error vals and image names"""

    def __init__(self, name, csv_name=None):
        self.reset()
        self.name = name

    def reset(self):
        self.vals = []
        self.img_names = []

    def update(self, val, img_name):
        self.vals.append(val)
        self.img_names.append(img_name)

    def return_average(self):
        values_array = np.array(self.vals, dtype=np.float)
        return np.nanmean(values_array)

    def return_std(self):
        values_array = np.array(self.vals, dtype=np.float)
        return np.nanstd(values_array)


class ErrorLogger(object):

    def __init__(self):
        self.variables = {'train': dict(),
                          'validation': dict(),
                          'test': dict()
                          }

    def update(self, input_dict, split):

        for key, value in input_dict.items():
            if key not in self.variables[split]:
                if np.isscalar(value):
                    self.variables[split][key] = AverageMeter(name=key)
                else:
                    self.variables[split][key] = BaseMeter(name=key)

            self.variables[split][key].update(value)


    def get_errors(self, split):
        output = dict()
        for key, meter_obj in self.variables[split].items():
            output[key] = meter_obj.get_value()
        return output

    def reset(self):
        for key, meter_obj in self.variables['train'].items():
            meter_obj.reset()
        for key, meter_obj in self.variables['validation'].items():
            meter_obj.reset()
        for key, meter_obj in self.variables['test'].items():
            meter_obj.reset()


class StatLogger(object):

    def __init__(self):
        self.variables = {'train': dict(),
                          'validation': dict(),
                          'test': dict()
                          }

    def update(self, input_dict, split):
        img_name = input_dict.pop('img_name', None)
        for key, value in input_dict.items():
            if key not in self.variables[split]:
                self.variables[split][key] = StatMeter(name=key)
            self.variables[split][key].update(val=value, img_name=img_name)

    def get_errors(self, split):
        output = dict()
        for key, meter_obj in self.variables[split].items():
            output[key] = (meter_obj.return_average(), meter_obj.return_std())
        return output

    def statlogger2csv(self, split, out_csv_name):
        csv_values = []; csv_header = []
        for loopId, (meter_key, meter_obj) in enumerate(self.variables[split].items(), 1):
            if loopId == 1: csv_values.append(meter_obj.img_names); csv_header.append('img_names')
            csv_values.append(meter_obj.vals)
            csv_header.append(meter_key)
        csv_write(out_csv_name, csv_header, csv_values)

    def reset(self):
        for key, meter_obj in self.variables['train'].items():
            meter_obj.reset()
        for key, meter_obj in self.variables['validation'].items():
            meter_obj.reset()
        for key, meter_obj in self.variables['test'].items():
            meter_obj.reset()
