import numpy as np
import scipy.stats

train_types = {'st':'standard training', 
               'at':'adversarial training', 
               'jr':'jacobian regularization', 
               'ir': 'interpretation regularization'}

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h