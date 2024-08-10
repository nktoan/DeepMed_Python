import numpy as np
import random
from  deepmed.DeepMed_bin import DeepMed_bin
from  deepmed.DeepMed_bin_cv import DeepMed_bin_cv
from  deepmed.DeepMed_cont import DeepMed_cont
from deepmed.DeepMed_cont_cv import DeepMed_cont_cv
def DeepMed_cv(y,d,m,x,method,hyper_grid,epochs,batch_size):
    mbin = 1 * (
        (len(np.unique(m)) == 2) &
        (np.min(m) == 0) &
        (np.max(m) == 1)
    )
    if method =='DNN':
        random.seed(1)
    if mbin==0:
        hyper=DeepMed_cont_cv(y,d,m,x,method,hyper_grid,epochs,batch_size)
    else:
        hyper=DeepMed_bin_cv(y,d,m,x,method,hyper_grid,epochs,batch_size)
    return hyper