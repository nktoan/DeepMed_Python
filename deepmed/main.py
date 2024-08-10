import numpy as np
import pandas as pd
from scipy.stats import norm
from  deepmed.DeepMed_bin import DeepMed_bin
from  deepmed.DeepMed_bin_cv import DeepMed_bin_cv
from  deepmed.DeepMed_cont import DeepMed_cont
from deepmed.DeepMed_cont_cv import DeepMed_cont_cv
from  deepmed.DeepMed_cv import DeepMed_cv
import warnings
warnings.filterwarnings("ignore") 
def expand_grid(*para):
    if len(para)==3:
        x,y,z = para[0],para[1],para[2]
        xG, yG ,zG= np.meshgrid(x, y,z) # create the actual grid
        xG = xG.flatten() # make the grid 1d
        yG = yG.flatten() # same
        zG = zG.flatten()
        return pd.DataFrame({'Var1':xG, 'Var2':yG,'Var3':zG}) # return a dataframe
    else:
        x,y = para[0],para[1]
        xG, yG= np.meshgrid(x, y) # create the actual grid
        xG = xG.flatten() # make the grid 1d
        yG = yG.flatten() # same
        return pd.DataFrame({'Var1':xG, 'Var2':yG}) # return a dataframe


class DeepMed:
    def __init__(self,y,d,m,x,method="DNN",hyper_grid=None,epochs=500,batch_size=100,trim=0.05):
        self.y = y
        self.d = d
        self.m = m
        self.x = x
        self.method = method
        self.hyper_grid = hyper_grid
        self.epochs =epochs
        self.batch_size = batch_size
        self.trim = trim
    def run(self):
        if self.method !="Lasso":
            self.hyper_grid = pd.DataFrame(self.hyper_grid)
            hyper=DeepMed_cv(self.y,self.d,self.m,self.x,self.method,self.hyper_grid,self.epochs,self.batch_size)
        else:
            hyper = np.empty(shape = (2,30))
        
        mbin = 1 * (
          (len(np.unique(self.m)) == 2) &
          (np.min(self.m) == 0) &
         (np.max(self.m) == 1)
        )
        #mbin = 1 * ((len(np.unique(self.m)) == 2) & (np.min(self.m) == 0) &(np.max(self.m) == 1))
        if mbin==0:
            temp =DeepMed_cont(self.y,self.d,self.m,self.x,self.method,hyper,self.trim)
        else:
            temp=DeepMed_bin(self.y,self.d,self.m,self.x,self.method,hyper,self.trim)
        ATE = np.array(temp)
        eff=ATE[0:5]
        se=np.sqrt(ATE[5:10]/ATE[10])
        results = [eff,se,2*norm.cdf(-abs(eff/se),loc=0,scale=1)]
        results = pd.DataFrame(results)
        #results= pd.concat([pd.Series(eff), pd.Series(se), pd.Series(2*norm.cdf(-abs(eff/se),loc=0,scale=1))])
        results.columns=["total", "dir.treat", "dir.control", "indir.treat", "indir.control"]
        results.index=["effect","se","pval"]
        ntrimmed=len(self.d)-ATE[10]
        print(results)
        print('ntrimmed:',ntrimmed)
        return results,ntrimmed,hyper
