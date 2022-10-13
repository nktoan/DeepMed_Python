import numpy as np
import random
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor 
from sklearn.metrics import mean_squared_error

def rf_out(y, x, ytest, xtest, hyper):
    train_ybin=1*(len(np.unique(y))==2 & min(y)==0 & max(y)==1)
    hyper = hyper.values
    hyper = hyper.T.reshape(-1)    
    node=hyper[0]
    ntree=hyper[1]
    
    if train_ybin==1:
        rf_model = RandomForestClassifier(n_estimators=ntree,  min_samples_leaf=node )  
        rf_model.fit(x,y)
        ypred = rf_model.predict_proba(xtest)
        ypred[ypred==1]=1-1e-5
        ypred[ypred==0]=1e-5
        loss = -mean(ytest*log(ypred) + (1-ytest)*log(1-ypred))
        
    else:
        rf_model = RandomForestRegressor(n_estimators=ntree,  min_samples_leaf=node )  
        rf_model.fit(x,y)
        ypred = rf_model.predict(xtest)
        loss =  mean_squared_error(ytest, ypred)
    return loss,ypred