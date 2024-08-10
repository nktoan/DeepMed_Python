import numpy as np
import random
import pandas as pd
import multiprocess
from deepmed.ml_cv import ml_cv
from  deepmed.dnn import dnn
from deepmed.gbm import gbm_out
from deepmed.rf import rf_out
from deepmed.lasso import ls_out
def flatten(l):
    # Check if l is a list of lists
    if isinstance(l, list):
        return [item for sublist in l for item in sublist]
    # If l is a numpy array, just return it as-is
    elif isinstance(l, np.ndarray):
        return l.ravel()  # or use l.flatten()
    else:
        return l  # Return l as is if it's neither a list nor numpy array

def DeepMed_cont_cv(y,d,m,x,method,hyper_grid,epochs,batch_size):
    if method=='DNN':
        ml=dnn
        hyper_grid.insert(hyper_grid.shape[1], 'epoch', epochs)
        hyper_grid.insert(hyper_grid.shape[1], 'batch_size', batch_size)
    if method=='GBM':
        ml=gbm_out   
    if method=='RF':
        ml=rf_out
    if method=='Lasso':
        ml=ls_out
    # xm = np.append(x,m,axis=1)
    xm = np.column_stack((x,m))
    n_hyper= hyper_grid.shape[1]
    hyper = pd.DataFrame()
    
    stepsize= int(np.ceil((1/3)*len(d)))
    nobs = int(min(3*stepsize,len(d)))
    # random.seed(1)
    # idx = [i for i in range(int(nobs))]
    # random.shuffle(idx)
    
    idx = np.random.permutation(len(d))
    sample1 = idx[0:int(stepsize)]
    sample2 = idx[int(stepsize):int(2*stepsize)]
    sample3 = idx[int(2*stepsize):nobs]
    
    
    # crossfitting procedure that splits sample in training an testing data
    for k in range(1,4):
        if k==1:
            tesample=np.array(sample1)
            musample=np.array(sample2)
            deltasample=np.array(sample3)
        if k==2:
            tesample=np.array(sample3)
            musample=np.array(sample1)
            deltasample=np.array(sample2)
        if  k==3:
            tesample=np.array(sample2)
            musample=np.array(sample3)
            deltasample=np.array(sample1)
    
        trsample=np.append(musample,deltasample)
        dte=d[tesample]
        yte=y[tesample]
        hyper_k = pd.DataFrame()
        # 1. fit Pr(D=1|M,X) in total of training data
        # 2. fit Pr(D=1|X) in total of training data
        # 3. fit E(Y|M,X,D=1) in first training data
        # 5. predict E(Y|X,D=1) in the test data
        # 6. fit E(Y|M,X,D=0) in first training data
        # 8. predict E(Y|X,D=0) in the test data
        out = pd.DataFrame()
        for t in range(hyper_grid.shape[0]):
            if method =='DNN':
                out1 =  pd.DataFrame(ml_cv(d[trsample],xm[trsample,:],method,hyper_grid,t))
                out2 =  pd.DataFrame(ml_cv(d[trsample],x[trsample,:],method,hyper_grid,t))
                out3 =  pd.DataFrame(ml_cv(y[musample[np.where(d[musample]==1)[0]]],xm[musample[np.where(d[musample]==1)[0]],:],method,hyper_grid,t))
                out4 =  pd.DataFrame(ml_cv(y[trsample[np.where(d[trsample]==1)[0]]],x[trsample[np.where(d[trsample]==1)[0]],:],method,hyper_grid,t))
                out5 =  pd.DataFrame(ml_cv(y[musample[np.where(d[musample]==0)[0]]],xm[musample[np.where(d[musample]==0)[0]],:],method,hyper_grid,t))
                out6 =  pd.DataFrame(ml_cv(y[trsample[np.where(d[trsample]==0)[0]]],x[trsample[np.where(d[trsample]==0)[0]],:],method,hyper_grid,t))
                outi = pd.concat([out1.T,out2.T,out3.T,out4.T,out5.T,out6.T],axis=1,ignore_index=True)
                out = out.append(outi)
                 
            else:
                cores = multiprocess.cpu_count()
                # Start all worker processes
                pool= multiprocess.Pool(processes=cores)
                random.seed(1)
                p1= [d[trsample],d[trsample],y[musample[np.where(d[musample]==1)[0]]],y[trsample[np.where(d[trsample]==1)[0]]],y[musample[np.where(d[musample]==0)[0]]],y[trsample[np.where(d[trsample]==0)[0]]]]
                p2= [xm[trsample,:],x[trsample,:],xm[musample[np.where(d[musample]==1)[0]],:],x[trsample[np.where(d[trsample]==1)[0]],:],xm[musample[np.where(d[musample]==0)[0]],:],x[trsample[np.where(d[trsample]==0)[0]],:]]
                p3= [method,method,method,method,method,method]
                p4 = [hyper_grid,hyper_grid,hyper_grid,hyper_grid,hyper_grid,hyper_grid]
                p5 = [t,t,t,t,t,t] 
                outi = pool.starmap(ml_cv, list(zip(p1,p2,p3,p4,p5)))
                pool.close()
                pool.join()
                outi = flatten(outi)
                outi = pd.DataFrame(outi)
                out = pd.concat([out, outi.T], ignore_index=True)
        for i in range(0,6):
            outi=out.iloc[:,0: int(n_hyper+1)]
            out=out.iloc[:,int(n_hyper+1):]
            # loc = np.argmin(outi.iloc[:,-1])
            loc = np.argmin(outi.iloc[:,n_hyper])
            hyper_k= pd.concat([hyper_k,pd.DataFrame(outi.iloc[loc,:]).reset_index(drop = True,inplace=False)], axis = 1)
            
        hyper_k.columns=[str(i) for i in [1,2,3,5,6,8]]
        random.seed(1)
        out = pd.DataFrame()
        for t in range(hyper_grid.shape[0]):
            dtrte=d[deltasample]
            xtrte=x[deltasample,:]
    
    

       ############## 4. fit E[E(Y|M,X,D=1)|D=0,X] in delta sample
            eymx1te_all = ml(y[musample[np.where(d[musample]==1)[0]]],xm[musample[np.where(d[musample]==1)[0]],:],
                             y[np.append(tesample,deltasample)],xm[np.append(tesample,deltasample),:],
                             hyper_k.loc[:,"3"])[1]
            eymx1te = eymx1te_all[0:len(tesample)-1] # ypredict E(Y|M,X,D=1) in test data
            eymx1trte = eymx1te_all[len(tesample):]  # ypredict E(Y|M,X,D=1) in delta sample
            print(dtrte.shape)
            out4 = ml_cv(eymx1trte[np.where(flatten(dtrte==[0]))],xtrte[np.where(flatten(dtrte==[0])),:][0],method, hyper_grid, t)
            out4 = pd.DataFrame(out4)
    ############ 7. fit E[E(Y|M,X,D=0)|D=1,X] in delta sample

            eymx0te_all = ml(y[musample[np.where(d[musample]==0)[0]]],xm[musample[np.where(d[musample]==0)[0]],:],
                                 y[np.append(tesample,deltasample)],xm[np.append(tesample,deltasample),:],
                                 hyper_k.loc[:,"6"] )[1]

            eymx0te = eymx0te_all[0:len(tesample)] # ypredict E(Y|M,X,D=0) in test data
            eymx0trte = eymx0te_all[len(tesample):]  # ypredict E(Y|M,X,D=0) in delta sample
    
            out7 = ml_cv(eymx0trte[np.where(flatten(dtrte==[1]))],xtrte[np.where(flatten(dtrte==[1])),:][0], method,hyper_grid, t)
            
            out7 = pd.DataFrame(out7)
            out7 = pd.concat([out4.T.reset_index(drop = True,inplace=False), out7.T.reset_index(drop = True,inplace=False)], axis = 1)
            
            out = pd.concat([out, out7], ignore_index=True)
            #out = out.append(out7)

        for i in range(0,2):
            outi=out.iloc[:,0: int(n_hyper+1)]
            out=out.iloc[:,int(n_hyper+1):]
            loc = np.argmin(outi.iloc[:,-1])
            hyper_k= pd.concat([hyper_k, pd.DataFrame(outi.iloc[loc,:]).reset_index(drop = True,inplace=False)], axis = 1)
        
        hyper_k.columns=[str(i) for i in [1,2,3,5,6,8,4,7]]

        
        hyper_k = hyper_k.loc[:,['1','2','3','4','5','6','7','8']]
        hyper_k.loc[n_hyper,:]=np.round(hyper_k.loc[n_hyper,:],3)
        if method=="DNN":
            hyper_k.loc[3,:]=np.round(hyper_k.loc[3,:])

        if k==1:                      
            hyper=hyper_k
        else:
            hyper = pd.concat([hyper,hyper_k], axis=1)

    return hyper                                                        
                                                                      
                      
        
        
    
