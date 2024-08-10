import numpy as np
import random
import pandas as pd
import multiprocess
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

def DeepMed_cont(y,d,m,x,method,hyper,trim=0.05):
    if method=='DNN':
        ml=dnn
    if method=='GBM':
        ml=gbm_out   
    if method=='RF':
        ml=rf_out
    if method=='Lasso':
        ml=ls_out
    xm = np.append(x,m,axis=1)
    
    stepsize= int(np.ceil((1/3)*len(d)))
    nobs = int(min(3*stepsize,len(d)))
    random.seed(1)
    # idx = [i for i in range(nobs)]
    # random.shuffle(idx)
    idx = np.random.permutation(len(d))
    
    sample1 = idx[0:int(stepsize)]
    sample2 = idx[int(stepsize):int(2*stepsize)]
    sample3 = idx[int(2*stepsize):nobs]
    
    y1m0=list()
    y1m1=list()
    y0m0=list()
    y0m1=list() 
    selall=list()
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

        trsample= np.append(musample,deltasample)
        dte=d[tesample]
        yte=y[tesample]
        dtrte=d[deltasample]
        xtrte=x[deltasample,:]
        if method =='DNN':
            out = []
            hyper = np.array(hyper)
            out1 =  ml(d[trsample],xm[trsample,:],d[tesample],xm[tesample,:],hyper[:,0])
            out2 =  ml(d[trsample],x[trsample,:],d[tesample],x[tesample,:],hyper[:,1])
            out3 = ml(y[musample[np.where(d[musample]==1)[0]]],xm[musample[np.where(d[musample]==1)[0]],:],y[np.append(tesample,deltasample)],xm[np.append(tesample,deltasample),:],hyper[:,3])
            out4 = ml(y[trsample[np.where(d[trsample]==1)[0]]],x[trsample[np.where(d[trsample]==1)[0]],:],y[tesample],x[tesample,:],hyper[:,4])
            out5 = ml(y[musample[np.where(d[musample]==0)[0]]],xm[musample[np.where(d[musample]==0)[0]],:],y[np.append(tesample,deltasample)],xm[np.append(tesample,deltasample),:],hyper[:,5])
            out6 =ml(y[trsample[np.where(d[trsample]==0)[0]]],x[trsample[np.where(d[trsample]==0)[0]],:],y[tesample],x[tesample,:],hyper[:,6])
            out.append(out1)
            out.append(out2)
            out.append(out3)
            out.append(out4)
            out.append(out5)
            out.append(out6)
        else:       

            # Get all worker processes
            cores = multiprocess.cpu_count()
            # Start all worker processes
            pool= multiprocess.Pool(processes=cores)
            random.seed(1)
            hyper = np.array(hyper)
            p1= [d[trsample],d[trsample],y[musample[np.where(d[musample]==1)[0]]],y[trsample[np.where(d[trsample]==1)[0]]],y[musample[np.where(d[musample]==0)[0]]],y[trsample[np.where(d[trsample]==0)[0]]]]
            p2= [xm[trsample,:],x[trsample,:],xm[musample[np.where(d[musample]==1)[0]],:],x[trsample[np.where(d[trsample]==1)[0]],:],xm[musample[np.where(d[musample]==0)[0]],:],x[trsample[np.where(d[trsample]==0)[0]],:]]
            p3= [d[tesample],d[tesample],y[np.append(tesample,deltasample)],y[tesample],y[np.append(tesample,deltasample)],y[tesample]]
            p4 = [xm[tesample,:],x[tesample,:],xm[np.append(tesample,deltasample),:],x[tesample,:],xm[np.append(tesample,deltasample),:], x[tesample,:]]
            p5 = [hyper[:,0],hyper[:,1],hyper[:,2],hyper[:,4],hyper[:,5],hyper[:,6]]
            out = pool.starmap(ml, list(zip(p1,p2,p3,p4,p5)))
            pool.close()
        pmxte=out[0][1]
        pxte=out[1][1]
        eymx1te_all=out[2][1]

        eymx1te = eymx1te_all[0:len(tesample)] # ypredict E(Y|M,X,D=1) in test data
        eymx1trte = eymx1te_all[len(tesample):] # ypredict E(Y|M,X,D=1) in delta sample
        
        #print(xtrte[np.where(flatten(dtrte==[0])),:][0])
        regweymx1te = ml(eymx1trte[np.where(flatten(dtrte==[0]))],xtrte[np.where(flatten(dtrte==[0])),:][0],eymx1te, x[tesample,:],hyper[:,3])[1]

        eyx1te=out[3][1]
        
        eymx0te_all=out[4][1]
        eymx0te = eymx0te_all[0:len(tesample)] # ypredict E(Y|M,X,D=0) in test data
        eymx0trte =eymx0te_all[len(tesample):] # ypredict E(Y|M,X,D=0) in delta sample

        regweymx0te = ml(eymx0trte[np.where(flatten(dtrte==[1]))],xtrte[np.where(flatten(dtrte==[1])),:][0],eymx0te, x[tesample,:], hyper[:,6])[1]

        eyx0te=out[5][1]
        #hyper=hyper[:,0:7]
        hyper=hyper[:,8:]
        # select observations satisfying trimming restriction

        sel= 1*((((1-pmxte)*pxte)>=trim) & ((1-pxte)>=trim)  & (pxte>=trim) &  (((pmxte*(1-pxte)))>=trim))
        # ypredict E(Y0,M(1)) in the test data
        temp=((1-dte).flatten()*pmxte/((1-pmxte).flatten()*pxte)*(yte.flatten()-eymx0te)+dte.flatten()/pxte*(eymx0te-regweymx0te)+regweymx0te)
        # y0m1 = y1m0+ temp[sel==1].tolist()
        y0m1.extend(temp[sel==1])
        # ypredict E(Y0,M(0)) in the test data
        temp=(eyx0te + (1-dte).flatten()*(yte.flatten()-eyx0te)/(1-pxte))
        # y0m0 = y0m0 + temp[sel==1].tolist()
        y0m0.extend(temp[sel==1])
        # ypredict E(Y1,M(0)) in the test data
        temp=(dte.flatten()*(1-pmxte).flatten()/(pmxte*(1-pxte).flatten())*(yte.flatten()-eymx1te)+(1-dte).flatten()/(1-pxte).flatten()*(eymx1te-regweymx1te)+regweymx1te)
        # y1m0 = y1m0+ temp[sel==1].tolist()
        y1m0.extend(temp[sel==1])
        # ypredict E(Y1,M(1)) in the test data
        temp=(eyx1te + dte.flatten()*(yte.flatten()-eyx1te)/pxte)
        # y1m1 = y1m1+ temp[sel==1].tolist()
        y1m1.extend(temp[sel==1])
        # collect selection dummies
        # selall = selall+ sel.tolist()
        selall.extend(sel)
    # average over the crossfitting steps
    my1m1=np.mean(y1m1)
    my0m1=np.mean(y0m1)
    my1m0=np.mean(y1m0)
    my0m0=np.mean(y0m0)
    # compute effects
    tot=my1m1-my0m0
    dir1=my1m1-my0m1
    dir0=my1m0-my0m0
    indir1=my1m1-my1m0
    indir0=my0m1-my0m0
    #compute variances
    vtot=np.mean(pow((np.array(y1m1)-np.array(y0m0)-tot),2))
    vdir1=np.mean(pow((np.array(y1m1)-np.array(y0m1)-dir1),2))
    vdir0=np.mean(pow((np.array(y1m0)-np.array(y0m0)-dir0),2))
    vindir1=np.mean(pow((np.array(y1m1)-np.array(y1m0)-indir1),2))
    vindir0=np.mean(pow((np.array(y0m1)-np.array(y0m0)-indir0),2))
    ATE = [tot, dir1, dir0, indir1, indir0, vtot, vdir1, vdir0, vindir1, vindir0, sum(selall)]
    return ATE
        
        
    