import numpy as np
import pandas as pd


rawdata=pd.read_table('NRTI_DataSet.txt',delimiter = "\t",header=0)

rawdata.drop(rawdata.columns[[0,2,3,4,5,6]],axis=1,inplace = True)

attrval=rawdata.ix[:,1:241]
#operate each row of attrval

#some bugs here,which make it a bad representation
attrval.ix[:,0:240]=np.where(attrval.ix[:,0:240] == '.',None,attrval.ix[:,0:240])
attrval['label']=rawdata['3TC']

attrval = attrval.dropna(how="any")

attrval.ix[:,0:240]=np.where(attrval.ix[:,0:240] =='-',0,1)

lengthattr=len(attrval.columns)
binwidth=10
numbins=int(lengthattr/binwidth)

binattr=pd.DataFrame()
for i in range(0,numbins):
    sumattr=np.sum(attrval.ix[:,range(i*binwidth,(i+1)*binwidth)],axis=1)
    binattr[i]=sumattr
binattr['label'] = attrval.label

binattr.sort(columns='label',axis=0,inplace=True)

#assign new index

lengthsample=len(binattr.index)
binattr.index=range(1266)

width=int(lengthsample/5)
for i in range(1,6):
    binattr.loc[range((i-1)*width,i*width),'label']=i

binattr.loc[binattr['label']==200,'label']=5

dataname = "Genedata."
randperm=np.random.permutation(lengthsample)
newX=binattr.loc[randperm]

#generate files in specific dir

for i in range(20):
    randperm=np.random.permutation(lengthsample)
    newX=binattr.loc[randperm]
    trainItem=int(lengthsample*0.7)
    trainData=newX.iloc[range(trainItem)]
    testData = newX.iloc[range(trainItem,lengthsample)]
    #write to file
    trainname="../compareCodecomplete_HingeLoss/dataset/Genedata/train_"+dataname+str(i)
    np.savetxt(trainname,trainData.values,fmt="%f")
    testname="../compareCodecomplete_HingeLoss/dataset/Genedata/test_"+dataname +str(i)
    np.savetxt(testname,testData.values,fmt="%f")
        
    orcatrain="/home/guo/ordinal/orca/datasets/Genedata/matlab/train_" + dataname +str(i)
    orcatest="/home/guo/ordinal/orca/datasets/Genedata/matlab/test_" + dataname +str(i)
    np.savetxt(orcatrain,trainData.values,fmt="%f")
    np.savetxt(orcatest,testData.values,fmt="%f")


