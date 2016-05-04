function result=Main_fun(dataname)

totalrato=0;
totalmze=[];
totalmae=[];

addpath ./libqp/matlab/
mydir='./dataset/';
bestpars=[];
for i = 0:19,

i
%call CrossV to find the best hyperpara and return the trained coefficient ,then Call testmodel to test the result
trainname=[mydir,dataname,'/','train_',dataname,'.',num2str(i)];
traindata=load(trainname);
testname=[mydir,dataname,'/','test_',dataname,'.',num2str(i)];
testdata=load(testname);

[bestret,bestpar]=CrossV(traindata);
bestpars=[bestpars,bestpar];
[mzerato,mae]=testModel(traindata,bestret,testdata);
mzerato
totalmze=[totalmze,mzerato];
totalmae=[totalmae,mae];

end

result.mze=sum(totalmze)/20;
result.mzestd=std(totalmze);
result.mae=sum(totalmae)/20;
result.maestd=std(totalmae);
resultdir = './Result/'
rename=[resultdir,dataname,'jieguo'];
save(rename,'result','bestpars')





