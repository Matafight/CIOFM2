function result=Main_fun(dataname)

totalrato=0;
totalacc=[];
totalmae=[];

addpath ./libqp/matlab/
mydir='./dataset/';
bestpars=[];
for i = 1:20,

i
%call CrossV to find the best hyperpara and return the trained coefficient ,then Call testmodel to test the result
trainname=[mydir,dataname,'/','train_',dataname,'.',num2str(i)];
traindata=load(trainname);
testname=[mydir,dataname,'/','test_',dataname,'.',num2str(i)];
testdata=load(testname);

[bestret,bestpar]=CrossV(traindata);
bestpars=[bestpars,bestpar];
[accrato,mae]=testModel(traindata,bestret,testdata);
accrato
totalacc=[totalacc,accrato];
totalmae=[totalmae,mae];

end

result.acc=sum(totalacc)/20;
result.accstd=std(totalacc);
result.mae=sum(totalmae)/20;
result.maestd=std(totalmae);
resultdir = './Result/'
rename=[resultdir,dataname,'jieguo'];
save(rename,'result','bestpars')





