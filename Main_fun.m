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

rename=[dataname,'jieguo'];
save(rename,'result','bestpars')





function oneret=SetparaTrain(data,lambda,wtr,svdw,Ytr)


[nrow,ncol]=size(data);
%[B,ncol,wtr,wte,Ytr,Yte,dataname]=loadData(data);

B=zeros(ncol,ncol);
matD=B;
matE=B;
matF=B;
matG=B;
matH=B;
g.gamma1=B;
g.gamma2=B;
g.gamma3=B;
g.gamma4=B;
g.gamma5=B;
rho=1;

oneret =ADMMmat(wtr,svdw,Ytr,lambda,B,matD,matE,matF,matG,matH,g,rho);



function [svdw,wtr,wte,Ytr,Yte]=loadMyData(data,testdata);



[datam,datan]=size(data);
testrow=size(testdata,1);

Xtr=data(:,1:datan-1);
Ytr=data(:,datan);

Xte=testdata(:,1:datan-1);
Yte=testdata(:,datan);

Xtr=[ones(datam,1),Xtr];
Xte=[ones(testrow,1),Xte];

B=zeros(datan+1,datan+1);   

wtr=[];
wte=[];
for  i=1:datan
    for j =1:datan
   wtr=[wtr,Xtr(:,j).*Xtr(:,i)];
   wte=[wte,Xte(:,j).*Xte(:,i)];
    end
end

[u,s,v]=svd(wtr,'econ');
grank=sum(sum(s~=0,2));
s=s(1:grank,1:grank);
u=u(:,1:grank);
v=v(:,1:grank);
svdw.s=s;
svdw.u=u;
svdw.v=v;


