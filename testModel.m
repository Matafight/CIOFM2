function [mze,mae]=testModel(traindata,oneret,testdata);

[wte,Yte]=transformData(testdata);

[wtr,Ytr]=transformData(traindata);
[mze,mae]=mypredict(wtr,Ytr,wte,Yte,oneret.B);






function [wte,Yte]=transformData(data);

[nrow,ncol]=size(data);
Xte=data(:,1:ncol-1);
Yte=data(:,ncol);

Xte=[ones(nrow,1),Xte];

wte=[];

for i=1:ncol;
     for j=1:ncol;
        wte=[wte,Xte(:,j).*Xte(:,i)];
     end
end





