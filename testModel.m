function [rato,mae]=testModel(traindata,oneret,data);

[wte,Yte]=transformData(data);

[wtr,Ytr]=transformData(traindata);
[rato,mae]=mypredict_another_cri(wtr,Ytr,wte,oneret.B,Yte);






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




