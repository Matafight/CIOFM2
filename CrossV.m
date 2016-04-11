function [newoneret,bestpar]=crossValid(data)

[nrow,ncol]=size(data);
X=data(:,1:ncol-1);
Y=data(:,ncol);
X=[ones(nrow,1),X];
datalen=[];
for i=1:ncol;
    for j=1:ncol;
        datalen=[datalen,X(:,j).*X(:,i)];
     end
end
data=[datalen,Y];
nnewcol=size(data,2);


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

barray.alpha=0;
barray.lambda=0;
barray.oneret.finB=B;
barray.oneret.B=B;
barray.oneret.D=matD;
barray.oneret.E=matE;
barray.oneret.F=matF;
barray.oneret.rho=rho;
barray.oneret.glist=g;
barray.oneret.iters=0;


%5 fold 

ind=randperm(nrow);
data=data(ind,:);

%我觉得overfitting是这个参数的锅，
parasmaller=[1e-4,1e-3,1e-2,1e-1,1];
parameter=[1e-2,1e-1,1,10,100];

ret.par1=0;
ret.par2=0;
ret.par3=0;
ret.par4=0;
ret.rato=0;
bestret=repmat(ret,625,1);
iterk=0;
 for par1=1:5;
      for par2=1:5;
          for par3=1:5;
             for par4=1:5;
                   iterk=iterk+1;    
                   aboverato=0;
              % for runit=1:3;% run 3 times
                    ratotest=0;
                    averrate=0;
					indices=crossvalind('kfold',nrow,5);
                 for i=1:5;
                    test=(indices==i);
                    train=~test;
                    train_data=data(train,:);
                    test_data=data(test,:);
                    lambda.fir=parameter(par1);
                    lambda.sec=parameter(par1);
                    lambda.thi=parameter(par2);
                    lambda.four=parasmaller(par3);
                    lambda.fif=parameter(par4);
				
					wtr=train_data(:,1:nnewcol-1);
					Ytr=train_data(:,nnewcol);
					
					wte=test_data(:,1:nnewcol-1);
					Yte=test_data(:,nnewcol);
					
					[u,s,v]=svd(wtr,'econ');
					grank=sum(sum(s~=0,2));
					s=s(1:grank,1:grank);
					u=u(:,1:grank);
					v=v(:,1:grank);
					svdw.s=s;
					svdw.u=u;
					svdw.v=v;
					
					newoneret=ADMMmat(wtr,svdw,Ytr,lambda,B,matD,matE,matF,matG,matH,g,rho);
					[acc,mae]=mypredict_another_cri(wtr,Ytr,wte,newoneret.B,Yte);
                    ratotest=ratotest+acc;

                     %to accelerate the calculation 
                    B=newoneret.B;
                    matD=newoneret.D;
                    matE=newoneret.E;
                    matF=newoneret.F;
                    matG=newoneret.G;
                    matH=newoneret.H;
                    g=newoneret.glist;
                    rho=newoneret.rho;
                  end
                 
                  averrato=ratotest/5;
                 
                 % aboverato=aboverato+averrato;
             % end
                  bestret(iterk).par1=parameter(par1);
                  bestret(iterk).par2=parameter(par2);
                  bestret(iterk).par3=parasmaller(par3);
                  bestret(iterk).par4=parameter(par4);
                  bestret(iterk).rato=averrato;
				   %bestret(iterk).rato=aboverato/3;
                   %aboverato/3
          end

          end
      end

end
bestpar=seleBest(bestret);

wt=data(:,1:nnewcol-1);
Y=data(:,nnewcol);
[u,s,v]=svd(wt,'econ');
grank=sum(sum(s~=0,2));
s=s(1:grank,1:grank);
u=u(:,1:grank);
v=v(:,1:grank);
svdw.s=s;
svdw.u=u;
svdw.v=v;

lambda.fir=bestpar.par1;
lambda.sec=bestpar.par1;
lambda.thi=bestpar.par2;
lambda.four=bestpar.par3;
lambda.fif=bestpar.par4;

newoneret=ADMMmat(wt,svdw,Y,lambda,B,matD,matE,matF,matG,matH,g,rho);
%test if it is overfittng
[acc,mae]=mypredict_another_cri(wt,Y,wt,newoneret.B,Y);
acc



function bestpar=seleBest(bestret);

[row,col]=size(bestret);
bestrow=1;

for i = 1:row;
    if(bestret(i).rato>bestret(bestrow).rato);
        bestrow=i;
     end
end
bestpar=bestret(bestrow)






