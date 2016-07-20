function [newoneret,bestpar]=crossValid(data)

data = transform_data(data);
[nrow,nnewcol]=size(data);

ncol = sqrt(nnewcol-1);

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

%barray.alpha=0;
%barray.lambda=0;
%barray.oneret.finB=B;
%barray.oneret.B=B;
%barray.oneret.D=matD;
%barray.oneret.E=matE;
%barray.oneret.F=matF;
%barray.oneret.rho=rho;
%barray.oneret.glist=g;
%barray.oneret.iters=0;


%5 fold 

ind=randperm(nrow);
data=data(ind,:);


%parametersmaller=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1];
parametersmaller = [1e-3,1e-2,1e-1,1,10,100,1000];
parameter=[1e-3,1e-2,1e-1,1,10,100,1000];

bestret.par1=0;
bestret.par2=0;
bestret.par3=0;
bestret.par4=0;


bestrato = 0;
numparasmaller = length(parametersmaller);
numpara = length(parameter);
 for par1=1:numparasmaller;
      for par2=1:numparasmaller;
          for par3=1:numparasmaller;
             for par4=1:numpara;
                   aboverato=0;
                    ratotest=0;
					indices=crossvalind('kfold',nrow,5);
                 for i=1:5;
                    test=(indices==i);
                    train=~test;
                    train_data=data(train,:);
                    test_data=data(test,:);
                    lambda.fir=parametersmaller(par1);
                    lambda.sec=parametersmaller(par1);
                    lambda.thi=parametersmaller(par2);
                    %lambda.four=parasmaller(par3);
                    lambda.four=parametersmaller(par3);
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
					[mze,mae]=mypredict_another_cri(wtr,Ytr,wte,Yte,newoneret.finB);
                    ratotest=ratotest+mze;

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
                  if (averrato > bestrato);
                      bestrato = averrato ;
                      bestret.par1 = parametersmaller(par1);
                      bestret.par2 = parametersmaller(par2);
                      bestret.par3 = parametersmaller(par3);
                      bestret.par4 = parameter(par4);

                  end
           
          end

          end
      end

end
bestpar=bestret;

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
[mze,mae]=mypredict_another_cri(wt,Y,wt,Y,newoneret.finB);







