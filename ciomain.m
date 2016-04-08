function fin=ciomain(edata);

alphas=[0.01,0.5,0.99];
lambdas=linspace(0.01,0.3,50);
[B,ncol,wtr,wte,Ytr,Yte,dataname]=loadData(edata);

p1=ncol;
%initializatioin 

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

lengths.alpha=length(alphas);
lengths.lambda=length(lambdas);

[u,s,v]=svd(wtr,'econ');
grank=sum(sum(s~=0,2));
s=s(1:grank,1:grank);
u=u(:,1:grank);
v=v(:,1:grank);
svdw.s=s;
svdw.u=u;
svdw.v=v;

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

barrays=repmat(barray,1,lengths.lambda);

fin=repmat(barrays,lengths.alpha,1);

fs=fopen('result.txt','at');
fprintf(fs,'%s\n',dataname);
fclose(fs);


for ai=1:lengths.alpha;
    inalpha=alphas(ai);
    
    sprintf('fitting model for alpha = %f',inalpha);
    sprintf('fitting model for lambda =%f',lambdas(lengths.lambda));
    lambda.fir=lambdas(lengths.lambda)*(1-inalpha)*sqrt(p1);
    lambda.sec=lambdas(lengths.lambda)*(1-inalpha)*sqrt(p1);
    lambda.thi=inalpha*lambdas(lengths.lambda);
    lambda.four=lambda.thi;
    lambda.fif=lambda.thi;

    barrays(lengths.lambda).oneret=ADMMmat(wtr,svdw,Ytr,lambda,B,matD,matE,matF,matG,matH,g,rho);
    barrays(lengths.lambda).alpha=inalpha;
    barrays(lengths.lambda).lambda=lambdas(lengths.lambda);


    rato=mypredict(wtr,barrays(lengths.lambda).oneret.B,Ytr);
    ratotest=mypredict(wte,barrays(lengths.lambda).oneret.B,Yte);
    saveResult(inalpha,lambdas(lengths.lambda),rato,ratotest,dataname);

    for lam =lengths.lambda-1:-1:1
        sprintf('fitting model for alpha = %f ',inalpha);
        sprintf('fitting model for lambda =%f ',lambdas(lam));
        lambda.fir=lambdas(lam)*(1-inalpha)*sqrt(p1);
        lambda.sec=lambdas(lam)*(1-inalpha)*sqrt(p1);
        lambda.thi=inalpha*lambdas(lam);
        lambda.four=lambda.thi;
        lambda.fif=lambda.thi;

        barrays(lam).oneret=ADMMmat(wtr,svdw,Ytr,lambda,barrays(lam+1).oneret.B,barrays(lam+1).oneret.D,barrays(lam+1).oneret.E,barrays(lam+1).oneret.F,barrays(lam+1).oneret.G,barrays(lam+1).oneret.H,barrays(lam+1).oneret.glist,barrays(lam+1).oneret.rho);
        barrays(lam).alpha=inalpha;
        barrays(lam).alpha=lambdas(lam);

        rato=mypredict(wtr,barrays(lam).oneret.B,Ytr);
        ratotest=mypredict(wte,barrays(lam).oneret.B,Yte);
        saveResult(inalpha,lambdas(lam),rato,ratotest,dataname);
     end
     fin(ai,:)=barrays;
end


function saveResult(alpha,lambda,rato,ratotest,dataname)
    fs=fopen('result.txt','at');
    %fprintf(fs,'%s\n',dataname)
    fprintf(fs,'%s:\t','alpha');
    fprintf(fs,'%f\t',alpha);
    fprintf(fs,'%s:\t','lambda')
    fprintf(fs,'%f\n',lambda);
    fprintf(fs,'%f\n',rato);
    fprintf(fs,'%f\n',ratotest);
    fclose(fs);

