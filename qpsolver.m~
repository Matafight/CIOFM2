function W=qpsolver(X,rho,g,lambda,newB)

[xrow,xcol]=size(X);
[brow,bcol]=size(newB);
B=reshape(newB,xcol,1);
vg=reshape(g.gamma5,xcol,1);
A=B+vg/rho;

%construct q
q=(1-X*A);
%construct K
K=X*X';
lb=zeros(xrow,1);
lu=lambda.fif/rho.*ones(xrow,1);
a=zeros(xrow,1);
b=0;

 opt = struct('TolKKT', 1e-8/100, 'MaxIter', 500, 'verb', 0);

[alpha,~] = libqp_gsmo(K, q, a, b, lb, lu,[],opt);

W=(X'*alpha)/rho + A;
W=reshape(W,brow,bcol);

