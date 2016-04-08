function newH=nesterov_SVM(data,label,options,rho,g,mylam,B)

% =========================================================================
%                          Nesterov SVM algorithm
% =========================================================================
% nesterov_SVM is a optimal gradient method with smooth technique for
% classical support vector machine (C-SVM) problem (1-norm soft margin).
% Hinge loss is nonsmooth and it is smoothed by Yurii Nesterov's technique
% in his 2004 Math. Programm paper.Then the SVM is solved using Nesterov's
% optimal gradient method.
%
% Problem:
% min f(w)=lambda*(1/2)*w'*w+|(e-Y*(X*w-1*gamma))+|;               (linear)
% min f(w)=lambda*(1/2)*w'*w+|(e-Y*(K(X',X)*Y*w-1*gamma))+|;    (nonlinear)
%
% Classification hyperplane: g(x)=x*w-gamma (linear) g(x)=K(x',X)*w-gamma
% For convinience, let A=Y*[X -e] for linear and Y*[K(X',X)*Y -e] for
% nonlinear, let W=[w gamma]', the hinge loss can be written as (e-AW)+.
% =========================================================================
% INPUT ARGUMENTS:
% data:          n*p data matrix, n is #sample, p is #dimension;
% label:         label vector corresponding to n samples;
% options:       parameter setting;
%                options.W0 is the pre-estimated solution;
%                options.W1 is the starting point of solution;
%                options.mu is the parameter of smoothing;
%                options.lambda is the weight of margin maximization;
%                options.kernel is the choice of kernel;
%                options.bias is the option to choose bias in classifier;
%                options.a, b and c are parameters of selected kernel.
% =========================================================================
% OUTPUT ARGUMENTS:
% classifier:    classifier.w is weight vector of SVM classifier;
%                classifier.gamma is bias of classification hyperplane;
% options:       parameter setting;
% =========================================================================

%initialization
[n,p]=size(data);
y=label;
y(y==0)=-1;
%default parameter setting
if ~isfield(options,{'bias'})
    options.bias=false;
end
if options.bias
    if ~isfield(options,{'W0'})
        options.W0=zeros(1+p,1);
    end
    if ~isfield(options,{'W1'})
        options.W1=zeros(1+p,1);
    end
else
    if ~isfield(options,{'W0'})
        options.W0=zeros(p,1);
    end
    if ~isfield(options,{'W1'})
        options.W1=zeros(p,1);
    end
end
if ~isfield(options,{'mu'})
    options.mu=1e-4;
end
if ~isfield(options,{'lambda'})
    options.lambda=1e-4;
end
if ~isfield(options,{'kernel'})
    options.kernel='none';
end
if ~isfield(options,{'a'})
    options.a=4;
end
if ~isfield(options,{'b'})
    options.b=4;
end
if ~isfield(options,{'c'})
    options.c=4;
end
W0=options.W0;
W1=options.W1;
mu=options.mu;
lambda=options.lambda;

[brow,bcol]=size(B);
%kernel selection and calculation
if strcmp(options.kernel,'linear')
    K=data*data';
    Y=diag(y);
    X=[K*Y,-ones(n,1)];
    d=n+1;
    clear K y;
elseif strcmp(options.kernel,'poly')
    a=options.a;
    b=options.b;
    c=options.c;
    K=(a.*data*data'+b).^c;
    Y=diag(y);
    X=[K*Y,-ones(n,1)];
    d=n+1;
    clear K y;
elseif strcmp(options.kernel,'rbf') || strcmp(options.kernel,'gaussian')
    a=options.a;
    G=L2_distance(data',data',0);
    G=G.^2;
    G=G./max(max(G));
    K=exp(-a.*G);
    Y=diag(y);
    X=[K*Y,-ones(n,1)];
    d=n+1;
    clear K y;
elseif strcmp(options.kernel,'sigmoid')
    a=options.a;
    b=options.b;
    K=tanh(a.*data*data'+b);
    Y=diag(y);
    X=[K*Y,-ones(n,1)];
    d=n+1;
    clear K y;
elseif strcmp(options.kernel,'none')
    Y=diag(y);
    X=[data,-ones(n,1)];
    d=p+1;
    clear y;
else
    fprintf('wrong kernel option, available selection: linear, poly, rbf(gaussian), sigmoid, none');
end
if ~options.bias
    X=X(:,1:end-1);
    d=d-1;
end
A=Y*X;
clear Y X;
A_l1=max(abs(A),[],2);
%为什么我感觉加上正则项前的系数完全没有道理呢？,c=1 lambda=rho/mylam.fif
%max_Q=max(abs(rho/mylam.fif+sum(A.*(A./repmat(A_l1,[1,d])))));
max_Q=n*max(abs(sum(A.*(A./repmat(A_l1,[1,d])))));
%sum_Al1=sum(A_l1);
D1=mu;
mu0=2*sqrt(2*D1/d);

%initialization of loop
iter=1;
%iter too many times 
iter_max=35*d;

%stop criteria of relative error
epsilon=1e-3;
delta_f=epsilon+1e-5;
memory=10;
W=W1;
f=[];
f_real=[];
%differential of objective f
diff_f=[];
%accumulated differential information of f according to nesterov's method
acc_diff_f=zeros(d,1);

VB=B(:);
VBgamma=g.gamma5(:);
rho=1e-4;
%gradient method loop
while iter<iter_max
    %update mu and Lipschitz constant
    mu=mu0/(iter+1);
    %Lipschitz constant of smoothed objective function
	%有问题吧，这里貌似加上了一部分？？
	
    L=rho/mylam.fif + max_Q/mu;
    %dual variable u
    temp1=A*W;
    U_mu=((1-temp1)./A_l1)./mu;
    P1=find(U_mu<=1 & U_mu>=0);
    P2=find(U_mu>1);
    U_mu(P2)=1;
    U_mu(U_mu<0)=0;
	
    %update objective value of last step
	%这里的求值要改，后面计算梯度的地方也要改
    %temp2=(lambda/2).*W(1:end-1)'*W(1:end-1);
	temp2=(rho/(2)).*(W)'*(W);
	
    hingeloss=1-temp1;
	
	%貌似没有用到f_real 的地方
    f_real=[f_real,temp2+sum(hingeloss(hingeloss>0))];
	%f应该是函数值,后面少减了一个x的无穷范数
	Alimit=diag(A_l1);
    %f=[f,temp2+hingeloss(P1)'*U_mu(P1)+sum(hingeloss(P2))-(mu/2)*((U_mu(P1))'*U_mu(P1)+length(P2))];
	f=[f,temp2+hingeloss(P1)'*U_mu(P1)+sum(hingeloss(P2))-(mu/2)*((U_mu(P1))'*Alimit(P1,P1)*U_mu(P1)+sum(sum(Alimit(P2,P2))))];
    clear hingeloss temp1 temp2;
    %stop criteria justification
    if iter>1
        %f_memory=mean(f(iter-min(memory,iter-1):iter-1));
        f_memory=f(iter-1);
        delta_f=abs((f(iter)-f_memory)/f_memory);
    end
    if delta_f<epsilon && iter>80
        break;
    end
    %differential of f
	%这是单次的梯度
    %diff_f=lambda.*[W(1:end-1);0]-(A(P1,:))'*U_mu(P1)-(sum(A(P2,:)))';
	diff_f=(rho).*(W-(VB+VBgamma/rho))-(A(P1,:))'*U_mu(P1)-(sum(A(P2,:)))';
    clear P1 P2 U_mu;
    %add weighted differential information into accumulated vector
	
	%这是累加的梯度
    acc_diff_f=acc_diff_f+((iter+1)/2).*diff_f;
    %update W
    W=(2/(iter+3)).*(W0-(1/L).*acc_diff_f)+((iter+1)/(iter+3)).*(W-(1/L).*diff_f);
    
	clear diff_f;
    %update counter
    iter=iter+1;
end

newH=reshape(W,brow,bcol);
%recover w and gamma from W
%if options.bias
%    w=W(1:d-1);
 %   gamma=W(d);
 %   classifier.w=w;
 %   classifier.gamma=gamma;
%else
 %   classifier.w=W;
%end
%options.X=data;
%options.y=label;

% figure,plot(f,'b');
% hold on;plot(f_real,'r');
% box on;
% legend('smooth objective value','original objective value');
% xlabel('Iteration');
% ylabel('Objective value');
% title('Objective convergence');