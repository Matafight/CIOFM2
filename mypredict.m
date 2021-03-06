function [mze,mae]=mypredict(wtr,ytr,wte,yte,B);


[meanfeat,uniqueY]=separateClasses(wtr,ytr);

bk=[];
numcls=size(meanfeat,1);
for i=1:numcls-1,
    bk=[bk;(meanfeat(i,:)+meanfeat(i+1,:))/2];
end


%B is a matrix .
[m,n]=size(wte);
preval=wte*reshape(B,n,1);

bk=bk*reshape(B,n,1);
bk=[bk;inf];

preval=repmat(preval,1,numcls);
bk=bk';
bk=repmat(bk,m,1);

tempdiff=preval-bk;
%find the first element that are below 0
[val,index]=max(tempdiff<=0,[],2);


diffval=index-yte;
 
%return MZE,this is the errorrato
ind=find(diffval == 0);
mze=length(ind)/m;

%return MAE
tmae=abs(index-yte);
mae=sum(tmae)/m;



%help function for update H,about the hinge loss , separate data for different class
function [meanfeat,uniqueY] = separateClasses(w,y);
%y is a vector,sort in ascending order,Y=y(I)
[Y,I]=sort(y);
W=w(I,:);
[numEx,numfeat]=size(W);
uniqueY=unique(Y);
numrealCls=length(uniqueY);
%calculate the mean features of different classes
%numclass is the number of different classes,cla is the class in ascending order 
%note that numclasses can be zero due to the continuous number in cla
%[numclass,cla]=hist(Y);      hist may not work here 
%lencla=length(numclass);

%这个代码写的很好！！,不过没有考虑样本数为0的类
d=diff([Y;max(Y)+1]);
count=diff(find([1;d]));
numclass=count;
cla=Y(find(d));

lencla=length(numclass);
meanfeat=zeros(numrealCls,numfeat);
startpos=1;     
realClspos=1;
for i =1:lencla;
    if(numclass(i,:)~=0);
       meanfeat(realClspos,:)=mean(W(startpos:startpos+numclass(i,:)-1,:));
       realClspos=realClspos+1;
       startpos=startpos+numclass(i,:);
    end
end
