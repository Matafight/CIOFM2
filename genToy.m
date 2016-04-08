function data=genToy()

%7¸öfeature,1000¸ösamples
numsample=500;
x=randn(numsample,7);

w=[]



x=[ones(numsample,1),x];


for i=1:8;
    for j=1:8;
        w=[w,x(:,j).*x(:,i)];
      
     end
end
B=zeros(8,8);


B(1,1:4)=1;
B(1:4,1)=1;



y=w*B(:);

data=equalBin(x,y);
savefile(data);

function newdata=equalBin(x,y)

data=[x,y]
[n,m]=size(x)

newdata=sortrows(data,m+1)

%divide the data into 5bins

binsize=n/5

for i = 1:5,
    newdata((i-1)*binsize+1:i*binsize,m+1)=i
end




function savefile(data)

[nrow,ncol]=size(data);

dataname=['toy'];
lentrain=nrow*0.7;

for i=0:19,

ind=randperm(nrow);
data=data(ind,:);
traindata=data(1:lentrain,:);
testdata=data(lentrain+1:end,:);

filename=['train_',dataname,'.',num2str(i)];
testfile=['test_',dataname,'.',num2str(i)];

fs=fopen(filename,'wt');
ftest=fopen(testfile,'wt');

for j = 1:lentrain,
	for k=1:ncol,
     fprintf(fs,'%f \t',traindata(j,k));
    end
	fprintf(fs,'\n');
  end
fclose(fs);

testlen=size(testdata,1);
for j=1:testlen,
	for k=1:ncol,
	fprintf(ftest,'%f \t',testdata(j,k));
	end
	fprintf(ftest,'\n');
end
fclose(ftest)
end














