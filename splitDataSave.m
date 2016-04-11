function splitDataSave(data)

savefile(data);




function savefile(data)

[nrow,ncol]=size(data);

dataname=['FG-NET'];
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














