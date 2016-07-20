function ret = learn_coefficients(dataname)

mydir = './dataset/';
addpath ./libqp/matlab
trainname = [mydir,dataname,'/','train_',dataname,'.',num2str(0)];
testname = [mydir,dataname,'/','test_',dataname,'.',num2str(1)];
traindata = load(trainname);
testdata = load(testname);

data = [traindata;testdata];

[bestret,bestpar] = CrossV(data);
ret = bestret.finB;
