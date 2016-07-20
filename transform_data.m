function data = transform_data(data)

[nrow,ncol] = size(data);
X = data(:,1:ncol-1);
Y = data(:,ncol);
X = [ones(nrow,1),X];
datalen = [];
for i = 1:ncol;
    for j = 1:ncol;
            datalen=[datalen,X(:,j).*X(:,i)];
    end
end
data = [datalen,Y];

