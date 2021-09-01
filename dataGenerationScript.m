x = 1:10000;

y = (sin(x/10)+sin(2*x/10)+sin(3*x/10))';
writematrix(y, 'dataSet.txt');

y = (sin(x/10)+sin(1.6*x/10)+sin(2.7*x/10))';
writematrix(y, 'simpleDataSet.txt');