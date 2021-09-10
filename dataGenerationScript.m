x = 1:10000;
y = (sin(x/10)+sin(2*x/10)+sin(3*x/10)+sin(4*x/10)+sin(5*x/10))';
writematrix(y, 'dataSet.txt');