% load the data
trainLen = 2000;
testLen = 2000;

data = load('slightlySimpleDataSet.txt');

% plot some of it
%figure;
%plot(data(1:1000));
%title('A sample of data');

a = 0.3; % leaking rate
rand( 'seed', 42 );

% main reservoirs
numberOfReservoirs = 2;
inSize = 1; outSize = 1;
resSize = 50;
[Win, W] = deal(cell(1,numberOfReservoirs));
for i = 1:numberOfReservoirs
    Win{i} = (rand(resSize,1+inSize)-0.5) .* 1;%(rand(resSize,1+inSize)-0.5) .* 1;
    W{i} = rand(resSize,resSize)-0.5;
    rhoW = abs(eigs(W{i},1,'LM'));
    W{i} = W{i} .* ( 1.25 /rhoW);
end

%reservoir mini
inSizeMini = resSize*numberOfReservoirs; outSizeMini = 1;
resSizeMini = 20;
WinMini = (rand(resSizeMini,1+inSizeMini)-0.5) .* 1;
WMini = rand(resSizeMini,resSizeMini)-0.5;
rhoW = abs(eigs(WMini,1,'LM'));
WMini = WMini .* ( 1.25 /rhoW);

%reservoir state matrices
XMini = zeros(resSizeMini,trainLen);

%X = zeros(1+inSize1+resSize1+inSize2+inSize2+inSizeMini+resSizeMini,trainLen);
%Output matrix
Yt = data(2:trainLen+1)';

% run the reservoirs with the data and collect the final XMini states
x = cell(1,numberOfReservoirs);
for i = 1:numberOfReservoirs
    x{i} = zeros(resSize,1);
end

%x1 = zeros(resSize1,1);
%x2 = zeros(resSize2,1);
xMini = zeros(resSizeMini,1);
for t = 1:trainLen
	u = data(t);
    
    for i = 1:numberOfReservoirs
        if ~rem(t,i)
            x{i} = (1-a)*x{i} + a*tanh( Win{i}*[1;u] + W{i}*x{i} );
        end
    end

   	xMini = (1-a)*xMini + a*tanh( WinMini*[1;cell2mat(x')] + WMini*xMini );
    
    XMini(:,t) = xMini;
end

% train the output
reg = 1e-8;  % regularization coefficient
XMini_T = XMini';
Wout = Yt*XMini_T * inv(XMini*XMini_T + reg*eye(resSizeMini));

% run the trained RoR in a generative mode
Y = zeros(outSizeMini,testLen);
u = data(trainLen+1);
for t = 1:testLen 
    
    for i = 1:numberOfReservoirs
        if ~rem(t,i)
            x{i} = (1-a)*x{i} + a*tanh( Win{i}*[1;u] + W{i}*x{i} );
        end
    end
    
   	xMini = (1-a)*xMini + a*tanh( WinMini*[1;cell2mat(x')] + WMini*xMini );
    
	y = Wout*xMini;
	Y(:,t) = y;

	u = data(trainLen+t+1);
end

errorLen = 500;
desired_output = data(trainLen+2:trainLen+errorLen+1)';
system_output = Y(1,1:errorLen);

mse = sum((desired_output-system_output).^2)./errorLen;
nmse = mean((desired_output - system_output).^2)/var(desired_output);
disp( ['MSE = ', num2str( mse ),' NMSE = ', num2str( nmse )] );

% plot the signal
% figure;
% plot( data(trainLen+2:trainLen+testLen+1), 'color', [0,0.75,0] );
% hold on;
% plot( Y', 'b' );
% hold off;
% axis tight;
% title('Target and generated signals y(n) starting at n=0');
% legend('Target signal', 'Free-running predicted signal');