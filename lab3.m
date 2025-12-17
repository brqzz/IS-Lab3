clc
clear all


x = 0.1:1/22:1;
x_in=0.1:1/200:1;
y = ((1+0.6*sin(2*pi*x/0.7))+0.3*sin(2*pi*x))/2;
y_out = ((1+0.6*sin(2*pi*x_in/0.7))+0.3*sin(2*pi*x_in))/2;

figure(1)
plot(x_in,y_out)
grid on
hold on

%First RBF
c1=rand();
r1=rand();

%Second RBF
c2=rand();
r2=rand();

F1 = exp(-(x-c1).^2/(2.*r2^2));
F2 = exp(-(x-c2).^2/(2.*r2^2));

% Initialize output layer weights
w1 = rand();
w2 = rand();
w0 = rand();

% Learning rate
learningRate = 0.1;
learningRateCR=0.01;

%Number of learning cycles
lc = 10000;

%Errors
%----------------------------------------
%Dif between desired and output
err=0;

%Total err over epoch
sqr_err=0;

%Mean squared err
ms_err=0;

%Mean squared err over training
prev_err=zeros(1, lc);
%----------------------------------------
for epoch = 1:lc
    sqr_err=0;
    for i = 1:length(x)
        
        %Gaussian basis for each RBF neuron
        phi1 = exp(-((x(i)-c1)^2)/(2*r1^2));
        phi2 = exp(-((x(i)-c2)^2)/(2*r2^2));

        %RBF network output
        y_pred(i) = w1*phi1 + w2*phi2 + w0;
        
        %err
        err = y(i) - y_pred(i);
        sqr_err = sqr_err + err^2;
        
        % Update the weights and centers based on the error
        w1 = w1 + learningRate * err * phi1;
        w2 = w2 + learningRate * err * phi2;
        w0 = w0 + learningRate * err;
        c1 = c1 + learningRateCR * err * w1 * phi1 * ((x(i)-c1)/(r1^2));
        c2 = c2 + learningRateCR * err * w2 * phi2 * ((x(i)-c2)/(r2^2));
        r1 = r1 + learningRateCR * err * w1 * phi1 * (((x(i)-c1)^2)/(r1^3));
        r2 = r2 + learningRateCR * err * w2 * phi2 * (((x(i)-c2)^2)/(r2^3));

    end
    %Mean square error
    ms_err = sqr_err/length(x);

    %Mean square error over training
    prev_err(epoch) = ms_err;
end

%Creating test environment
Y_test = zeros(size(x_in));
for i=1:length(x_in)
    phi1_test = exp(-((x_in(i)-c1)^2)/(2*r1^2));
    phi2_test = exp(-((x_in(i)-c2)^2)/(2*r2^2));

    Y_test(i) = w1*phi1_test + w2*phi2_test + w0;
end

plot(x_in, Y_test, 'r', 'LineWidth', 1.5)
hold off
legend('Target','Predicted')