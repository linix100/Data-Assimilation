clear;set(0,'defaultaxesfontsize',20);format long
%%% Kalman Filter
%% setup

J=1e3;% number of steps
N=2;% dimension of state
I=eye(N);% identity operator
gamma=1;% observational noise variance is gamma^2*I
sigma=1;% dynamics noise variance is sigma^2*I
C0=eye(2);% prior initial condition variance
m0=[0;0];% prior initial condition mean
sd=10;rng(sd);% choose random number seed
A=[0 1;-1 0];% dynamics determined by A

m=zeros(N,J);v=m;y=zeros(J,1);c=zeros(N,N,J);% pre-allocate
v(:,1)=m0+sqrtm(C0)*randn(N,1);% initial truth
m(:,1)=10*randn(N,1);% initial mean/estimate
c(:,:,1)=100*C0;% initial covariance
H=[1,0];% observation operator

%% solution % assimilate!

for j=1:J   
    v(:,j+1)=A*v(:,j) + sigma*randn(N,1);% truth
    y(j)=H*v(:,j+1)+gamma*randn;% observation
    
    mhat=A*m(:,j);% estimator predict
    chat=A*c(:,:,j)*A'+sigma^2*I;% covariance predict  
    
    d=y(j)-H*mhat;% innovation
    K=(chat*H')/(H*chat*H'+gamma^2);% Kalman gain
    m(:,j+1)=mhat+K*d;% estimator update
    c(:,:,j+1)=(I-K*H)*chat;% covariance update    
end

figure;
js=21; % how many time steps we need to show
plot([0:js-1],v(2,1:js)); %plot the true state
hold;
plot([0:js-1],m(2,1:js),'m'); % estimated state
plot([0:js-1],m(2,1:js)+reshape(sqrt(c(2,2,1:js)),1,js),'r--'); %upper
plot([0:js-1],m(2,1:js)-reshape(sqrt(c(2,2,1:js)),1,js),'r--');% lower
hold;
grid;
xlabel('iteration, j');
title('Kalman Filter');

figure;
plot([0:J],reshape(c(1,1,:)+c(2,2,:),J+1,1));
hold;
plot([0:J],cumsum(reshape(c(1,1,:)+c(2,2,:),J+1,1))./[1:J+1]','m', ...
'Linewidth',2); 
grid; 
hold;
xlabel('iteration, j');
axis([1 1000 0 50]);
title('Kalman Filter Covariance,');

figure;
plot([0:J],sum((v-m).^2));
hold;
%plot([0:J],cumsum(sum((v-m).^2))./[1:J+1],'m','Linewidth',2);
grid;
hold;
xlabel('iteration, j');
axis([1 1000 0 50]);
title('Kalman Filter Error,')
