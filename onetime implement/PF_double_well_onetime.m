%%% SIR(sequential importance resampling) double well single time 
clear;set(0,'defaultaxesfontsize',20);format long
%% setup

J=2000;% number of steps
alpha=4;% dynamics determined by alpha
gamma=0.06;% observational noise variance is gamma^2
sigma=0.4;% dynamics noise variance is sigma^2
C0=0.1;% prior initial condition variance
m0=0;% prior initial condition mean
sd=1;rng(sd);% Choose random number seed
N=20;% number of ensemble members

m=zeros(J,1);v=m;z=m;z(1)=0;c=m;U=zeros(J+1,N);% pre-allocate
v(1)=m0+sqrt(C0)*randn;% initial truth
m(1)=2*randn;% initial mean/estimate
c(1)=10*C0;H=1;% initial covariance and observation operator
U(1,:)=m(1)+sqrt(c(1))*randn(1,N);m(1)=sum(U(1,:))/N;% initial ensemble
tau=0.02;st=sigma*sqrt(tau);% time discretization is tau
EFF=0; %effective sample size

%%
for j=1:J 
    
    %step jump
    %if mod(j,500)==0
      %  if v(j)<0
        %    v(j)=1
        %else
          %  v(j)=-1
        %end
    %end
    
    v(j+1)=v(j)+tau*alpha*(v(j)-v(j)^3) + st*randn;% truth

    %v(j+1)=v(j)+tau*alpha*(v(j)-v(j)^3) ;% truth
    z(j+1)=H*v(j+1) + gamma*randn;% observation

    Uhat=U(j,:)+tau*alpha*(U(j,:)-U(j,:).^3)+st*randn(1,N);%ensemble predict
    d=z(j+1)-H*Uhat;% ensemble innovation  
    what=exp( -0.5 * (  (1.0/gamma^2) *d.^2 )   );% weight update
    w=what/sum(what); randn(1,N);% normalize predict weights   
 
    EFF=EFF+(1.0/sum(w.^2));
    ws=cumsum(w);% resample: compute cdf of weights
    for n=1:N
        ix=find(ws>rand,1,'first');% resample (i)
        U(j+1,n)=Uhat(ix);% resample (ii)
    end
    
    m(j+1)=sum(U(j+1,:))/N;% estimator update
    c(j+1)=(U(j+1,:)-m(j+1))*(U(j+1,:)-m(j+1))'/N;% covariance update
     
end

disp('SIR average effective sample size:')
disp(EFF/J)
%defind how many time steps
iter=1000;

for tt=1:iter
    time_t = 0;
    %for each time step, gater the information for plot
    %we have the state as U
    bb=tabulate ( U(tt+1,:) ) ;%compute the frequency table
    state_x = bb(:,1);%select the state points for x
    weight_z = bb(:,2)/N; %select the weight of corresponding state
    time_t(1:length(state_x)) = tt;
    plot3(state_x,time_t',weight_z ,'.'); %plot the points
    axis([-2 2 0 iter 0 0.4]);
    hold on;
    grid on;
end

tt=1:iter;
% Add title and axis labels
xlabel('State');
ylabel('Time');
zlabel('Weight');
title('SIR Particle Filter');

%true
measure_w(1:iter)=0;
hold on;
p2=plot3(v(tt+1),tt,measure_w  ,'k*','MarkerSize',7);

%observation
hold on;
z(1)=m(1);
p3=plot3(z(tt),tt, measure_w  ,'rs','MarkerSize',7);

hold off;
legend([p2,p3],{'true state','observation'});

%example for one sigle time step
tt=34;
bb=tabulate ( U(tt+1,:) ) ;%compute the frequency table
state_x = bb(:,1);%select the state points for x
weight_z = bb(:,2)/N; %select the weight of corresponding state
figure;
plot(state_x,weight_z,'.');
axis([-2 2 0 0.4]);
ylabel('weight')
xlabel('x')
title('distribution of particles at t=34')


js=1000;% plot truth, mean, standard deviation, observations
figure;
p4 =plot([0:js-1],v(1:js)); %truth
hold;
p5=plot([0:js-1],m(1:js),'m'); %mean
p6=plot([1:js-1],z(1:js-1),'kx'); %observation
hold;grid;xlabel('t');
ylabel('x')
title('SIR Particle Filter ');
legend([p4,p5,p6],{'truth','mean','data'});


figure;
plot([0:J],(v-m).^2);
hold;
plot([0:J],cumsum((v-m).^2)./[1:J+1]','m','Linewidth',2);
grid
hold;
xlabel('t');title('SIR Particle Filter Error')
ylabel('RSE')

disp('Root square error')
a = cumsum((v-m).^2)./[1:J+1]';

disp( sqrt( a(length(a) )   )  )


