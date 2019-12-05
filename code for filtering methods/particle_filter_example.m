clear;set(0,'defaultaxesfontsize',20);format long
%%% p14.m Particle Filter (SIRS), sin map 
%% setup

J=100;% number of steps
alpha=2.5;% dynamics determined by alpha
gamma=0.1;% observational noise variance is gamma^2
sigma=4e-1;% dynamics noise variance is sigma^2
C0=9e-2;% prior initial condition variance
m0=0;% prior initial condition mean
sd=1;rng(sd);% choose random number seed
N=100;% number of ensemble members

m=zeros(J,1);v=m;y=m;c=m;U=zeros(J,N);% pre-allocate
v(1)=m0+sqrt(C0)*randn;% initial truth
m(1)=10*randn;% initial mean/estimate
c(1)=10*C0;H=1;% initial covariance and observation operator
U(1,:)=m(1)+sqrt(c(1))*randn(1,N);m(1)=sum(U(1,:))/N;% initial ensemble
save_w = U;

%% solution % Assimilate!

for j=1:J  
    
    v(j+1)=alpha*sin(v(j)) + sigma*randn;% truth
   
    y(j)=H*v(j+1)+gamma*randn;% observation, this step generate observation
 
    Uhat=alpha*sin(U(j,:))+sigma*randn(1,N);% ensemble predict
    d=y(j)-H*Uhat;% ensemble innovation  
    
    what=exp(-1/2*(1/gamma^2*d.^2));% weight update
    w=what/sum(what);% normalize predict weights   
   
    
    %resampling
    ws=cumsum(w);% resample: compute cdf of weights
    for n=1:N
        ix=find(ws>rand,1,'first');% resample: draw rand \sim U[0,1] and 
        % find the index of the particle corresponding to the first time 
        % the cdf of the weights exceeds rand.
        U(j+1,n)=Uhat(ix);% resample: reset the nth particle to the one 
        % with the given index above
        
        if ix>1
            save_w(j+1,n)=ws(ix)-ws(ix-1);
        else
            save_w(j+1,n)=ws(ix);
        end
    end
    
    %output 
    m(j+1)=sum(U(j+1,:))/N;% estimator update
    c(j+1)=(U(j+1,:)-m(j+1))*(U(j+1,:)-m(j+1))'/N;% covariance update
    

end


%defind how many time steps
iter=100;

for tt=1:iter
    time_t = 0;
    %for each time step, gater the information for plot
    %we have the state as U
    bb=tabulate ( U(tt+1,:) ) ;%compute the frequency table
    state_x = bb(:,1);%select the state points for x
    weight_z = bb(:,2)/100; %select the weight of corresponding state
    time_t(1:length(state_x)) = tt;
    plot3(state_x,time_t',weight_z ,'.'); %plot the points
    axis([-10 10 0 iter 0 0.5]);
    hold on;
    grid on;
end

tt=1:iter;
% Add title and axis labels
xlabel('State');
ylabel('Time');
zlabel('Weight');
title('Implicit Particle Filter');

%true
measure_w(1:iter)=0;
hold on;
p2=plot3(v(tt+1),tt,measure_w  ,'k*','MarkerSize',7);


%observation
hold on;
p3=plot3(y(tt),tt, measure_w  ,'rs','MarkerSize',7);


hold off;
legend([p2,p3],{'true state','observation'});

%after resample, each point has 1/100 weight , count how many points are duplicated,


%example for one sigle time step
tt=15;
bb=tabulate ( U(tt+1,:) ) ;%compute the frequency table
state_x = bb(:,1);%select the state points for x
weight_z = bb(:,2)/100; %select the weight of corresponding state
figure;
plot(state_x,weight_z,'--');
axis([-1 4 0 0.5]);

