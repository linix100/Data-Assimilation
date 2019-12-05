clear;set(0,'defaultaxesfontsize',20);format long
%%% Particle Filter (SIRS), double-well
%% setup
J=100;% number of steps
alpha=2.5;% dynamics determined by alpha
gamma=0.1;% observational noise variance is gamma^2
sigma=0.3;% dynamics noise variance is sigma^2
C0=0.1;% prior initial condition variance
m0=0;% prior initial condition mean

N=20;% number of ensemble members

err_sum=0;
k=1000; % number of times
N_on_target = 0;  %count number of times within 99% of N(v, 0.01)
AEFF=0; %average effective sample size
for ei=1:k
    sd=floor(abs(k*10*randn) );rng(sd);% Choose random number seed
    disp('iteration=');
    disp(ei);
    
    m=zeros(J,1);v=m;z=m;z(1)=0;c=m;U=zeros(J,N);% pre-allocate
    v(1)=m0+sqrt(C0)*randn;% initial truth
    m(1)=2*randn;% initial mean/estimate
    c(1)=10*C0;H=1;% initial covariance and observation operator
    U(1,:)=m(1)+sqrt(c(1))*randn(1,N);m(1)=sum(U(1,:))/N;% initial ensemble
    tau=0.02;st=sigma*sqrt(tau);% time discretization is tau
  
    for j=1:J  
        %v(j+1)=v(j)+tau*alpha*(v(j)-v(j)^3) + st*randn;% truth

        v(j+1)=v(j)+tau*alpha*(v(j)-v(j)^3) +st*randn;% truth
        z(j+1)=H*v(j+1) + gamma*randn;% observation

        Uhat=U(j,:)+tau*alpha*(U(j,:)-U(j,:).^3)+st*randn(1,N);%ensemble predict
        d=z(j+1)-H*Uhat;% ensemble innovation  
        what=exp(-1/2*(1/gamma^2*d.^2));% weight update
        w=what/sum(what); randn(1,N);% normalize predict weights   

        EFF=1.0/sum(w.^2);
        ws=cumsum(w);% resample: compute cdf of weights
        for n=1:N
            ix=find(ws>rand,1,'first');% resample (i)
            U(j+1,n)=Uhat(ix);% resample (ii)
        end

        m(j+1)=sum(U(j+1,:))/N;% estimator update
        c(j+1)=(U(j+1,:)-m(j+1))*(U(j+1,:)-m(j+1))'/N;% covariance update
     
    end
    AEFF = AEFF+EFF;
    a=(m(j+1)-v(j+1) )^2;
    err_sum = err_sum+ a;
    if and(m(j+1)<=v(j+1)+gamma , m(j+1)>=v(j+1)-gamma )
        N_on_target = N_on_target+1;
    end
    disp(m(j+1)-v(j+1))
   
end
disp('average effective sample size:')
disp(AEFF/k)

rmse=sqrt(err_sum/k);
disp('rmse=')
disp(rmse)
accuracy_rate = N_on_target /k;
disp('accuracy=')
disp(accuracy_rate)
disp('N_on_target')
disp(N_on_target)
