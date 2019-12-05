clear;set(0,'defaultaxesfontsize',20);format long
%%% Auxiliary Particle Filter deterministic model
%% setup

J=100;% number of steps
alpha=2.5;% dynamics determined by alpha
gamma=0.1;% observational noise variance is gamma^2
sigma=0.3;% dynamics noise variance is sigma^2
C0=0.1;% prior initial condition variance
m0=0;% prior initial condition mean

N=20;% number of ensemble members

err_sum=0; %accumulate square error
k=1000; % repeatedly runing times
N_on_target = 0;  %count number of times within 99% of N(v, 0.01)
AEFF=0; %effective sample size

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

        v(j+1)=v(j)+tau*alpha*(v(j)-v(j)^3)+st*randn ;% truth
        z(j+1)=H*v(j+1) + gamma * randn;% observation

        %calculate the mu which is the mean, we can use the formula mean m(j)
        Uhat0 = U(j,:)+tau*alpha*(U(j,:)-U(j,:).^3);
        %update the weight
        d= z(j+1) - H * Uhat0;
        what=exp(-1/2*(1/gamma^2 * d.^2)) ;
        w =  what/ sum( what);

        %resample for x 
        ws=cumsum(w);% resample: compute cdf of weights
        for n=1:N
            ix=find(ws>rand,1,'first');% resample: draw rand \sim U[0,1] and 
            % find the index of the particle corresponding to the first time 
            % the cdf of the weights exceeds rand.
            U(j,n)=Uhat0(ix);% resample: reset the nth particle to the one 
            % with the given index above
        end

        Uhat=U(j,:)+tau*alpha*(U(j,:)-U(j,:).^3)+st*randn(1,N); % ensemble predict all path
        d=z(j+1)-H*Uhat;% ensemble innovation  
        what=exp(-1/2*(1/gamma^2*d.^2));% weight update 

        Uhat2= U(j,:)+tau*alpha*(U(j,:)-U(j,:).^3);
        d2 =z(j+1)-H*Uhat2;
        what2 =exp(-1/2*(1/gamma^2*d2.^2));

        what = what./what2;

        w= what/sum(what);% normalize predict weights   

        EFF=1.0/sum(w.^2);

        %resampling
        ws=cumsum(w);% resample: compute cdf of weights
        for n=1:N
            ix=find(ws>rand,1,'first');% resample: draw rand \sim U[0,1] and 
            % find the index of the particle corresponding to the first time 
            % the cdf of the weights exceeds rand.
            U(j+1,n)=Uhat(ix);% resample: reset the nth particle to the one 
            % with the given index above

        end

        %output 
        m(j+1)=sum(U(j+1,:))/N;% estimator update
        c(j+1)=(U(j+1,:)-m(j+1))*(U(j+1,:)-m(j+1))' / N ;

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

