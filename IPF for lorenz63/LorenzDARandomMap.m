%% 
%clear all
%close all
%clc
rng('default');

%% ....................................................................

%% Settings
% .........................................................................
%dt        = 0.01;                        % discrete time
dt        = 0.01;
%Steps  = 4*100; 
Steps = 300;
%po       = [3.6314    6.6136   10.6044]';
po = [-8.416  -10.1101  24.665 ]';
%po       = [8 2 10]';
%po=[15  20  32]';
t          = dt*(0:Steps);
% .........................................................................

%% Experiment
% ........................................................................
g           =  0.3*sqrt(6);
%g=0.1;
xExp   =  EulerLorenz(dt,Steps,po,g);
%xExp2 =  EulerLorenz2(dt, Steps,po,g);
%figure
%plot3(xExp(1,:), xExp(2,:),xExp(3,:) ,'r'), hold on, plot3(xExp2(1,:), xExp2(2,:),xExp2(3,:) ,'bo')
%xExp2 =  EulerLorenz(dt,Steps,po,g); %  run again to see effect of filter
% ........................................................................
 
%% Observations
% ........................................................................
s       = 0.25*sqrt(6);% setting for observation error
%s=0.1;
%Gap  = 10;
Gap = 2;
%Gap = 2;
xObs = xExp(:,Gap+1:Gap:end);
xObs = xObs+sqrt(s)*randn(size(xObs));
tObs = dt*(Gap:Gap:Steps);
% ........................................................................

rng(4);
%% Data Assimilation
% ........................................................................
P = 40;                      % number of particles
nA = Steps/Gap; % Number of assimilations

% parameters for minimization
maxiter = 2000;
tol = 0.005;

% initialize particles
Xn = po*ones(1,P); 
%Xn            = (po + 3*randn(3,1))*ones(1,P);
weights   = ones(P,1);
xEst         = zeros(3,Steps+1);
xEst(:,1)  =  po; 
%xEst(:,1)  = po + 2*randn(3,1) ; %po
xPath       = zeros(Gap*3,P);
ESS          = zeros(nA,1); % effective sample size
R              = 0;
xPathAll = zeros(Steps*3,P);
for ll=1:nA
    fprintf('Assimilation %g/%g \n',ll,nA)
    for kk=1:P
        fprintf('     Particle %g/%g :  ',kk,P)
        % Minimization
        xo = Xn(:,kk);
        b   = xObs(:,ll);
        [mu phi] = SteepestDescent(xo,dt,Gap,Gap,g,s,b,maxiter,tol);
        % Sample around minimum
        mu = reshape(mu(:,2:end),numel(mu(:,2:end)),1);
        % Random map
        xi    = randn(3*Gap,1);
        rho = norm(xi)^2;
        eta  = 1/norm(xi)*xi;
        % first guess (here lambda = sqrt(xi^T*xi))
        x     =  mu+norm(xi)*eta;
        x     = reshape(x,3,Gap);
        [Mx fprime]  = compMx([xo x],Gap,dt);  
        [lambda x f0]= GetLambda([xo x],Mx,fprime,mu,phi,eta,rho,dt,Gap,g,s,b);
        x = reshape(x,3,Gap+1);
        
        % Weights
        LogLambda = (3*Gap-1)*log(lambda);
        LogRho         = (1-3*Gap/2)*log(rho);
        gradF = grad(x,Mx,fprime,Gap,Gap,g,s,b,dt);
        gradF = reshape(gradF,1,numel(gradF));
        dldrho =rho/(2*gradF*eta);
        weights(kk) = log(weights(kk))+LogLambda+LogRho+log(abs(dldrho))-0.5*phi - 0.5*f0;
        % Save endpoint for next assimilation cycle
        Xn(:,kk) = x(:,end);
        % Save particle path
        xPath(:,kk) = reshape(x(:,2:end),numel(x(:,2:end)),1);
        % Save all particle paths (for illustration)
         xPathAll((ll-1)*Gap*3+1:ll*Gap*3,kk) = xPath(:,kk);
    end
    weights = weights - max(weights);
    weights = exp(weights);
    weights = weights/sum(weights);

    % Compute path
    xEst(1,(ll-1)*Gap+2:ll*Gap+1) = (xPath(1:3:end,:)*weights)';
    xEst(2,(ll-1)*Gap+2:ll*Gap+1) = (xPath(2:3:end,:)*weights)';
    xEst(3,(ll-1)*Gap+2:ll*Gap+1) = (xPath(3:3:end,:)*weights)';
    % Resample
    ESS(kk)=1/(weights'*weights);
    if 1/(weights'*weights)<0.9*P
        Xn = resampling(weights,Xn,P);
        weights = ones(P,1);
        R = R+1;
        fprintf('Effective Sample Size %g, resampled. \n',ESS(kk));
    end   
end

% Plot all particle paths
%xPathAll = [po*ones(1,P);xPathAll];
%subplot(311)
%hold on, plot(t,xPathAll(1:3:end,:),'HandleVisibility','off')
%subplot(312)
%hold on, plot(t,xPathAll(2:3:end,:),'HandleVisibility','off')
%subplot(313)
%hold on, plot(t,xPathAll(3:3:end,:),'HandleVisibility','off')

%% Plot
figure
subplot(311)
hold on,plot(t(1,1:end  ) ,xExp(1,1:end),'k','LineWidth',1)
%hold on,plot(tObs,xObs(1,:),'o')
hold on, plot(t(1,Gap+1:Gap:end),xEst(1,Gap+1:Gap:end),'bx')
hold on, plot(t(1,Gap+1:Gap:end),xEst1(1,Gap+1:Gap:end),'ro')
ylabel('x')
title('DHIPF and IPF for Lorenz63')
legend('Lorenz 63 model simulated path','IPF estimation','DHIPF estimation')
subplot(312)
hold on,plot(t(1,1:end  ),xExp(2,1:end),'k','LineWidth',1)
%hold on,plot(tObs,xObs(2,:),'o')
hold on, plot(t(1,Gap+1:Gap:end),xEst(2,Gap+1:Gap:end),'bx')
hold on, plot(t(1,Gap+1:Gap:end),xEst1(2,Gap+1:Gap:end),'ro')
ylabel('y')
legend('Lorenz 63 model simulated path','IPF estimation','DHIPF estimation')
subplot(313)
hold on,plot(t(1,1:end  ),xExp(3,1:end),'k','LineWidth',1)
%hold on,plot(tObs,xObs(3,:),'o')
hold on, plot(t(1,Gap+1:Gap:end),xEst(3,Gap+1:Gap:end),'bx')
hold on, plot(t(1,Gap+1:Gap:end),xEst1(3,Gap+1:Gap:end),'ro')
xlabel('Time'),ylabel('z')
legend('Lorenz 63 model simulated path','IPF estimation','DHIPF estimation')

% ........................................................................
figure
subplot(311)
hold on,plot(t(1,1:150  ) ,xExp(1,1:150),'k','LineWidth',1)
%hold on,plot(tObs,xObs(1,:),'o')
hold on, plot(t(1,Gap+1:Gap:150),xEst(1,Gap+1:Gap:150),'bx')
hold on, plot(t(1,Gap+1:Gap:150),xEst1(1,Gap+1:Gap:150),'ro')
ylabel('x')
title('DHIPF and IPF for Lorenz63')
legend('Lorenz 63 model simulated path','IPF estimation','DHIPF estimation')
subplot(312)
hold on,plot(t(1,1:150  ),xExp(2,1:150),'k','LineWidth',1)
%hold on,plot(tObs,xObs(2,:),'o')
hold on, plot(t(1,Gap+1:Gap:150),xEst(2,Gap+1:Gap:150),'bx')
hold on, plot(t(1,Gap+1:Gap:150),xEst1(2,Gap+1:Gap:150),'ro')
ylabel('y')
legend('Lorenz 63 model simulated path','IPF estimation','DHIPF estimation')
subplot(313)
hold on,plot(t(1,1:150  ),xExp(3,1:150),'k','LineWidth',1)
%hold on,plot(tObs,xObs(3,:),'o')
hold on, plot(t(1,Gap+1:Gap:150),xEst(3,Gap+1:Gap:150),'bx')
hold on, plot(t(1,Gap+1:Gap:150),xEst1(3,Gap+1:Gap:150),'ro')
xlabel('Time'),ylabel('z')
legend('Lorenz 63 model simulated path','IPF estimation','DHIPF estimation')


%%%%%
figure
subplot(311)
hold on,plot(t(1,151:end  ),xExp(1,151:end),'k','LineWidth',1)
%hold on,plot(tObs,xObs(1,:),'o')
hold on, plot(t(1,151:Gap:end),xEst(1,151:Gap:end),'bx')
hold on, plot(t(1,151:Gap:end),xEst1(1,151:Gap:end),'ro')
ylabel('x')
title('DHIPF and IPF for Lorenz63')
legend('Lorenz 63 model simulated path','IPF estimation','DHIPF estimation')
subplot(312)
hold on,plot(t(1,151:end  ),xExp(2,151:end),'k','LineWidth',1)
%hold on,plot(tObs,xObs(2,:),'o')
hold on, plot(t(1,151:Gap:end),xEst(2,151:Gap:end),'bx')
hold on, plot(t(1,151:Gap:end),xEst1(2,151:Gap:end),'ro')
ylabel('y')
legend('Lorenz 63 model simulated path','IPF estimation','DHIPF estimation')
subplot(313)
hold on,plot(t(1,151:end  ),xExp(3,151:end),'k','LineWidth',1)
%hold on,plot(tObs,xObs(3,:),'o')
hold on, plot(t(1,151:Gap:end),xEst(3,151:Gap:end),'bx')
hold on, plot(t(1,151:Gap:end),xEst1(3,151:Gap:end),'ro')
xlabel('Time'),ylabel('z')
legend('Lorenz 63 model simulated path','IPF estimation','DHIPF estimation')

% Show some results
fprintf('Mean effective sample size: %g \n',mean(ESS)*P)
fprintf('Resampled %g / %g \n',R,nA)
% ........................................................................

%Plot 3d graph
figure
plot3(xExp(1,:),xExp(2,:),xExp(3,:) )
hold on; plot3(xEst(1,:), xEst(2,:),xEst(3,:) )
hold on, plot3(xEst1(1,:), xEst1(2,:),xEst1(3,:) )
view(30.179999999999993,-42.385964929660716);
legend('Lorenz63 Simulation Path','IPF estimation path','DHIPF estimation path')
title('DHIPF for Lorenz63 model 3d view')
% save ForMovieIS.mat t xExp xObs tObx xEst xPathAll po P Steps Gap
%Show Convergence: error of three vs time step
Total_Err2 = abs(xEst(:,Gap+1:Gap:end) - xExp(:,Gap+1:Gap:end) );
Total_Err2 = sum(Total_Err2);
figure
plot( t(1,Gap+1:Gap:end) ,Total_Err2,'r')
hold on, plot( t(1,Gap+1:Gap:end) ,Total_Err1,'b')
title('Absolute Error of DHIPF vs IPF with same computational power')
legend({'IPF absolute error','DHIPF absolute error'},'Location','northeast')
%[psnr,mse,maxerr,L2rat] = measerr(xEst(:,Gap+1:Gap:end),xExp(:,Gap+1:Gap:end) );
%mse