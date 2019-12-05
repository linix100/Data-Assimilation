function [Mp fPrime] = compMx(p,Steps,dt)
%%
% lorenz parameters
sig  = 10;
rho  = 28;
beta = 8/3;

Mp = zeros(3,Steps);
fPrime = zeros(3,3*Steps);
for ll=1:Steps
    % evaluate f at previous instant
    f  = [sig*(p(2,ll)-p(1,ll));
          p(1,ll)*(rho-p(3,ll))-p(2,ll);
          p(1,ll)*p(2,ll)-beta*p(3,ll)];
    Mp(:,ll)=p(:,ll) + f*dt;
    fPrime(:,(ll-1)*3+1:ll*3) = (eye(3)+dt*[-sig sig 0;
                                                          rho-p(3,ll) -1 -p(1,ll)
                                                          p(2,ll)     p(1,ll)   -beta])';  
end