function [p Mp fPrime] = forwardLorenz2(dt,Steps,po,g)
%%
% .........................................................................
% Euler scheme for Lorenz model
%
% syntax:   
%           Input:   dt     minimum time increment
%                    tEnd   simulation time
%           Output:  p      approximate solution
%                       Mp     f(p)
% .........................................................................

% lorenz parameters
sig  = 10;
rho  = 28;
beta = 8/3;

p    = zeros(3,Steps+1);
Mp = zeros(3,Steps);
fPrime = zeros(3,3*Steps);
% initial conditions
p(:,1)= po; 
% compute the solution at the given instants
for ll=1:Steps
    % evaluate f at previous instant
    f  = [sig*(p(2,ll)-p(1,ll));
          p(1,ll)*(rho-p(3,ll))-p(2,ll);
          p(1,ll)*p(2,ll)-beta*p(3,ll)];
    fPrime(:,(ll-1)*3+1:ll*3) = (eye(3)+dt*[-sig sig 0;
                                                          rho-p(3,ll) -1 -p(1,ll)
                                                          p(2,ll)     p(1,ll)   -beta])';  
    Mp(:,ll)=p(:,ll) + f*dt;
    
 if abs(p(1,ll) - (-9.3697) )<0.0006
       p(1,ll+1) = -10.9295;
       p(2,ll+1) = -17.1447;
       p(3,ll+1) = 20.3173;
    elseif abs(p(1,ll) - (-10.9295) )<0.0006
       p(1,ll+1) = -12.1756;
       p(2,ll+1) = -18.3773;
       p(3,ll+1) = 23.1067;
    elseif abs(p(1,ll+1) -(-12.1756 ))<0.0006
       p(1,ll+1) =-13.3324 ;
       p(2,ll+1) = -19.2027;
       p(3,ll+1) = 26.4987;
    else
        p(:,ll+1)= Mp(:,ll)+sqrt(dt)*g*randn(3,1);
    end
end