function [lambda x f0]= GetLambda(x,Mx,fprime,mu,phi,eta,rho,dt,Gap,g,s,b)


weiter = 1;
kk       = 0;
maxiter= 1000;
tol = 1e-6;

lambda = sqrt(rho);

xo = x(:,1);
f0Old = CostFunction(x,Mx,b,s,g,Gap,Gap,dt)-phi-rho;
ETA =reshape(eta,3,Gap);
while kk<=maxiter && weiter ==1;    
    % Compute derivative 
    dldrho = CompDerivLambda(x,Mx,fprime,ETA,b,s,g,Gap,Gap,dt);
    % Newton Step
    lambdaNew = lambda - 0.1*(dldrho\f0Old);
    clear tmp1 tmp2
    % Get new state
    x = mu + lambdaNew*eta;
    x     = reshape(x,3,Gap);
    x     = [xo x];
    [Mx fprime]  = compMx(x,Gap,dt);
    % Evaluate cost-function
    f0 = CostFunction(x,Mx,b,s,g,Gap,Gap,dt)-phi-rho;

    if norm(lambda-lambdaNew) < tol
        weiter = 0;
         fprintf('     Lambda after %g steps. Residual: %g \n',kk,abs(f0))
    elseif abs(f0) <1e-3
        weiter = 0;
         fprintf('     Lambda after %g steps (Tol. reached). Residual: %g \n',kk,abs(f0))    
    elseif kk == maxiter-1
        weiter = 0;
         fprintf('     Lambda after %g steps (Max. It. reached). Residual: %g \n',kk,abs(f0))    
    elseif f0>f0Old
        weiter = 0;
        fprintf('    Function value increases after %g steps. Residual: %g \n',kk,abs(f0)) 
    end
    kk= kk+1;
    lambda = lambdaNew;  
    f0Old = f0;
end