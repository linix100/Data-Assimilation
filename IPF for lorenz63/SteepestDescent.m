function [x J H]= SteepestDescent(xo,dt,Steps,Gap,g,s,xObs,maxiter,tol)
kk         = 0;
weiter   = 1; 
JOld      = 1e16;
JSave   = zeros(maxiter,1);

%% Minimize with gradient descent
[x Mx fprime] =  EulerLorenz(dt,Steps,xo,g);
while kk<=maxiter && weiter ==1;
    kk = kk+1;  
    % Full gradient (no linearization)  
    gr = grad(x,Mx,fprime,Steps,Gap,g,s,xObs,dt);    
    % line search for step size
    a = 1e-4:1e-4:5e-3;
    Js = zeros(length(a),1);
    for ll=1:length(a) 
        tmp1 = [x(:,1) x(:,2:end)-a(ll)*gr];
        tmp2 = compMx(tmp1,Steps,dt);
        Js(ll) = CostFunction(tmp1,tmp2,xObs,s,g,Gap,Steps,dt);
    end
    clear tmp1 tmp2
    [tmp1 tmp2]=min(Js);
    a= a(tmp2);clear tmp1 tmp2
    x(:,2:end) = x(:,2:end)-a*gr;
    [Mx fprime]= compMx(x,Steps,dt);
    J = CostFunction(x,Mx,xObs,s,g,Gap,Steps,dt);
    if kk>1
        if abs(JOld-J)/J < tol;
            weiter = 0;
            fprintf('Minimization done after %g steps \n',kk)
        elseif J-JOld>0
            weiter = 0;
            fprintf('Max. number of iterations reached! \n',kk)
        end
    end
    JOld = J;
end

%% Compute approximate Hessian at minimum
H = zeros(3*Steps);
for ll = 1:Steps-1    
        fxo = Mx(:,ll);            
        x1  = x(:,ll+1);          
        fx1 = Mx(:,ll+1);        
        x2  = x(:,ll+2);   
        fPrime = fprime(:,(ll-1)*3+1:ll*3);
        H((ll-1)*3+1:ll*3,(ll-1)*3+1:ll*3) = (dt*g^2)\(fPrime'*fPrime+eye(3));
        H((ll-1)*3+1:ll*3,(ll-1)*3+1+3:ll*3+3) = -(dt*g^2)\fPrime';
        if mod(ll,Gap)==0
            H((ll-1)*3+1:ll*3,(ll-1)*3+1:ll*3) = H((ll-1)*3+1:ll*3,(ll-1)*3+1:ll*3) +s\eye(3);
        end
end

H(end-2:end,end-2:end) = (dt*g^2)\eye(3);
if mod(Steps,Gap)==0
    H(end-2:end,end-2:end) = H(end-2:end,end-2:end)+eye(3)/s;
end
H = triu(H)+triu(H)'-diag(diag(H));

end
