function dfdl = CompDerivLambda(x,fx,fprime,eta,b,s,g,Gap,Steps,dt)
dfdl =(dt*g^2)\(eta(:,1)'*(x(:,2)-fx(:,1))); 
for ll = 2:Steps-1                
        fx1 = fx(:,ll+1);        
        x2  = x(:,ll+2);     
        eta1 = eta(:,ll);
        eta2 = eta(:,ll+1);
        fPrime = fprime(:,(ll-1)*3+1:ll*3);
        dfdl = dfdl +  (dt*g^2)\((eta2-fPrime'*eta1)'*(x2-fx1));
        if mod(ll,Gap)==0
            dfdl = dfdl + s\(x2-b(:,ll/Gap));
        end
end
dfdl = dfdl +  (dt*g^2)\( (eta(:,end)-fprime(:,end-2:end)'*eta(:,end-1))'*(x(:,end)-fx(:,end)) );
if mod(Steps,Gap)==0
    dfdl = dfdl + s\(eta(:,end)'*(x(:,end)-b(:,end)));
end
end