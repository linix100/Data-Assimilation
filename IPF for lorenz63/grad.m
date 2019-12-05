function DF = grad(x,fx,fprime,Steps,Gap,g,s,b,dt)
DF = zeros(3,Steps);

for ll = 1:Steps-1    
        fxo = fx(:,ll);            
        x1  = x(:,ll+1);          
        fx1 = fx(:,ll+1);        
        x2  = x(:,ll+2);   
        fPrime = fprime(:,(ll-1)*3+1:ll*3);
        DF(:,ll) = (dt*g^2)\((x1-fxo)+ fPrime*(fx1-x2));
        if mod(ll,Gap)==0
            DF(:,ll)=DF(:,ll)+s\(x1-b(:,ll/Gap));
        end
end
DF(:,end) = (dt*g^2)\(x(:,end)-fx(:,end));
if mod(Steps,Gap)==0
    DF(:,end) =DF(:,end) +s\(x(:,end)-b(:,end));
end
end

