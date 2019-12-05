function J =  CostFunction(x,fx,b,s,g,Gap,Steps,dt)
J = 0;
for ll = 1:Steps-1    
        fxo = fx(:,ll);            
        x1  = x(:,ll+1);          
        fx1 = fx(:,ll+1);        
        x2  = x(:,ll+2);          
        J= J+ (dt*g^2)\(norm(x1-fxo)^2+ norm(fx1-x2)^2);
        if mod(ll,Gap)==0
            J=J+s\norm(x1-b(:,ll/Gap))^2;
        end
end
J= J + (dt*g^2)\norm(x(:,end)-fx(:,end))^2;
if mod(Steps,Gap)==0
    J= J + s\norm(x(:,end)-b(:,end))^2;
end
end