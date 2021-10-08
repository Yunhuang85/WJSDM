function [delta] = main_WJSDM(X, W, F, lambda, wg, wf, flag)
K=size(X,1);
p=size(X{1,1},2);
obj_old = 0;
delta_sum = 0;
% initialize:
weight = ones(p,p);
n = zeros(K,2);
for k = 1:K
    delta{k}=ones(p);
    alpha{k} = 1;
    for c = 1:2
        n(k,c) =  size(X{k,c},1);
        if  flag == 1
             temp = KS(X{k,c},1);
             if sum(eig(temp)<0)>0
                S{k,c} = nearestSPD(temp);
             else
                S{k,c} = temp;
             end
        end
    end
     H = hyp_test(X(k,:),'rtest');
     Dweight{k} = (1-(1-H)*(1-H)');
end
weight(F == 0) = wf;
weight(W == 1) = wg;
delta_new = delta;

times=200;
tol=1e-5;
for k = 1 : K
    [f_Delta{k}, ~] = WJSDM_grad(S{k,1}, S{k,2}, delta{k});
    obj_old = obj_old + f_Delta{k};
    delta_sum = delta_sum + Dweight{k}.*abs(delta{k});
end
obj_old =obj_old + lambda*sum(sum(weight.*sqrt(delta_sum)));

for iter=1:times
%     value=0;
%     for i=1:K
%         W=w;
   delta_old = delta;
   delta = delta_new;
   obj = 0;
   delta_sum = 0;
        [delta_new, alpha] = WJSDM_solve(delta, delta_old, S(:,1), S(:,2), lambda, weight, Dweight, iter, alpha);
        for k = 1 : K
            [f_Delta{k}, ~] = WJSDM_grad(S{k,1}, S{k,2}, delta_new{k});
            obj = obj + f_Delta{k};
            delta_sum = delta_sum + Dweight{k}.*abs(delta_new{k});
        end
        obj = obj + lambda*sum(sum(weight.*sqrt(delta_sum)));
        
    if abs(obj_old - obj) < tol*(abs(obj))
            break;
    end
    obj_old = obj; 
end
