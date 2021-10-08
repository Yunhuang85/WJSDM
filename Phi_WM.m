function [ phi ] = Phi_WM( delta ,w)
K = length(delta);
phi=0;
for k = 1: K
%     phi = phi + abs(delta{k});
    phi = phi + w{k}.*abs(delta{k});
end
phi = 1./(2*sqrt(phi)+eps);
end

