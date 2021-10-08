function  [delta, alpha] = WJSDM_solve(delta, delta_old, S1, S2, lambda1, W, DW, iter, alpha, varargin)

argin = inputParser;
argin.addRequired('delta', @(x)  iscell(x));
argin.addRequired('delta_old', @(x)  iscell(x));
argin.addRequired('S1', @(x)  iscell(x));
argin.addRequired('S2', @(x)  iscell(x));
argin.addRequired('lambda1', @(x) isnumeric(x) && x>=0);
argin.addRequired('W', @(x) isnumeric(x));
argin.addRequired('iter', @(x)  isnumeric(x));
argin.addRequired('alpha', @(x) iscell(x));

% argin.addParameter('alpha', 1, @(x) isnumeric(x) && x>0);
% argin.addParameter('MAX_ITER', 5e2, @(x) isnumeric(x) && x>0);
% argin.addParameter('TOL', 1e-5, @(x) isnumeric(x) && x>0);
argin.parse(delta, delta_old, S1, S2, lambda1, W, iter, alpha, varargin{:});

%% 
% alpha = argin.Results.alpha;
% MAX_ITER = argin.Results.MAX_ITER;
% TOL = argin.Results.TOL;

%% 
p = size(S1,1);
% Delta = zeros(p);
% Delta_old = Delta;
% obj_fun = 0;

%% 
% for iter = 1:MAX_ITER
    phi = Phi_WM(delta,DW);
    
     K = length(delta);
     Weight = cell(K,1);
    for k = 1 : K
        Weight{k} = W.*DW{k}.*phi;
     Y{k} = delta{k} + (iter/(iter+3))*(delta{k} - delta_old{k});
%     alpha{k} = 1;
   
    
    while 1
        [f_Y{k}, grad_Y{k}] = WJSDM_grad(S1{k}, S2{k}, Y{k});

        Z{k} = soft_thresh_WM( Y{k} - alpha{k}*grad_Y{k}, alpha{k}*lambda1, Weight{k}, true);
        [f_Z{k}, ~] = WJSDM_grad(S1{k}, S2{k}, Z{k});
        if f_Z{k} <= (f_Y{k} + sum( diag(grad_Y{k}*(Z{k} - Y{k})) ) + (norm(Z{k} - Y{k},'fro'))^2 /2 /alpha{k})
            break
        end
        alpha{k} = alpha{k} * 0.5;
    end
    delta{k} = (Z{k} + Z{k}')/2;
    end
    
    


% Delta = Delta-diag(diag(Delta));
