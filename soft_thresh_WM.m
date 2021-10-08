function X = soft_thresh_WM(Y, lambda, Weight, penalize_diagonal)

if nargin == 1
    disp('Not enough inputs');
    disp('Enter both Y and lambda');
    return
end

X = sign(Y).*(max(abs(Y) - lambda.*Weight, 0));

if ~penalize_diagonal
    X = X - diag(diag(X)) + diag(diag(Y));
end
