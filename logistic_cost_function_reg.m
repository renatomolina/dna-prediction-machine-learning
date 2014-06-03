function [ J, grad ] = logistic_cost_function_reg( theta, X, y, lambda )
%LOGISTIC_COST_FUNCTION_REG Summary of this function goes here
%   Detailed explanation goes here
    m = length(y);
    J = 0;
    grad = zeros(size(theta));

    H = X * theta;

    a = (-y .* log(logistic_sigmoid(H)));
    b = ((1 - y) .* log(1 - logistic_sigmoid(H)));
    c = lambda/(2*m) * (sum(theta.^2)-1);
    J = (1/m * sum(abs(a - b))) + c;

    grad(1) =  1/m * (sum((logistic_sigmoid(H) - y) .* X(:,1)));
    for i = 2:size(theta)
        grad(i) =  1/m * (sum((logistic_sigmoid(H) - y) .* X(:,i))) + lambda/m * theta(i);
    end

end

