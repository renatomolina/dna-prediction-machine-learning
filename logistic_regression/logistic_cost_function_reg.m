function [ J, grad ] = logistic_cost_function_reg( theta, X, y, lambda )
    m = length(y);
    grad = zeros(size(theta));

    H = X * theta;

    a = (-y .* log(sigmoid(H)));
    b = ((1 - y) .* log(1 - sigmoid(H)));
    c = lambda/(2*m) * (sum(theta.^2)-1);
    J = (1/m * sum(abs(a - b))) + c;

    grad(1) =  1/m * (sum((sigmoid(H) - y) .* X(:,1)));
    for i = 2:size(theta)
        grad(i) =  1/m * (sum((sigmoid(H) - y) .* X(:,i))) + lambda/m * theta(i);
    end
end

