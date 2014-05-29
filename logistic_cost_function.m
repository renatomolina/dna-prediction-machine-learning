function [J, grad] = logistic_cost_function( theta, X, y )

    m = length(y);

    grad = zeros(size(theta));

    H = X * theta;

    a = -y .* log(sigmoid(H));
    b = (1 - y) .* log(1 - sigmoid(H));
    J = 1/m * sum(abs(a - b));

    for i = 1:size(theta)
        grad(i) = 1/m * (sum((sigmoid(H) - y) .* X(:,i)));
    end

end

