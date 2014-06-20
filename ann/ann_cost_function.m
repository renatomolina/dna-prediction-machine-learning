function [J, grad]= ann_cost_function(H, Y, theta,delta, L, lambda, m)
    %m = size(H, 1);
    J = 0;
    %grad = zeros(size(theta));
    %grad = cell(1,L-1);
    %grad = 0;
    
    %% Calculando J
    a = -Y(1:m, :) * log(H(1:m, :)');
    b = (1 - Y(1:m, :)) * log(1 - H(1:m, :))';
    c = lambda/(2*m) * sum(sum(sum(theta .^ 2)));
    J = (1/m * sum(sum(abs(a - b)))) + c;
    
    
    %% Calculando Gradiente ANN - página 9 slide[1,1]
    grad = zeros(size(delta));
    grad = 1/m * delta + (lambda * theta);
    grad(:, 1, :) = 1/m * delta(:, 1, :);
    %for l=1:L-1
    %    grad{l} = (1/m) .* delta{l} + (lambda * theta{l});
    %end
    %% Gradiente do Logístico
    %for i = 1:size(theta)
    %    grad(i) = 1/m * (sum((logistic_sigmoid(H) - y) .* X(:,i)));
    %end
end
