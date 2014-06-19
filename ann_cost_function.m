function [J, grad]= ann_cost_function(H, Y, theta, L, lambda)
    m = size(H, 1);
    J = 0;
    %grad = zeros(size(theta));
    %grad = cell(1,L-1);
    grad = 0;
    
    %% Calculando J
    a = log(H) * -Y ;
    b = log(1 - H) * (1 - Y);
    c = lambda/(2*m) * sum(sum(theta .^ 2));
    J = (1/m * sum(sum(abs(a - b)))) + c;
    
    
    %% Calculando Gradiente ANN - página 9 slide[1,1]
    %for l=1:L-1
    %    grad{l} = (1/m) .* delta{l} + (lambda * theta{l});
    %end
    %% Gradiente do Logístico
    %for i = 1:size(theta)
    %    grad(i) = 1/m * (sum((logistic_sigmoid(H) - y) .* X(:,i)));
    %end
end
