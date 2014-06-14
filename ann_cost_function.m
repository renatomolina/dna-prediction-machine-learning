function J = ann_cost_function( theta, lambda, X, Y )

    % qtd de classes
    k = 3;

    m = length(Y);
    Yk = zeros(m, k);
    [Yk(:,1), Yk(:,2), Yk(:,3)] = logistic_data_binarization(Y);

    H = X * theta;

    parcial_j = zeros(k);
    for i = 1:k
        a = (-Y(:,i) .* log(logistic_sigmoid(H)));
        b = ((1 - Y(:,i)) .* log(1 - logistic_sigmoid(H)));
        parcial_j(i) = sum(abs(a - b));
    end

    c = 0;
    % c = (lambda/(2*m)) * theta^2;
    % a somatória de regularização não roda com uma camada intermediária
    J = sum(parcial_j)/m + c;
end

