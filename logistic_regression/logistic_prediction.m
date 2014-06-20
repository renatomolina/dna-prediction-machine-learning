function p = logistic_prediction( theta, X )
    m = size(X, 1);

    p = zeros(m, 1);

    H = X * theta;
    for i = 1:m
        if logistic_sigmoid(H(i)) >= 0.5
            p(i) = 1;
        else
            p(i) = 0;
        end
    end

end

