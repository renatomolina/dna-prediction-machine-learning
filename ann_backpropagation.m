function Sigma = ann_backpropagation( A, Y,theta, L )
%ANN_BACKPROPAGATION Summary of this function goes here
%   Detailed explanation goes here
    Sigma = zeros(Size(A));
    Sigma(:,L) = A(:,L) - Y;
    for l=L-1:2
        Sigma(:,l) = (theta(:,l) * Sigma(:,l+1)) .* (A(:,l) .* (1 - A(:,l)));
    end

end

