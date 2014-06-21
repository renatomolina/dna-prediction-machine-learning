function sigma = ann_backpropagation( a, y,theta, L, s)
    sigma = zeros(s, 1, L);
    sigma(:, :, L) = a(:, :, L) - y';

    for l=L-1:-1:2
        sigma(:, :, l) = (theta(:, :, l)' * sigma(:, :, l+1)) .* a(:, :, l)  .* (1.-a(:, :, l));
    end    
    
end
