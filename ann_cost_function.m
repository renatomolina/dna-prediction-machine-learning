function J = ann_cost_function(H, Y, theta, L, lambda)
    m = length(Y);
    J = 0;
    a = -Y .* log(H);
    b = ((1 - Y) .* log(1 - H));
    for l=1:L-1
        c = lambda/(2*m) * sum(sum(theta{l}^2));
        J = J + (1/m * sum(sum(abs(a - b)))) + c;
    end
    
end
