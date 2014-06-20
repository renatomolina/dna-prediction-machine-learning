function g = ann_sigmoid( z )
%ANN_SIGMOID Summary of this function goes here
%   Detailed explanation goes here
    g = zeros(size(z));
    g = 1./(1 + exp(-z));

end

