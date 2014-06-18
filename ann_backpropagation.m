function sigma = ann_backpropagation( a, y,theta, L)
    sigma = cell(1,L);
    sigma{L} = a{L} - y;
    for l=L-1:-1:2
        sigma{l} = (theta{l}' * sigma{l+1}) .* a{l}  .* (1.-a{l});
    end    
    
%     Sigma = zeros(size(a));
%     y = zeros(sl,1);
%     y(1,1) = Y;
%     Sigma(:,L) = a(:,L) - y;
%     for l=L-1:-1:2
%         for j=(1:sl)
%             a = theta(:,j,l) * Sigma(j,l+1);
%             b = a(j,l) * (1-a(j,l));
%             Sigma(j,l) = a * b;
%             %Sigma(j,l) = (theta(:,j,l) * Sigma(j,l+1)) * (a(j,l) * (1-a(j,l)));
%             %Sigma(:,l) = (theta(:,l) * Sigma(:,l+1)) .* (a(:,l) .* (1 - a(:,l)));
%         end
%     end

end

