function distance = knn_hamming( x, X )
    distance = zeros(size(X,1),1);
    
    for i = 1:size(X,1)
        hamming = 0;
        %hamming = sum(x == X(i,:));
        for j = 1:size(X,2)
            if ~(x(j) == X(i,j))
                hamming = hamming + 1;
            end
        end
        distance(i) = hamming;
    end

end
