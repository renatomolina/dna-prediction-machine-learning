function y = knn_classification( x, X, Y, K )
    y = 0;                
    neighborhood = ones(K,1);
    d = zeros(size(X,1));
    d = knn_hamming(x, X);
    [~, neighborhood] = sort(d, 1, 'ascend');
    y = mode(Y(neighborhood(1:K)));
end
