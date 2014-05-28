function [ accuracy ] = knn( X, Y )
    result = zeros(size(X,1),1);
    K = 5;
    for i = 1:size(X, 1)
        result(i,1) = knn_classification(X(i,:)', X, Y , K);
    end
    accuracy = sum(result==Y)/size(Y,1)*100;
end
