function [ accuracy ] = KNN_main( X, Y)

    K = 7;   
    s = size(X,1);

    indices = crossvalind('Kfold',s,10);
    hit = zeros(10,1);
    for i = 1:10
        test = (indices == i); train = ~test;
        sample = X(test,:);
        classes = Y(test);
        count = 0;
        l = size(sample,1);
        for j=1:l
            class = knn(sample(j,:),X(train,:),Y,K);
            if class == classes(j)
                count = count+1;
            end 
        end
        hit(i) = (count/l)*100;
    end
        result = mean(hit);
        accuracy = result(1);
end

