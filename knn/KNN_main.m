function [accuracy,C,I] = KNN_main(X_training,Y_training, X_test, Y_test)    
    l = size(X_test,1);
    hit = zeros(5,1);
    count = 0;
    K = 3;
    for j=1:5  
        for i=1:l
                class = knn(X_test(i,:),X_training,Y_training,K);
                if class == Y_test(i)
                    count = count+1;
                end 
        end
        hit(j) = (count/l)*100;
        K = K+2;
        count = 0;
    end
    result = mean(hit);
    accuracy = result(1);
    [C,I] = max(hit);

        
end

