function [accuracy,C,I] = KNN_main(X_training,Y_training, X_test, Y_test,class_name)    
    l = size(X_test,1);
    hit = zeros(5,2);
    count = 0;
    K = 3;
    for j=1:100 
        for i=1:l
                class = knn(X_test(i,:),X_training,Y_training,K);
                if class == Y_test(i)
                    count = count+1;
                end 
        end
        hit(j,1) = (count/l)*100;
        hit(j,2) = K;
        K = K+2;
        count = 0;
    end
    
    figure
    plot(hit(:,2),hit(:,1));
    title(class_name);
    xlabel('K');
    ylabel('accuracy');
    
    result = mean(hit(:,1));
    accuracy = result(1);
    [C,I] = max(hit(:,1));

        
end

