function [ training_accuracy, test_accuracy, learning_curve ] = naive_bayes( X_training, Y_training, X_test, Y_test )
    % Running the Training and the Test 10 times
    % To generate a Learning Curve
    learning_curve = zeros(10,2);
    
    for i =1:10
        start= 1;
        finish = i*265;
        X_local = X_training(start:finish,:);
        Y_local = Y_training(start:finish,:);
        
        %% ================= Part 1: Training ====================
        pEI = sum(Y_local==1)/size(Y_training,1); 
        pIE = sum(Y_local==2)/size(Y_local,1);
        pN = sum(Y_local==3)/size(Y_local,1);

        [pAtrEI, pAtrIE, pAtrN] = naive_compute_probability(X_local,Y_local);

        %% ================= Part 2: Test with Training =================
        result = zeros(size(X_local,1),1);

        for j = 1:size(X_local, 1)
            result(j,1) = naive_classification(X_local(j,:)',pEI,pIE,pN,pAtrEI,pAtrIE, pAtrN);
        end
        
        learning_curve(i,1) = sum(result~=Y_local);
        training_accuracy = sum(result==Y_local)/size(Y_local,1)*100;
        
        %% ================= Part 3: Test with Test =================
        result = zeros(size(X_test,1),1);

        for j = 1:size(X_test, 1)
            result(j,1) = naive_classification(X_test(j,:)',pEI,pIE,pN,pAtrEI,pAtrIE, pAtrN);
        end
        
        learning_curve(i,2) = sum(result~=Y_test);
        test_accuracy = sum(result==Y_test)/size(Y_test,1)*100;
    end

end
