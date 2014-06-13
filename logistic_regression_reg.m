function [ training_accuracy, test_accuracy, learning_curve ] = logistic_regression_reg( X_training, Y_training, X_test, Y_test )
%LOGISTIC_REGRESSION_REG Summary of this function goes here
%   Detailed explanation goes here
    learning_curve = zeros(10,2);
    X_test = [ones(size(X_test,1),1) X_test];
    
    for i=1:10
        start= 1;
        finish = i*265;
        [m, n] = size(X_training(start:finish,:));
        X_local_training = [ones(m, 1) X_training(start:finish,:)];
        Y_local = Y_training(start:finish,:);
        
        %% Training
        initial_theta = zeros(size(X_local_training, 2), 1);
        lambda = 1;
        [cost, gradient] = logistic_cost_function_reg(initial_theta, X_local_training, Y_local, lambda);
        options = optimset('GradObj', 'on', 'MaxIter', 400);
        [theta, J, exit_flag] = fminunc(@(t)(logistic_cost_function_reg(t, X_local_training, Y_local, lambda)), initial_theta, options);
        
        %% Test with Training
        result = logistic_prediction(theta, X_local_training);        
        learning_curve(i,1) = sum(result~=Y_local);
        training_accuracy = mean(double(result == Y_local)) * 100;
        
        %% Test with Test
        result = logistic_prediction(theta, X_test);        
        learning_curve(i,2) = sum(result~=Y_test);
        test_accuracy = mean(double(result == Y_test)) * 100;
        
    end
    
end
