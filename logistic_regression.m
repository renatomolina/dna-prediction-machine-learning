function [ training_accuracy, test_accuracy, learning_curve ] = logistic_regression( X_training, Y_training, X_test, Y_test )
    % Running the Training and the Test 10 times
    % To generate a Learning Curve
    learning_curve = zeros(10,2);
    X_test = [ones(size(X_test,1),1) X_test];

    %% Running 10 times
    for i=1:10
        start= 1;
        finish = i*265;
        [m, n] = size(X_training(start:finish,:));
        X_local_training = [ones(m, 1) X_training(start:finish,:)];
        Y_local = Y_training(start:finish,:);
        
        %% Training
        initial_theta = zeros(n + 1, 1);

        [cost, gradient] = logistic_cost_function(initial_theta, X_local_training, Y_local);

        options = optimset('GradObj', 'on', 'MaxIter', 400);

        [theta, cost] = fminunc(@(t)(logistic_cost_function(t, X_local_training, Y_local)), initial_theta, options);
        
        %% Test with Training
        result = logistic_prediction(theta, X_local_training);
        
        learning_curve(i,1) = sum(result~=Y_local);
        training_accuracy = mean(double(result == Y_local)) * 100;
        
        %% Test with Test
        result = logistic_prediction(theta, X_test);
        
        learning_curve(i,2) = sum(result~=Y_test);
        test_accuracy = mean(double(result == Y_test)) * 100;
    end

    %% =========== Parte 7: Regressao Logistica com Regularizacao ============
%     
%     X = atributosPolinomiais(X(:,1), X(:,2));
% 
%     initial_theta = zeros(size(X, 2), 1);
% 
%     lambda = 1;
% 
%     [cost, gradient] = funcaoCustoReg(initial_theta, X, y, lambda);
% 
%     options = optimset('GradObj', 'on', 'MaxIter', 400);
% 
%     [theta, J, exit_flag] = fminunc(@(t)(funcaoCustoReg(t, X, y, lambda)), initial_theta, options);
%     
%     p = predicao(theta, X);
%     accuracy2 = mean(double(p == Y)) * 100;


end

