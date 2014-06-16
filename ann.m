function [ training_accuracy, test_accuracy, learning_curve ] = ann( X_training, Y_training, X_test, Y_test )
%ANN Summary of this function goes here
%   Detailed explanation goes here
    learning_curve = zeros(10,2);

    [m, n] = size(X_training);
    X_training = [ones(m, 1) X_training];

    for i=1:10
        start= 1;
        finish = i*265;
        X_local_training = X_training(start:finish,:);
        Y_local_training = Y_training(start:finish,:);

        %% Training
        % Neurônios de entrada (s1) = 61
        % Neurônios de saída (sn) = 3
        % Camadas = 3
        L = 3;
        % Neurônios da camada intermediária = 61
        sl =61;
        % Inicializar pesos com valores aleatórios (theta = rand(20 + 1, L)) próximos de zero
        %initial_theta = rand(L, sl);
        initial_theta = -1 + (1-(-1)).*rand(L,sl);
        % Chamar func forward_propagation
        [m,n] = size(X_local_training);
        H = zeros(m);
        for i=1:m
            H(i) = ann_forward(X_local_training(i,:), initial_theta, L, sl);
        end
        
        % função custo J
        %lambda = 1;
        %cost = ann_cost_function(initial_theta, lambda, X_local_training, Y_local);
        %options = optimset('GradObj', 'on', 'MaxIter', 400);
        %[theta, cost] = fminunc(@(t)(ann_cost_function(t, X_local_training, Y_local)), initial_theta, options);
        
        % Chamar func back_propagation

        %% Test with Training Set
        result = zeros(size(X_local,1),1);
        learning_curve(i,1) = sum(result~=Y_local);
        training_accuracy = sum(result==Y_local)/size(Y_local,1)*100;

        %% Test with Test Set
        result = zeros(size(X_test,1),1);
        learning_curve(i,2) = sum(result~=Y_test);
        test_accuracy = sum(result==Y_test)/size(Y_test,1)*100;
    end
end
