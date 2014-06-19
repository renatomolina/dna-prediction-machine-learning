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
        [m,n] = size(X_local_training);
        
        %% Training Parameters
        % Camadas = 3
        L = 3;
        % Neurônios por camada
        s = n;
        % Quantidade de Classes
        K = s;
        
        %% Training
        % Inicializar Delta com zeros
        delta = zeros(s,1, L-1);
        % Inicializar pesos com valores aleatórios [-1, 1]
        theta = -1 + (1-(-1)).*rand(s,s, L-1);
        % Vetor h - resultado de cada amostra
        h = zeros(m,K);
        % Para cada Amostra
        for i=1:m
            x = X_local_training(i,:);
            y = y_vetorization(Y_local_training(i), s);
            % Forward Propagation
            a = ann_forward(x, theta, L, s);
            h(i, :) = a(:,:,L);
            % Backpropagation
            sigma = ann_backpropagation(a, y, theta, L, s);
            % Acumular Derivadas parciais
            for l=(1:L-1)
                delta(:, :, l) = delta(:, :, l) + a(:, :, l)' * sigma(:, :, l+1);
            end
            lambda = 1;
            [J, gradient] = ann_cost_function(h, y, theta, L, lambda);
            options = optimset('GradObj', 'on', 'MaxIter', 400);
            [theta, J] = fminunc(@(t)(ann_cost_function(h, Y_local_training, theta, L, lambda)), theta, options);
        end

        %% Test with Training Set
        result = zeros(size(X_local_training,1),1);
        learning_curve(i,1) = sum(result~=Y_local_training);
        training_accuracy = sum(result==Y_local_training)/size(Y_local_training,1)*100;

        %% Test with Test Set
        result = zeros(size(X_test,1),1);
        learning_curve(i,2) = sum(result~=Y_test);
        test_accuracy = sum(result==Y_test)/size(Y_test,1)*100;
    end
end
