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

        %% Training Parameters
        % Camadas = 3
        L = 3;
        s = zeros(L, 1);
        % Neurônios de entrada (s1) = 61
        s(1) = 61;
        % Neurônios na camada intermediária
        for i=(2:L-1)
            s(i) = 61;
        end
        % Neurônios de saída (sn) = 1
        s(L) = 1;
        K = s(L);
        
        %% Training
        [m,n] = size(X_local_training);
        % Inicializar Delta com zeros
        delta = cell(1,L-1);
        for i=(1:L-1)
            delta{i} = zeros(s(i+1),1);
        end
        % Inicializar pesos com valores aleatórios (theta = rand(20 + 1, L)) próximos de zero
        theta = cell(L-1, 1);
        for i=(1:size(s)-1)
            %theta{i} = rand(s(i+1),n);
            theta{i} = -1 + (1-(-1)).*rand(s(i+1),n);
        end
        
        h = zeros(m,K);
        % Para cada Amostra
        for i=1:m
            x = X_local_training(i,:);
            y = Y_local_training(i);
            % Forward Propagation
            a = ann_forward(x, theta, L);
            h(i) = a{L};
            % Backpropagation
            sigma = ann_backpropagation(a, y, theta, L);
            % Acumular Derivadas parciais
            for l=(1:L-1)
                delta{l} = delta{l} + a{l}' * sigma{l+1};
            end
            lambda = 1;
            [J, gradient] = ann_cost_function(h, Y_local_training, theta, L, lambda);
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
