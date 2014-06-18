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
        s(3) = 1;
        K = s(3);
        
        %% Training
        [m,n] = size(X_local_training);
        % Inicializar Delta com zeros
        %delta = zeros(s, s, L);
        % Inicializar pesos com valores aleatórios (theta = rand(20 + 1, L)) próximos de zero
        theta = cell(L-1, 1);
        for i=(1:size(s)-1)
            theta{i} = rand(s(i+1),n);
            %theta{i} = -1 + (1-(-1)).*rand(s(i),n);
        end        
        
        H = zeros(m,K);
        % Para cada Amostra
        for i=1:m
            x = X_local_training(i,:);
            y = Y_local_training(i);
            % Forward
            a = ann_forward(x, theta, L);
            %H(i,1) = a(K,L);
            % Backpropagation
            sigma = ann_backpropagation(a, y, theta, L);
            % Acumular Derivadas parciais
        end
        % Calcular a derivada da função custo
        % função custo J
        %lambda = 1;
        %cost = ann_cost_function(theta, lambda, X_local_training, Y_local);
        %options = optimset('GradObj', 'on', 'MaxIter', 400);
        %[theta, cost] = fminunc(@(t)(ann_cost_function(t, X_local_training, Y_local)), initial_theta, options);

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
