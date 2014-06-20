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
        delta = zeros(s,s, L-1);
        % Inicializar pesos com valores aleatórios [-1, 1]
        theta = -1 + (1-(-1)).*rand(s,s, L-1);
        % Vetor h - resultado de cada amostra
        h = zeros(m,K);
        % classificação de cada Amostra
        y = y_vetorization(Y_local_training, s);
        % Para cada Amostra
        for i=1:m
            x = X_local_training(i,:);            
            % Forward Propagation
            a = ann_forward(x, theta, L, s);
            h(i, :) = a(:,:,L);
            % Backpropagation
            sigma = ann_backpropagation(a, y(i), theta, L, s);
            % Acumular Derivadas parciais
            for l=(1:L-1)
                delta(:,:,l) = delta(:,:,l) + (sigma(:,:,l+1) * a(:,:,l)');
            end
            %delta = delta + (sigma * a)
            %delta = delta + (a(:,1))
            %delta(:, :, l) = delta(:, :, l) + a(:, :, l)' * sigma(:, :, l+1);
            %for l=(1:L-1)
            %    for j=1:s
            %        for ii=1:s
            %            delta(ii,j,l) = delta(ii,j,l) + (a(:,j,l) * sigma(ii,:, l+1));
            %        end
            %    end                
            %end
            lambda = 1;
            [J, gradient] = ann_cost_function(h, y, theta, delta, L, lambda, i);
            options = optimset('GradObj', 'on', 'MaxIter', 400, 'Display', 'off');
            [theta, J] = fminunc(@(t)(ann_cost_function(h, y, theta, delta, L, lambda, i)), theta, options);
        end

        %% Test with Training Set
        result = zeros(m,1);
        for i=1:m
            a = ann_forward(X_local_training(i, :), theta, L, s);
            result(i, :) = a(:,:,L);
            learning_curve(i,1) = sum(result~=y);
            training_accuracy = sum(result==y)/size(y,1)*100;
        end
        
        %% Test with Test Set
        [m, n] = size(X_test);
        y_test = y_vetorization(Y_test, s);
        result = zeros(m,1);
        for i=1:m
            a = ann_forward(X_test(i, :), theta, L, s);
            result(i, :) = a(:,:,L);
            learning_curve(i,2) = sum(result~=y_test);
            test_accuracy = sum(result==Y_test)/size(y_test,1)*100;
        end
                
    end
end
