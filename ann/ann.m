function [ training_accuracy, test_accuracy, learning_curve ] = ann( X_training, Y_training, X_test, Y_test )
    learning_curve = zeros(10,2);
    
    %% Putting ones in position '0'
    [m1, n1] = size(X_training);
    X_training = [ones(m1, 1) X_training];
    [m2, n2] = size(X_test);
    X_test = [ones(m2,1) X_test];
    
    %% Debug
    m1 = 10;
    m2 = 10;
    %% Training Parameters
    % Camadas = 3
    L = 3;
    % Neurônios por camada
    s = n1 + 1;
    % Quantidade de Classes
    K = s;
    
    %% Training Preparations
    % Inicializar Delta com zeros
    delta = zeros(s,s, L-1);
    % Inicializar pesos com valores aleatórios [-1, 1]
    theta = -1 + (1-(-1)).*rand(s,s, L-1);
    % Vetor h - resultado de cada amostra
    h = zeros(m1,K);
    % classificação de cada Amostra
    Y_training = y_vetorization(Y_training, s);
    Y_test = y_vetorization(Y_test, s);
   
    %% Training
    for i=1:m1
        fprintf('Begin - Training sample %d!\n', i);
        x = X_training(i, :);
        y = Y_training(i, :);
        % Forward Propagation
        a = ann_forward(x, theta, L, s);
        h(i, :) = a(:,:,L)';
        % Backpropagation
        sigma = ann_backpropagation(a, y, theta, L, s);
        % Acumular Derivadas parciais
        for l=(1:L-1)
            delta(:,:,l) = delta(:,:,l) + (sigma(:,:,l+1) * a(:,:,l)');
        end
        % Otimizando Theta
        lambda = 1;
        [J, gradient] = ann_cost_function(h, Y_training, theta, delta, L, lambda, i);
        options = optimset('GradObj', 'on', 'MaxIter', 400, 'Display', 'off');
        [theta, J] = fminunc(@(t)(ann_cost_function(h, Y_training, theta, delta, L, lambda, i)), theta, options);
        fprintf('End - Training sample %d!\n', i);
    end
    
    %% Test with Training Set
    result = zeros(K,m1);
    for i=1:m1
        a = ann_forward(X_training(i, :), theta, L, s);
        result(:, i) = a(:,:,L);
        %binarização de result?
    end
    %learning_curve(i,1) = sum(result~=Y_training);
    training_accuracy = sum(sum(result==Y_training(1:m1, :)'))/m1*100;
        
    %% Test with Test Set
    result = zeros(K,m2);
    for i=1:m2
        a = ann_forward(X_test(i, :), theta, L, s);
        result(:, i) = a(:,:,L);
        %binarização de result?
    end
    %learning_curve(i,2) = sum(result~=Y_test);
    test_accuracy = sum(sum(result==Y_test(1:m2, :)'))/m2*100;

end
