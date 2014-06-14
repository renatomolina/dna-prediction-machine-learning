function [ training_accuracy, test_accuracy, learning_curve ] = ann( X_training, Y_training, X_test, Y_test )
%ANN Summary of this function goes here
%   Detailed explanation goes here
    learning_curve = zeros(10,2);

    [m, n] = size(X_local);
    X_training = [ones(m, 1) X_training];

    for i=1:10
        start= 1;
        finish = i*265;
        X_local_training = X_training(start:finish,:);
        Y_local_training = Y_training(start:finish,:);

        %% Training
        % Neur�nios de entrada (s1) = 21 +1 (vi�s)
        % Neur�nios de sa�da (sn) = 3
        % Camadas intermedi�rias (L) = 1
        % Neur�nios da camada intermedi�ria = 20 +1 (vi�s)

        % Inicializar pesos com valores aleat�rios (theta = rand(20 + 1, L)) pr�ximos de zero
        L = 1;
        theta = rand(21, L)/100;
        lambda = 1;

        % fun��o custo J
        cost = ann_cost_function(initial_theta, lambda, X_local_training, Y_local);
        options = optimset('GradObj', 'on', 'MaxIter', 400);
        [theta, cost] = fminunc(@(t)(ann_cost_function(t, X_local_training, Y_local)), initial_theta, options);

        % Chamar func forward_propagation

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
