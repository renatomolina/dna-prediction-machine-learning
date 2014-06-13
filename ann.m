function [ training_accuracy, test_accuracy, learning_curve ] = ann( X_training, Y_training, X_test, Y_test )
%ANN Summary of this function goes here
%   Detailed explanation goes here
    learning_curve = zeros(10,2);
    for i=1:10
        start= 1;
        finish = i*265;
        X_local = X_training(start:finish,:);
        Y_local = Y_training(start:finish,:);

        %% Training
        % Neurônios de entrada (s1) = 20
        % Neurônios de saída (sn) = 3
        % Camadas intermediárias (L) = 1
        % Neurônios da camada intermediária = 20

        % Inicializar pesos com valores aleatórios (theta) próximos de zero

        % Chamar func forward_propagation

        % função custo J
        % K = quantidade de classes?

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
