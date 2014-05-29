function accuracy1 = logistic_regression( X, Y )
    %% ============ Parte 2: Calculo do Custo e do Gradiente ============
    [m, n] = size(X);
    X = [ones(m, 1) X];
    initial_theta = zeros(n + 1, 1);

    [cost, gradient] = logistic_cost_function(initial_theta, X, Y);

    options = optimset('GradObj', 'on', 'MaxIter', 400);
 
    [theta, cost] = fminunc(@(t)(logistic_cost_function(t, X, Y)), initial_theta, options);

    p = logistic_prediction(theta, X);
    accuracy1 = mean(double(p == Y)) * 100;

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

