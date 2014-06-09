%% Initialization
clear ; close all; clc

%% Debug
run_knn = false;
run_logistic = false;
run_logistic_reg = false;
run_naive = true;
run_ann = false;
run_svm = false;

%% Loading database
fprintf('Loading database...\n');
load('input.mat');
fprintf('Database loaded with success!\n');



%% KNN
if run_knn
    fprintf('\nRunning KNN...\n');
    accuracy_knn = KNN_main_(X_training,Y_training, X_test, Y_test);
    fprintf('KNN accuracy = %.2f percent!\n', accuracy_knn);
    fprintf('Press any key to continue...\n');
    pause;
end

%% Logistic Regression
if run_logistic
    %% Preparing Y for 3 rounds of Logistic Regression
    [Y_training_1, Y_training_2, Y_training_3] = logistic_data_binarization(Y_training);
    [Y_test_1, Y_test_2, Y_test_3] = logistic_data_binarization(Y_test);
    fprintf('\nRunning Logistic Regression...\n');
    [training_accuracy_logistic1, test_accuracy_logistic1, learning_curve ]= logistic_regression(X_training,Y_training_1, X_test, Y_test_1);
    plot_learning_curve(learning_curve, 'Logistic Regression - Class EI');
    [training_accuracy_logistic2, test_accuracy_logistic2, learning_curve ]= logistic_regression(X_training,Y_training_2, X_test, Y_test_2);
    plot_learning_curve(learning_curve, 'Logistic Regression - Class IE');
    [training_accuracy_logistic3, test_accuracy_logistic3, learning_curve ]= logistic_regression(X_training,Y_training_3, X_test, Y_test_3);
    plot_learning_curve(learning_curve, 'Logistic Regression - Class N');
    training_accuracy_logistic = (training_accuracy_logistic1 + training_accuracy_logistic2 + training_accuracy_logistic3)/3;
    test_accuracy_logistic = (test_accuracy_logistic1 + test_accuracy_logistic2 + test_accuracy_logistic3)/3;
    fprintf('Logistic Regression accuracy with training data= %.2f percent!\n', training_accuracy_logistic);
    fprintf('Logistic Regression accuracy with test data= %.2f percent!\n', test_accuracy_logistic);
    fprintf('Press any key to continue...\n');
    pause;
end

%% Logistic Regression with Regularization
if run_logistic_reg
    %% Preparing Y for 3 rounds of Logistic Regression
    [Y_training_1, Y_training_2, Y_training_3] = logistic_data_binarization(Y_training);
    [Y_test_1, Y_test_2, Y_test_3] = logistic_data_binarization(Y_test);
    fprintf('\nRunning Logistic Regression...\n');
    %accuracy_logistic = 0;
    [training_accuracy_logistic1, test_accuracy_logistic1, learning_curve ]= logistic_regression_reg(X_training,Y_training_1, X_test, Y_test_1);
    plot_learning_curve(learning_curve, 'Logistic Regression with Regularization - Class EI');
    [training_accuracy_logistic2, test_accuracy_logistic2, learning_curve ]= logistic_regression_reg(X_training,Y_training_2, X_test, Y_test_2);
    plot_learning_curve(learning_curve, 'Logistic Regression with Regularization - Class IE');
    [training_accuracy_logistic3, test_accuracy_logistic3, learning_curve ]= logistic_regression_reg(X_training,Y_training_3, X_test, Y_test_3);
    plot_learning_curve(learning_curve, 'Logistic Regression with Regularization - Class N');
    training_accuracy_logistic = (training_accuracy_logistic1 + training_accuracy_logistic2 + training_accuracy_logistic3)/3;
    test_accuracy_logistic = (test_accuracy_logistic1 + test_accuracy_logistic2 + test_accuracy_logistic3)/3;
    fprintf('Logistic Regression with Regularization accuracy with training data= %.2f percent!\n', training_accuracy_logistic);
    fprintf('Logistic Regression with Regularization accuracy with test data= %.2f percent!\n', test_accuracy_logistic);
    fprintf('Press any key to continue...\n');
    pause;
end

%% Naive Bayes
if run_naive
    fprintf('\nRunning Naive Bayes...\n');
    [ training_accuracy, test_accuracy, learning_curve ] = naive_bayes(X_training,Y_training, X_test, Y_test);
    fprintf('Naive Bayes accuracy with training data = %.2f percent!\n', training_accuracy);
    fprintf('Naive Bayes accuracy with test data = %.2f percent!\n', test_accuracy);
    plot_learning_curve(learning_curve, 'Naive Bayes');
    fprintf('Press any key to continue...\n');
    pause;
end

%% Artificial Neural Network
if run_ann
    fprintf('\nRunning Neural Network...\n');
    accuracy_neural = 0;
    fprintf('Neural Network accuracy = %.2f percent!\n', accuracy_neural);
    fprintf('Press any key to continue...\n');
    pause;
end

%% SVM
if run_svm
    fprintf('\nRunning SVM...\n');
    accuracy_svm = 0;
    fprintf('SVM accuracy = %.2f percent!\n', accuracy_svm);
    fprintf('Press any key to continue...\n');
    pause;
end
%% Finishing
fprintf('No more Learning Machine methods to run!\n');
fprintf('Press any key to finish execution...\n');
pause;