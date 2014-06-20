%% Initialization
clear ; close all; clc

%% Debug
base =4;
run_knn = false;
run_logistic = false;
run_logistic_reg = false;
run_naive = false;
run_ann = false;
run_svm = false;

%% Loading Paths
if ispc
    addpath .\common
    addpath .\knn
    addpath .\logistic_regression
    addpath .\naive_bayes
    addpath .\ann
    addpath .\svm
else
    addpath ./common
    addpath ./knn
    addpath ./logistic_regression
    addpath ./naive_bayes
    addpath ./ann
    addpath ./svm
end

%% Loading database
fprintf('Loading database...\n');
switch base
    case 1
        load('data_codons_codification1.mat');
    case 2
        load('data_codons_codification2.mat');
    case 3
        load('data_codons_codification3.mat');
    case 4
        load('data_nucleotides_codification1.mat');
end
fprintf('Database loaded with success!\n');

%% Preparing Y for binary classification
[Y_training_1, Y_training_2, Y_training_3] = y_binarization(Y_training);
[Y_test_1, Y_test_2, Y_test_3] = y_binarization(Y_test);

%% KNN
if run_knn
    fprintf('\nRunning KNN...\n');
    %% Retrieving the data that corresponds to each class on the Database
    teste1 = X(Y==1,:);
    teste2 = X(Y==2,:);
    teste3 = X(Y==3,:);
    %% Running the KNN for each class and finding the respective accuracy.
    [accuracy_knn1,C,I] = KNN_main(X,Y, teste1, 1);
    fprintf('KNN mean accuracy for class EI = %.2f percent!\n', accuracy_knn1);
    fprintf('Greatest accuracy was %.2f for K=%d\n',C,((I*2)+1));
    [accuracy_knn2,C,I] = KNN_main(X,Y, teste2, 2);
    fprintf('KNN mean accuracy for class IE = %.2f percent!\n', accuracy_knn2);
    fprintf('Greatest accuracy was %.2f for K=%d\n',C,((I*2)+1));
    [accuracy_knn3,C,I] = KNN_main(X,Y, teste3, 3);
    fprintf('KNN mean accuracy for class N = %.2f percent!\n', accuracy_knn3);
    fprintf('Greatest accuracy was %.2f for K=%d\n',C,((I*2)+1));
    accuracy_knn = (accuracy_knn1+accuracy_knn2+accuracy_knn3)/3;
    fprintf('KNN overall accuracy = %.2f percent!\n', accuracy_knn);
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
end

%% Naive Bayes
if run_naive
    fprintf('\nRunning Naive Bayes...\n');
    [ training_accuracy, test_accuracy, learning_curve ] = naive_bayes(X_training,Y_training, X_test, Y_test);
    fprintf('Naive Bayes accuracy with training data = %.2f percent!\n', training_accuracy);
    fprintf('Naive Bayes accuracy with test data = %.2f percent!\n', test_accuracy);
    plot_learning_curve(learning_curve, 'Naive Bayes');
end

%% Artificial Neural Network
if run_ann
    fprintf('\nRunning Neural Network...\n');
    [ training_accuracy, test_accuracy, learning_curve ] = ann( X_training, Y_training, X_test, Y_test );
    fprintf('Neural Network accuracy at training = %.2f percent!\n', training_accuracy);
    fprintf('Neural Network accuracy at test = %.2f percent!\n', test_accuracy);
    plot_learning_curve(learning_curve, 'Artificial Neural Network');
end

%% SVM
if run_svm
    fprintf('\nRunning SVM...\n');
    parameters = '-t 1 -r 32 -g 4 -q';
    accuracy_svm = SVM(X, Y, parameters);
    fprintf('SVM accuracy = %.2f\n',accuracy_svm);
end

%% Finishing
fprintf('\nNo more Learning Machine methods to run!\n');
fprintf('Press any key to finish execution...\n');
pause;
