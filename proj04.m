%% Initialization
clear ; close all; clc

%% Loading database
fprintf('Loading database...\n');
load('input.mat');
fprintf('Database loaded with success!\n');

%% KNN
fprintf('\nRunning KNN...\n');
accuracy_knn = 0;
%accuracy_knn = KNN_main(X_training,Y_training, X_test, Y_test);
fprintf('KNN accuracy = %.2f percent!\n', accuracy_knn);
fprintf('Press any key to continue...\n');
pause;

%% Logistic Regression
fprintf('\nRunning Logistic Regression...\n');
accuracy_logistic = 0;
%accuracy_logistic1 = logistic_regression(X,Y1);
%accuracy_logistic2 = logistic_regression(X,Y2);
%accuracy_logistic3 = logistic_regression(X,Y3);
%accuracy_logistic = accuracy_logistic1 + accuracy_logistic2 + accuracy_logistic3;
fprintf('Logistic Regression accuracy = %.2f percent!\n', accuracy_logistic);
fprintf('Press any key to continue...\n');
pause;

%% Naive Bayes
fprintf('\nRunning Naive Bayes...\n');
[ training_accuracy, test_accuracy, learning_curve ] = naive_bayes(X_training,Y_training, X_test, Y_test);
fprintf('Naive Bayes accuracy with training data = %.2f percent!\n', training_accuracy);
fprintf('Naive Bayes accuracy with test data = %.2f percent!\n', test_accuracy);
plot(learning_curve);
fprintf('Press any key to continue...\n');
pause;

%% Neural Network
fprintf('\nRunning Neural Network...\n');
accuracy_neural = 0;
fprintf('Neural Network accuracy = %.2f percent!\n', accuracy_neural);
fprintf('Press any key to continue...\n');
pause;

%% SVM
fprintf('\nRunning SVM...\n');
accuracy_svm = 0;
fprintf('SVM accuracy = %.2f percent!\n', accuracy_svm);
fprintf('Press any key to finish execution...\n');
pause;

%% Finishing
