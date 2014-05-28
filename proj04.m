%% Initialization
clear ; close all; clc

%% Loading database
fprintf('Loading database...\n');
load('input.mat');
fprintf('Database loaded with success!\n');

%% KNN
fprintf('\nRunning KNN...\n');
accuracy_knn = 0;
fprintf('KNN accuracy = %.2f percent!\n', accuracy_knn);
fprintf('Press any key to continue...\n');
pause;

%% Linear Regression
fprintf('\nRunning Linear Regression...\n');
accuracy_linear = 0;
fprintf('Linear Regression accuracy = %.2f percent!\n', accuracy_linear);
fprintf('Press any key to continue...\n');
pause;

%% Logistic Regression
fprintf('\nRunning Logistic Regression...\n');
accuracy_logistic = 0;
fprintf('Logistic Regression accuracy = %.2f percent!\n', accuracy_logistic);
fprintf('Press any key to continue...\n');
pause;

%% Naive Bayes
fprintf('\nRunning Naive Bayes...\n');
accuracy_naive = naive_bayes(X,Y);
fprintf('Naive Bayes accuracy = %.2f percent!\n', accuracy_naive);
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
