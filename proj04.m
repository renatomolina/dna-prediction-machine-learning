%% Initialization
clear ; close all; clc

%% Loading database
fprintf('Loading database...\n');
load('input.mat');
fprintf('Database loaded with success!\n');

%% Preparing Y for 3 rounds for Linear e Logistic Regression
Y1 = Y;
Y2 = Y;
Y3 = Y;
for i = 1:size(Y)
    if Y1(i) > 1
        Y1(i) = 0;
    end
    if Y2(i) < 2 || Y2(i) > 2 
        Y2(i) = 0;
    else
        Y2(i) = 1;
    end
    if Y3(i) < 2
        Y3(i) = 0;
    else
        Y3(i) = 1;
    end
end

%% KNN
fprintf('\nRunning KNN...\n');
accuracy_knn = 0;
fprintf('KNN accuracy = %.2f percent!\n', accuracy_knn);
fprintf('Press any key to continue...\n');
pause;

%% Linear Regression
fprintf('\nRunning Linear Regression...\n');
%accuracy_linear1 = linear_regression(X,Y1);
%accuracy_linear2 = linear_regression(X,Y2);
%accuracy_linear3 = linear_regression(X,Y3);
%accuracy_linear = accuracy_linear1 + accuracy_linear2 + accuracy_linear3;
accuracy_linear = 0;
fprintf('Linear Regression accuracy = %.2f percent!\n', accuracy_linear);
fprintf('Press any key to continue...\n');
pause;

%% Logistic Regression
fprintf('\nRunning Logistic Regression...\n');
%accuracy_logistic = 0;
accuracy_logistic1 = logistic_regression(X,Y1);
accuracy_logistic2 = logistic_regression(X,Y2);
accuracy_logistic3 = logistic_regression(X,Y3);
accuracy_logistic = accuracy_logistic1 + accuracy_logistic2 + accuracy_logistic3;
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
