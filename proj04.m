%% Initialization
clear ; close all; clc

%% Loading database
fprintf('Loading database...\n');
load('input.mat');
fprintf('Database loaded with success!\n');

%% Preparing Y for 3 rounds of Logistic Regression
[Y_training_1, Y_training_2, Y_training_3] = logistic_data_binarization(Y_training);
[Y_test_1, Y_test_2, Y_test_3] = logistic_data_binarization(Y_test);

%% KNN
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
fprintf('Press any key to continue...\n');
pause;

%% Logistic Regression
fprintf('\nRunning Logistic Regression...\n');
%accuracy_logistic = 0;
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

%% Logistic Regression with Regularization
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

%% Naive Bayes
fprintf('\nRunning Naive Bayes...\n');
[ training_accuracy, test_accuracy, learning_curve ] = naive_bayes(X_training,Y_training, X_test, Y_test);
fprintf('Naive Bayes accuracy with training data = %.2f percent!\n', training_accuracy);
fprintf('Naive Bayes accuracy with test data = %.2f percent!\n', test_accuracy);
plot_learning_curve(learning_curve, 'Naive Bayes');
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
