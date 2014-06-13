%% Initialization
clear ; close all; clc

%% Loading database
fprintf('Loading database...\n');
load('input.mat');
fprintf('Database loaded with success!\n');

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

%% Finishing
