
function [ predict_label, accuracy_svm, decimal ] = svm(X_training, X_test, Y_training, Y_test, parameters)

model = svmtrain(Y_training, sparse(X_training), parameters);
[predict_label, accuracy_svm, decimal] = svmpredict(Y_test, sparse(X_test), model);

end