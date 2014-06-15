
function [ maxACC ] = SVM(X, Y, parameters)

    s = size(X,1);
    indices = crossvalind('Kfold',s,10);
    acc = zeros(10,1);
    for i = 1:10
        test = (indices == i); 
        train = ~test;
        sample = X(test,:);
        classes = Y(test);
        model = svmtrain(Y(train), sparse(X(train,:)), parameters);
        [predict_label, accuracy_svm, decimal] = svmpredict(classes, sparse(sample), model, '-q');
        acc(i) = accuracy_svm(1); 
    end
        result = mean(acc);
        maxACC = result(1);
end