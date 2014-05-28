function [ accuracy ] = naive_bayes( X, Y )
    %% ================= Part 1 ====================
    pEI = sum(Y==1)/size(Y,1); 
    pIE = sum(Y==2)/size(Y,1);
    pN = sum(Y==3)/size(Y,1);
    
    [pAtrEI, pAtrIE, pAtrN] = naive_compute_probability(X,Y);

    %% ================= Part 2 =================
    result = zeros(size(X,1),1);

    for i = 1:size(X, 1)
        result(i,1) = naive_classification(X(i,:)',pEI,pIE,pN,pAtrEI,pAtrIE, pAtrN);
    end
    accuracy = sum(result==Y)/size(Y,1)*100;
   
end
