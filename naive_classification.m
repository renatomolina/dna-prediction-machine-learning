function [classe, probEI, probIE, probN] = naive_classification(x,pEI,pIE,pN,pAtrEI,pAtrIE, pAtrN)
    classe = 0;
    probEI= pEI;
    probIE = pIE;
    probN = pN;

    for i = 1:size(x)
        probEI = probEI * pAtrEI(x(i),i);
        probIE = probIE * pAtrIE(x(i),i);
        probN = probN * pAtrN(x(i),i);    
    end

    if (probEI > probIE) && (probEI > probN)
        classe = 1;
    elseif (probIE > probEI) && (probIE > probN)
        classe = 2;
    elseif (probN > probEI) && (probN > probIE)
        classe = 3;
    end

end