function [pAtrEI, pAtrIE, pAtrN] = naive_compute_probability(X, Y)
  pAtrEI = zeros(64,size(X,2));
  pAtrIE = zeros(64,size(X,2));
  pAtrN = zeros(64,size(X,2));

  countEI = 0;
  countIE = 0;
  countN = 0;
  
  for i=1:size(Y)
      if Y(i) == 1
          countEI = countEI + 1;
          for j = 1:size(X,2)
            pAtrEI(X(i,j),j) = pAtrEI(X(i,j),j) + 1;
          end
      elseif Y(i) == 2
          countIE = countIE + 1;
          for j = 1:size(X,2)
            pAtrIE(X(i,j),j) = pAtrIE(X(i,j),j) + 1;
          end
      elseif Y(i) == 3
          countN = countN + 1;
          for j = 1:size(X,2)
            pAtrN(X(i,j),j) = pAtrN(X(i,j),j) + 1;
          end
      end
  end

  pAtrEI = pAtrEI ./ countEI;
  pAtrIE = pAtrIE ./ countIE;
  pAtrN = pAtrN ./ countN;

end