function Yv = y_vetorization(Y, s)
    Yv = zeros(size(Y,1),s);
    for i=1:size(Y,1)
    Yv(i,Y(i)) = 1;
    end

end

