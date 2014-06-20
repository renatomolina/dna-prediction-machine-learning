function [ Y1, Y2 Y3 ] = logistic_data_binarization( Y )
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
end

