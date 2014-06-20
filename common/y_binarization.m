function [Y1, Y2, Y3] = y_binarization(Y)
    Y1 = (Y == 1);
    Y2 = (Y == 2);
    Y3 = (Y == 3);
end
