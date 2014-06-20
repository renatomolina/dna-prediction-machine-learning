function plot_learning_curve( learning_curve, g_title )
%PLOT_LEARNING_CURVE Summary of this function goes here
%   Detailed explanation goes here
figure; hold on;
title(g_title);
plot(learning_curve);
hold off;
end

