function [y,ind_viz] = knn(x, X, Y, K)

y = 0;                % Inicializa rotulo como classe negativa
ind_viz = ones(K,1);  % Inicializa indices (linhas) em X das K amostras mais 
                      % proximas de x.

D = knn_hamming( x, X );
[value,ind] = sort(D);
ind_viz = ind(1:K);
y = mode(Y(ind_viz));

end