function A = ann_forward( X, theta, L, sl )
    A =  zeros(sl, L);
    A(:,1) = X;
    for l=(2:L)
        for i=(1:sl)
            Z = theta(i,:, l-1) * A(:,l-1);
            A(i,l) = ann_sigmoid(Z);
        end
    end
end

