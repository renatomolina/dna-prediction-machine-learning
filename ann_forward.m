function h = ann_forward( X, theta, L, sl )
    A =  zeros(sl, L);
    Z = zeros(sl);
    A(:,1) = X;
    for i=(2:L-1)
        Z = theta(i,:) * A(:,i-1);
        A(:,i) = ann_sigmoid(Z);
    end
    h = ann_sigmoid(Z);
end
