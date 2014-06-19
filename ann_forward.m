function a = ann_forward(X, theta, L, s)
    a = zeros(s, 1, L);
    a(:,:,1) = X';

    for l=(2:L)
        z = theta(:,:,l-1) * a(:,:,l-1);
        a(:, :, l) = ann_sigmoid(z);
    end
    
end
