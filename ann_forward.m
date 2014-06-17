function a = ann_forward(X, theta, L)
    a = cell(1,L);
    a{1} = X';
    for l=(2:L)
        thetaL = theta{l-1};
        aL = a{l-1};
        z = thetaL * aL;
        a{l} = ann_sigmoid(z);
    end
%     A =  zeros(sl, L);
%     A(:,1) = X;
%     for l=(2:L)
%         for i=(1:sl)
%             Z = theta(i,:, l-1) * A(:,l-1);
%             A(i,l) = ann_sigmoid(Z);
%         end
%     end
end

