function [ accuracy ] = validacao( weights, out_weights, X, Y)
    

    [X_test,idx] = datasample(X,100); %Embaralha o treinamento pra não viciar a rede
    Y_test = Y(idx);
    [m,n]=size(X_test);
    Layer = ones(n,1);  %Camada intermediaria
    wlayer = ones(n,1);
    Saida = ones(3,1); %Camada de saida
    wSaida = ones(3,1);
    classes = [1 0 0;0 1 0; 0 0 1]; %Possíveis saidas corretas
    bias=ones(n,1);
    out_bias = ones(3,1);
    [x,y] = size(X_test);
    hit = 0;
        for k=1:x
            input = X_test(k,:);
            for i=1:y
               wlayer(i) = sum(weights(i,:).*input)+bias(i);
               Layer(i) = 1/(1+exp(-wlayer(i)));
            end
           
            %Layer to output using Sigmoid
            for i=1:3
                wSaida(i) = sum(out_weights(i,:)'.*Layer)+out_bias(i);
                Saida(i) = 1/(1+exp(-wSaida(i)));
            end
             [~,j] = max(Saida);
            if(classes(j,:) == classes((Y_test(k)),:))
                hit = hit+1;
            end
        end
        fprintf('\nNumber of hits for validation: %d/%d',hit,x);
        accuracy = (hit/x)*100;
        
end