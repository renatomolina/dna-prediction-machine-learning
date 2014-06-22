function [ accuracy ] = rede_neural( X_training, Y_training, X_test, Y_test)
    

    epoch = 1000; %Número de épocas (treinamentos)
    [m,n] = size(X_training);
    weights  = -1 + (1-(-1)).*rand(n,n); %Matriz de pesos da entrada para camada intermediaria
    out_weights = -1 + (1-(-1)).*rand(3,n); %Matriz de pesos de inter-saida
    classes = [1 0 0;0 1 0; 0 0 1]; %Possíveis saidas corretas
    bias=ones(n,1);
    out_bias = ones(3,1);
    
    Layer = ones(n,1);  %Camada intermediaria
    wlayer = ones(n,1);
    
    Saida = ones(3,1); %Camada de saida
    wSaida = ones(3,1);
   
    difference = zeros(n,n);
    difference_out = zeros(3,n);
    error_out = zeros(3,1);
    
    %======Taxa de Aprendizagem e Momentum======%
    learn_rate=0.05;
    momentum = 0.9;
    for u=1:epoch
        
        fprintf('\nTreinando a epoch %d\n',u);
         [treino,idx] = datasample(X_training,size(X_training,1)); %Embaralha o treinamento pra não viciar a rede
         Y_treino = Y_training(idx);
        
        for k=1:m
            input = treino(k,:); %Camada de entrada
            %==========Forward Propagation==============%
            %Input to Layer
                for i=1:n
                    wlayer(i) = sum(weights(i,:).*input)+bias(i);
                    Layer(i) = 1/(1+exp(-wlayer(i)));
                   
                end    
            %Layer to output using Sigmoid
            for i=1:3
                wSaida(i) = sum(out_weights(i,:)'.*Layer)+out_bias(i);
                Saida(i) = 1/(1+exp(-wSaida(i)));
            end
           
           
            for i=1:3
                error_out(i) = Saida(i)*( 1-Saida(i))*(classes(Y_treino(k),i) -Saida(i));
                difference_out(i,:) = learn_rate* error_out(i)* Layer;
                difference_out(i,:) = difference_out(i,:)+(difference_out(i,:)*momentum);
            end
            
            for i=1:n
                difference(i,:) = learn_rate* (  Layer(i)*(1-Layer(i))* sum(out_weights(:,i).*error_out) ) *input;
                difference(i,:) = difference(i,:) + (difference(i,:)*momentum);
            end
            out_weights = out_weights+difference_out;
            weights = weights+difference;

                

           
        end
        %=========Inicio do teste da Epoch=========% 
        [x,y] = size(X_test);
        hit = 0;
        for k=1:x
            input = X_test(k,:);
            for i=1:n
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
        fprintf('\nNumber of hits for Epoch %d: %d/%d',u,hit,x);
        fprintf('\nAccuracy : %g',(hit/x)*100);
    end
   accuracy = ((hit/x)*100);
   
end