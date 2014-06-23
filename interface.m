function varargout = interface(varargin)

    %% Loading Paths
    if ispc
        addpath .\common
        addpath .\knn
        addpath .\logistic_regression
        addpath .\naive_bayes
        addpath .\ann
        addpath .\svm
    else
        addpath ./common
        addpath ./knn
        addpath ./logistic_regression
        addpath ./naive_bayes
        addpath ./ann
        addpath ./svm
    end
    
% INTERFACE MATLAB code for interface.fig
%      INTERFACE, by itself, creates a new INTERFACE or raises the existing
%      singleton*.
%
%      H = INTERFACE returns the handle to a new INTERFACE or the handle to
%      the existing singleton*.
%
%      INTERFACE('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in INTERFACE.M with the given input arguments.
%
%      INTERFACE('Property','Value',...) creates a new INTERFACE or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before interface_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to interface_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help interface

% Last Modified by GUIDE v2.5 22-Jun-2014 21:27:34

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @interface_OpeningFcn, ...
                   'gui_OutputFcn',  @interface_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before interface is made visible.
function interface_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to interface (see VARARGIN)

% Choose default command line output for interface
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes interface wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = interface_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;
    

% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
    clc;
    clear;
    load('data_nucleotides_codification1.mat');
    set(findobj('Tag','console'),'String','Please wait...');
    drawnow
    fprintf('\nRunning Naive Bayes...\n');
    [ training_accuracy, test_accuracy, learning_curve ] = naive_bayes(X_training,Y_training, X_test, Y_test);
    
    message = strcat('Naive Bayes accuracy with training data = ', num2str(training_accuracy), '%');
    message2 = strcat('Naive Bayes accuracy with test data = ', num2str(test_accuracy), '%');
    set(findobj('Tag','console'),'String', {message, message2});
    
    drawnow
    figure
    plot(learning_curve);
    title('Naive Bayes');
    xlabel('Interactions');
    ylabel('Errors');

% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    clc;
    clear;
    load('data_nucleotides_codification1.mat');
    set(findobj('Tag','console'),'String','Please wait...');
    drawnow
    
    fprintf('\nRunning KNN...\n');
    %% Running the KNN for the entire set of training and test data
    [accuracy_knn1,C,I, result2, result3] = KNN_main(X_training,Y_training, X_test, Y_test, 'All');
    figure
    plot(result2, result3);
    title('KNN');
    xlabel('K');
    ylabel('accuracy');
    message = strcat('Greatest accuracy was  ', num2str(C), ' for K = ', num2str(((I*2)+1)));
    set(findobj('Tag','console'),'String',message);
    drawnow

% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    clc;
    clear;
    load('data_nucleotides_codification1.mat');
    %% Preparing Y for binary classification
    [Y_training_1, Y_training_2, Y_training_3] = y_binarization(Y_training);
    [Y_test_1, Y_test_2, Y_test_3] = y_binarization(Y_test);
    set(findobj('Tag','console'),'String','Please wait...');
    drawnow
    
    [training_accuracy_logistic1, test_accuracy_logistic1, learning_curve ]= logistic_regression(X_training,Y_training_1, X_test, Y_test_1);
    figure
    plot(learning_curve);
    title('Logistic Regression - Class EI');
    xlabel('Interactions');
    ylabel('Errors');
    
    [training_accuracy_logistic2, test_accuracy_logistic2, learning_curve ]= logistic_regression(X_training,Y_training_2, X_test, Y_test_2);    
    figure
    plot(learning_curve);
    title('Logistic Regression - Class IE');
    xlabel('Interactions');
    ylabel('Errors');
    
    [training_accuracy_logistic3, test_accuracy_logistic3, learning_curve ]= logistic_regression(X_training,Y_training_3, X_test, Y_test_3);
    figure
    plot(learning_curve);
    title('Logistic Regression - Class N');
    xlabel('Interactions');
    ylabel('Errors');
    
    training_accuracy_logistic = (training_accuracy_logistic1 + training_accuracy_logistic2 + training_accuracy_logistic3)/3;
    test_accuracy_logistic = (test_accuracy_logistic1 + test_accuracy_logistic2 + test_accuracy_logistic3)/3;
    message = strcat('Logistic Regression accuracy with training data = ', num2str(training_accuracy_logistic), '%');
    message2 = strcat('Logistic Regression accuracy with test data = ', num2str(test_accuracy_logistic), '%');
    set(findobj('Tag','console'),'String', {message, message2});
    drawnow
    
% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    clc;
    clear;
    load('data_nucleotides_codification1.mat');
    %% Preparing Y for binary classification
    [Y_training_1, Y_training_2, Y_training_3] = y_binarization(Y_training);
    [Y_test_1, Y_test_2, Y_test_3] = y_binarization(Y_test);
    set(findobj('Tag','console'),'String','Please wait...');
    drawnow
    
    [training_accuracy_logistic1, test_accuracy_logistic1, learning_curve ]= logistic_regression_reg(X_training,Y_training_1, X_test, Y_test_1);
    figure
    plot(learning_curve);
    title('Logistic Regression - Class EI');
    xlabel('Interactions');
    ylabel('Errors');
    
    [training_accuracy_logistic2, test_accuracy_logistic2, learning_curve ]= logistic_regression_reg(X_training,Y_training_2, X_test, Y_test_2);    
    figure
    plot(learning_curve);
    title('Logistic Regression - Class IE');
    xlabel('Interactions');
    ylabel('Errors');
    
    [training_accuracy_logistic3, test_accuracy_logistic3, learning_curve ]= logistic_regression_reg(X_training,Y_training_3, X_test, Y_test_3);
    figure
    plot(learning_curve);
    title('Logistic Regression - Class N');
    xlabel('Interactions');
    ylabel('Errors');
    
    training_accuracy_logistic = (training_accuracy_logistic1 + training_accuracy_logistic2 + training_accuracy_logistic3)/3;
    test_accuracy_logistic = (test_accuracy_logistic1 + test_accuracy_logistic2 + test_accuracy_logistic3)/3;
    message = strcat('Logistic Regression accuracy with training data = ', num2str(training_accuracy_logistic), '%');
    message2 = strcat('Logistic Regression accuracy with test data = ', num2str(test_accuracy_logistic), '%');
    set(findobj('Tag','console'),'String', {message, message2});
    drawnow


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    clear;
    clc;
    load('data_nucleotides_codification1.mat');
    set(findobj('Tag','console'),'String','Please wait...');
    drawnow
    indices = crossvalind('Kfold',size(X_training,1),10);
    v = (indices == 1); 
    t = (indices ~= 1);
    validar = X_training(v,:);
    treino = X_training(t,:);
    treino_y = Y_training(t);
    validar_y = Y_training(v);
    [ weights, out_weights, total_error ] = rede_neural( treino, treino_y, X_test, Y_test);
    %===plotar a taxa de erro===%
    figure;
    hold on;
    title('Tax of network error');
    xlabel('Epoch') % x-axis label
    ylabel('Error Tax') % y-axis label
    plot(total_error(:,2),total_error(:,1))
    %===Validação====%
    accuracy = validacao( weights, out_weights, validar, validar_y);
    set(findobj('Tag','console'),'String',strcat('Neural Network Accuracy:', num2str(accuracy), '%' ));
    drawnow


% --- Executes on button press in pushbutton7.
function pushbutton7_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton7 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
    clear;
    clc;
    load('data_nucleotides_codification1.mat');
    set(findobj('Tag','console'),'String','Please wait...');
    drawnow
    
    parameters = '-q';
    [accuracy] = svm(X, Y, parameters);
    fprintf('SVM accuracy = %.2f\n',accuracy);
    
    set(findobj('Tag','console'),'String',strcat('SVM Accuracy:', num2str(accuracy), '%' ));
    drawnow
    
