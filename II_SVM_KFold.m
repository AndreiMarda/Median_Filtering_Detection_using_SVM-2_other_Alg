clc;
clear;
close all;

folderPath = "C:\Facultate\Anul_IV\Licenta\Database\GBRASNET\BOSSbase-1.01\cover"; 
%% Initializare variabile 
numberOfImages = 200;
foldsNumber = 10;
accuracy_vector = zeros(1, foldsNumber);
average_accuracy = zeros(1, 3);
standard_deviation = zeros(1, 3);
average_confMat = zeros(2, 2);
windowConfMat1 = 0;
windowConfMat2 = 0;
windowConfMat3 = 0;
fprintf('II_SVM_KFold');

for windowNumber = 1:3

    features_unfiltered = [];
    features_filtered = [];
    labels_unfiltered = [];
    labels_filtered = [];
    
%% Imagini originale 
    for i = 1:numberOfImages
        filePath = fullfile(folderPath, sprintf('%d.pgm', i));
                
        if exist(filePath, 'file')
            image = imread(filePath);
                
% Asigurarea ca imaginea este definita de nivele de gri            
            if size(image, 3) > 1
                grayImg = im2gray(image);
            else
                grayImg = image;
            end
    
% Numara toate cele 17 structuri LBP
             w0_unfiltered = countAllZeroLBP(grayImg);
            w1_unfiltered = vectorCountOneBit1LBPs(grayImg);
            w2_unfiltered = vectorCountTwoBit1LBPs(grayImg);

%% Caracteristici si etichete imagini originale

            imageLBPs_unfiltered = [w0_unfiltered, w1_unfiltered, w2_unfiltered];
            features_unfiltered = [features_unfiltered; imageLBPs_unfiltered];
            labels_unfiltered = [labels_unfiltered; 0];

            else
                fprintf('File %s not found.\n', filePath);
        end
    end
% partitionare a datelor din imagini originale 
    cv_unfiltered = cvpartition(size(features_unfiltered, 1), 'KFold', foldsNumber);    

%% Imagini filtrate 
        
        for i = 1:numberOfImages
            filePath = fullfile(folderPath, sprintf('%d.pgm', i));
                
            if exist(filePath, 'file')
% Filtrarea imaginii
                imgFiltered = MedianFiltering(filePath, windowNumber);
    
% numara toate structurile LBP din imaginile filtrate 
                 w0_filtered = countAllZeroLBP(imgFiltered);
                 w1_filtered = vectorCountOneBit1LBPs(imgFiltered);
                 w2_filtered = vectorCountTwoBit1LBPs(imgFiltered);
    
%% Caracteristici si etichete filtrate

            imageLBPs_filtered = [w0_filtered, w1_filtered, w2_filtered];
            features_filtered = [features_filtered; imageLBPs_filtered];
            labels_filtered = [labels_filtered; 1];
            else
                fprintf('File %s not found.\n', filePath);
            end 
        end

% partitionare date imagini filtrate 
        cv_filtered = cvpartition(size(features_filtered, 1), 'KFold', foldsNumber);
        
        for fold = 1:foldsNumber
            
            trainUnfiltered = training(cv_unfiltered, fold);
            testUnfiltered = test(cv_unfiltered, fold);
            trainFiltered = training(cv_filtered, fold);
            testFiltered = test(cv_filtered, fold);
%% Date de antrenare si testare                        
            XTrain = [features_filtered(trainFiltered, :); features_unfiltered(trainUnfiltered, :)];
            YTrain = [labels_filtered(trainFiltered, :); labels_unfiltered(trainUnfiltered, :)];
            XTest = [features_filtered(testFiltered, :); features_unfiltered(testUnfiltered, :)];
            YTest = [labels_filtered(testFiltered, :); labels_unfiltered(testUnfiltered, :)];
    
            SVMModel = fitcsvm(XTrain, YTrain, 'KernelFunction', 'linear', 'BoxConstraint', 1, 'Standardize', true);
               
% Predictie 
            YPred = predict(SVMModel, XTest);
            
% acuratete
            accuracy = sum(YPred == YTest) / length(YTest);
            accuracy_vector(fold) = accuracy;    
            
% Matrice de confuzie 
            confMat= confusionmat(YTest, YPred);
% Definire matricea de confuzie pentru toate iteratiile in cazul ferestrei
% de forma 3x3
            if windowNumber == 1
                windowConfMat1 = confMat + windowConfMat1;
            end
% Definire matricea de confuzie pentru toate iteratiile in cazul ferestrei
% de forma 5x5
            if windowNumber == 2
                windowConfMat2 = confMat + windowConfMat2;
            end
% Definire matricea de confuzie pentru toate iteratiile in cazul ferestrei
% de forma 7x7
            if windowNumber == 3
                windowConfMat3 = confMat + windowConfMat3;
            end    
            average_confMat = confMat + average_confMat;
        end
% Parametrii statistici
    average_accuracy(windowNumber) = mean(accuracy_vector);
    standard_deviation(windowNumber) = std(accuracy_vector);

end
    average_confMat = average_confMat / (foldsNumber * 1.00 * length(windowNumber));
    disp(average_confMat);
    average_confMat = round(average_confMat);
    confusionchart(average_confMat);
