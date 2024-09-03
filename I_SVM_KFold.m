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
fprintf('I_SVM_KFold');

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
    
% Numara structurile LBP-zero si LBP-unu de forma [00000001]
            zeroLBPsCount = countAllZeroLBP(grayImg);
            oneLBPsCount = countOneBit1LBP(grayImg);
    
%% Caracteristici si etichete imagini originale 
            imageLBPs_unfiltered = [zeroLBPsCount, oneLBPsCount];
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
    
% Numararea structurilor LBP-zero si LBP-unu de forma [00000001] din
% imaginile filtrate
                filteredZeroLBPCount = countAllZeroLBP(imgFiltered);
                filteredOneLBPsCount = countOneBit1LBP(imgFiltered);
        
%% Caracteristici si etichete filtrate
                imageLBPs_filtered = [filteredZeroLBPCount, filteredOneLBPsCount];
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

% Acuratete -> all cases when YPred(i) == YTest(i)
            accuracy = sum(YPred == YTest) / length(YTest);
            accuracy_vector(fold) = accuracy;    
     
% Matrice de confuzie 
            confMat = confusionmat(YTest, YPred);    
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

    fprintf('\nFinal Accuracy %i: %.2f%%\n', windowNumber, average_accuracy * 100);

end
    average_confMat = average_confMat / (foldsNumber * 1.00 * length(windowNumber));

%% Grafice

average_confMat = round(average_confMat);
confusionchart(average_confMat);

[x1Grid, x2Grid] = meshgrid(min(XTrain(:,1)):0.5:max(XTrain(:,1)), min(XTrain(:,2)):0.5:max(XTrain(:,2)));
xGrid = [x1Grid(:), x2Grid(:)];

[~, scores] = predict(SVMModel, xGrid);

figure(1);
gscatter(XTrain(:,1), XTrain(:,2), YTrain);
hold on
plot(XTrain(SVMModel.IsSupportVector,1), XTrain(SVMModel.IsSupportVector,2), 'ko', 'MarkerSize', 10);
contour(x1Grid, x2Grid, reshape(scores(:,2), size(x1Grid)), [0 0], 'k');
title(sprintf('Clasificarea datelor'));
xlabel('Număr LBP-zero');
ylabel('Număr LBP-unu');
legend({'Nefiltrat', 'Filtrat', 'Vectori de suport'}, 'Location', 'Best');
hold off

