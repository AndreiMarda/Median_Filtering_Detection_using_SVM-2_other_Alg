clc;
clear;
close all;

folderPath = "C:\Facultate\Anul_IV\Licenta\Database\GBRASNET\BOSSbase-1.01\cover"; 

%% Initializarea variabilelor
numberOfImages = 200;
numberOfIterations = 10;
average_accuracy = zeros(1, 3);
accuracy = zeros(3, numberOfIterations);
standard_deviation = zeros(1, 3);
initial_cv_filtered = zeros(1, numberOfImages);
initial_cv_unfiltered = zeros(1, numberOfImages);
average_confMat = zeros(2, 2);
windowConfMat1 = 0;
windowConfMat2 = 0;
windowConfMat3 = 0;
fprintf('II_SVM_Holdout');

for windowNumber = 1:3
    for iteration = 1:numberOfIterations
        accuracy_vector = zeros(1, numberOfIterations);
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
cv_unfiltered = cvpartition(size(features_unfiltered, 1), 'Holdout', 0.15);    

%% Imagini filtrate 
    
    for i = 1:numberOfImages
        filePath = fullfile(folderPath, sprintf('%d.pgm', i));
            
        if exist(filePath, 'file')

% Filtrarea imaginii
            imgFiltered = MedianFiltering(filePath, windowNumber);

% Numara toate cele 17 structuri LBP
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
    cv_filtered = cvpartition(size(features_filtered, 1), 'Holdout', 0.15);

%% Date de antrenare       
    XTrain = [features_filtered(training(cv_filtered), :); features_unfiltered(training(cv_unfiltered), :)];
    YTrain = [labels_filtered(training(cv_filtered), :); labels_unfiltered(training(cv_unfiltered), :)];
% mentinerea setului de testare in cazul modificarii marimii ferestrei de
% filtrare
    if windowNumber > 1
        XTest = [features_filtered(initial_cv_filtered, :); features_unfiltered(initial_cv_unfiltered, :)];
        YTest = [labels_filtered(initial_cv_filtered, :); labels_unfiltered(initial_cv_unfiltered, :)];
    else
        XTest = [features_filtered(test(cv_filtered), :); features_unfiltered(test(cv_unfiltered), :)];
        YTest = [labels_filtered(test(cv_filtered), :); labels_unfiltered(test(cv_unfiltered), :)];

    end
%     SVMModel = fitcsvm(XTrain(:, 1:2), YTrain, 'KernelFunction', 'linear', 'BoxConstraint', 1, 'Standardize', true);
      SVMModel = fitcsvm(XTrain, YTrain, 'KernelFunction', 'linear', 'BoxConstraint', 1, 'Standardize', true);
    
    if windowNumber == 1
        initial_cv_filtered = test(cv_filtered);
        initial_cv_unfiltered = test(cv_unfiltered);
    end

%% Testare

%     YPred = predict(SVMModel, XTest(:, 1:2));
    YPred = predict(SVMModel, XTest);

    accuracy(windowNumber, iteration) = sum(YPred == YTest) / length(YTest);
    accuracy_vector(iteration) = accuracy(windowNumber, iteration);    
    fprintf('\nAccuracy for window %i: %.2f%%\n',windowNumber, accuracy(windowNumber, iteration) * 100);
    
% Matricea de confuzie
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
    average_accuracy(windowNumber) = accuracy(windowNumber, iteration) + average_accuracy(windowNumber);
    average_confMat = confMat + average_confMat;

    end
    average_accuracy(windowNumber) = (average_accuracy(windowNumber)/length(average_accuracy(windowNumber))) * 10;
    standard_deviation(windowNumber) = std(accuracy(windowNumber, :));

    average_confMat = average_confMat / (numberOfIterations * 1.00 * length(windowNumber));
    fprintf('\nAverage accuracy for window %i: %.2f%%\n', windowNumber, (sum(average_accuracy)/length(average_accuracy)) * 100);
end

%% Grafice

% disp(average_confMat);
% average_confMat = round(average_confMat);
% confusionchart(average_confMat);
% 
% [x1Grid, x2Grid] = meshgrid(min(XTrain(:,1)):0.5:max(XTrain(:,1)), min(XTrain(:,2)):0.5:max(XTrain(:,2)));
% xGrid = [x1Grid(:), x2Grid(:)];
% 
% [~, scores] = predict(SVMModel, xGrid);
% 
% figure(1);
% gscatter(XTrain(:,1), XTrain(:,2), YTrain);
% hold on
% plot(XTrain(SVMModel.IsSupportVector,1), XTrain(SVMModel.IsSupportVector,2), 'ko', 'MarkerSize', 10);
% contour(x1Grid, x2Grid, reshape(scores(:,2), size(x1Grid)), [0 0], 'k');
% title(sprintf('Clasificarea datelor'));
% xlabel('Număr LBP-zero');
% ylabel('Număr LBP-unu');
% legend({'Nefiltrat', 'Filtrat', 'Vectori de suport'}, 'Location', 'Best');
% hold off
