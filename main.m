clc;
clear;
close all;

%% Setare date legate de localizarea bazei de date si numarul de iteratii
folderPath = "C:\Facultate\Anul_IV\Licenta\Database\GBRASNET\BOSSbase-1.01\cover"; 
numberOfImages = 200;
numberOfIterations = 1;

%% Initializarea variabilelor
I_accuracy = zeros(1, 3);
I_accuracy_mean = zeros(1, 3);
I_accuracy_iteration = zeros(3, numberOfIterations);
I_standard_deviation = zeros(1, 3);
I_correctDetected_total = 0;
II_accuracy = zeros(1, 3);
II_accuracy_mean = zeros(1, 3);
II_accuracy_iteration = zeros(3, numberOfIterations);
II_standard_deviation = zeros(1, 3);
II_correctDetected_total = 0;
zeroLBPsCount = zeros(1, numberOfImages);
oneLBPsCount = zeros(1, numberOfImages);
filteredZeroLBPCount = zeros(1, numberOfImages);
filteredOneLBPCount = zeros(1, numberOfImages);
leftTerm = zeros(1, numberOfImages);
rightTerm = zeros(1, numberOfImages);
imageArray = cell(numberOfImages, 1);
labels = zeros(1, numberOfImages*2);

for i = 1:400
    labels(i) = 0;
    if i > 200
        labels(i) = 1;
    end
end

for windowNumber = 1:3
    for iteration = 1:numberOfIterations
        I_correctDetected = 0;
        II_correctDetected = 0;
       
%% Algorithm I
        
        for i = 1:numberOfImages 
            filePath = fullfile(folderPath, sprintf('%d.pgm', i));
% verificare existenta fisierului
            if exist(filePath, 'file')
                imageArray{i} = imread(filePath);
                fprintf('\nWindow: %d; Iteration: %d; image number: %d\nFirst Algorithm', windowNumber, iteration, i);
                image = imageArray{i};
% verificare imagine definita pe nivele de gri
                if size(image, 3) > 1
                    grayImg = im2gray(image);
                else
                    grayImg = image;
                end
                
% Numără toate structurile LBP-zero
                zeroLBPsCount(i) = countAllZeroLBP(grayImg);
                
% Numără toate structurile LBP-unu 
                oneLBPsCount(i) = countOneBit1LBP(grayImg);
                
% Filtreaza imaginea originala
                imgFiltered = MedianFiltering(filePath, windowNumber);
        
                filteredZeroLBPCount(i) = countAllZeroLBP(imgFiltered);
                filteredOneLBPCount(i) = countOneBit1LBP(imgFiltered);
                
% d = N0/N1 if d < 1 => the image is median filtered 
%              d > 1 => the image is not median filtered
% Etapa de decizie
                d = zeroLBPsCount(i)/oneLBPsCount(i);
                d1 = filteredZeroLBPCount(i)/filteredOneLBPCount(i);
                if ((d < 1) && (labels(i) == 0))
                    fprintf(' => Filtered: YES\n');
                else
                    I_correctDetected = I_correctDetected + 1;
                    fprintf(' => Filtered: NO\n');
                end

                if ((d1 < 1) && (labels(200+i) == 1))
                    fprintf(' => Filtered: YES\n');
                    I_correctDetected = I_correctDetected + 1;
                else
                    fprintf(' => Filtered: NO\n');
                end
                
%% Algorithm II
                
            fprintf('Second Algorithm');
                
% Extragere parametrii imagini originale
                
            bitsum = 0;
            w0_original = countAllZeroLBP(grayImg);
            w1_original = vectorCountOneBit1LBPs(grayImg);
            w2_original = vectorCountTwoBit1LBPs(grayImg);
                
% Extragere parametrii imagini filtrate
                
            w0_filtered = countAllZeroLBP(imgFiltered);
            w1_filtered = vectorCountOneBit1LBPs(imgFiltered);
            w2_filtered = vectorCountTwoBit1LBPs(imgFiltered);
                
% Realizarea rapoartelor
            flagImageFiltrated = 0;
            dn0 = w0_original/w0_filtered;
            if dn0 > 1
                flagImageFiltrated = flagImageFiltrated + 1;
            end
            for j = 1:8
                dn1 = w1_original(j)/w1_filtered(j);
                dn2 = w2_original(j)/w2_filtered(j);
                if dn1 > 1
                   flagImageFiltrated = flagImageFiltrated + 1;
                end
                if dn2 > 1
                    flagImageFiltrated = flagImageFiltrated + 1;
                end
            end
% Etapa de decizie
            if flagImageFiltrated == 17
                fprintf(' => Filtered: YES\n');
                II_correctDetected = II_correctDetected + 1;
                fprintf('\n');
            else
                fprintf(' => Filtered: NO\n');
                fprintf('\n');                   
            end
                
            else
                fprintf('File %s not found.\n', filePath);
            end
        end
% Definire rezultate statistice pe iteratie si global        
        I_correctDetected_total = I_correctDetected_total + I_correctDetected;
        II_correctDetected_total = II_correctDetected_total + II_correctDetected;

        I_accuracy_iteration(windowNumber, iteration) = (I_correctDetected / (numberOfImages * 2)) * 100;
        II_accuracy_iteration(windowNumber, iteration) = (II_correctDetected / numberOfImages) * 100;
    end
% Definire rezultate statistice pe fereastra de filtrare 
    I_accuracy_mean(windowNumber) = mean(I_accuracy_iteration(windowNumber, :));
    II_accuracy_mean(windowNumber) = mean(II_accuracy_iteration(windowNumber, :));
    
    I_standard_deviation(windowNumber) = std(I_accuracy_iteration(windowNumber, :));
    II_standard_deviation(windowNumber) = std(II_accuracy_iteration(windowNumber, :));

    I_accuracy(windowNumber) = (I_correctDetected_total / (numberOfImages * numberOfIterations * windowNumber * 2)) * 100;
    II_accuracy(windowNumber) = (II_correctDetected_total / (numberOfImages * numberOfIterations * windowNumber)) * 100;
end
% Impartire la 2 pt ca in cazul Alg I, setul de imagini este dublat
I_accuracy_total = I_correctDetected_total/2;

% Afisarea acuratetii finale, globale:
fprintf('\nI Alg Total Accuracy: %.2f%%', I_accuracy);
fprintf('\nII Alg Total Accuracy: %.2f%%', II_accuracy);

%% Graphs

% w1 & w2 -> figure 4
figure(1);
x = 1:1:numberOfImages;
area(x, zeroLBPsCount, 'FaceColor', 'c', 'EdgeColor', 'none');
hold on;
area(x, oneLBPsCount, 'FaceColor', 'y', 'EdgeColor', 'none');
xlabel('Număr de imagini');
ylabel('Număr de valori LBP');
title('Număr LBP din imagini originale');
legend('Număr LBP-zero original', 'Număr LBP-unu original');

% w1 & w1M -> figure 5
figure(2);
area(x, zeroLBPsCount, 'FaceColor', 'b', 'EdgeColor', 'none');
hold on;
area(x, filteredZeroLBPCount, 'FaceColor', 'r', 'EdgeColor', 'none');
xlabel('Număr de imagini');
ylabel('Numărul structurilor LBP-zero');
title('Numărul de structuri LBP-zero din imaginea originala și filtrată');
legend('Număr de LBP-zero original', 'Număr de LBP-zero filtrat');

% w2 & w2M -> figure 6
figure(3);
area(x, oneLBPsCount, 'FaceColor', 'g', 'EdgeColor', 'none');
hold on;
area(x, filteredOneLBPCount, 'FaceColor', 'm', 'EdgeColor', 'none');
xlabel('Număr de imagini');
ylabel('Număr valori LBP-unu');
title('Număr LBP unu din imagini originale și filtrate');
legend('Număr LBP-unu original', 'Număr LBP-unu filtrat');

% (w1 - w1M) - (w2 - w2M) > w1 - w2  -> figure 7
leftTerm = ((zeroLBPsCount - filteredZeroLBPCount)-(oneLBPsCount-filteredOneLBPCount));
rightTerm = zeroLBPsCount - oneLBPsCount;
figure(4);
area(x, leftTerm, 'FaceColor', 'k', 'EdgeColor', 'none');
hold on;
area(x, rightTerm, 'FaceColor', 'g', 'EdgeColor', 'none');
xlabel('Numărul de imagini');
ylabel('Valoarea termenilor');
legend('Rezultatul calculului 𝑑𝑖𝑓1 − 𝑑𝑖𝑓2', 'Rezultatul calculului 𝑑𝑖𝑓3');

figure(5);
hold on;
plot(x, leftTerm-rightTerm, 'k', 'LineWidth', 2); 
yline(0, '--r', 'LineWidth', 2);
xlabel('Numărul de imagini');
ylabel('Valoarea ecuației (18)');
legend('Rezultatul calculului (𝑑𝑖𝑓1 − 𝑑𝑖𝑓2) − 𝑑𝑖𝑓3')
hold off;

% totalitatea structurilor de tip LBP-unu -> figure 8
data = [w1_original; w1_filtered]';
xLabels = {'00000001', '00000010', '00000100', '00001000', '00010000', '00100000', '01000000', '10000000'};
% Create grouped bar chart
figure(6);
b1 = bar(data);
b1(1).FaceColor = 'c';
b1(2).FaceColor = 'm';
set(gca, 'XTickLabel', xLabels);
xlabel('Structuri LBP-unu');
ylabel('Număr de structuri LBP-unu');
legend('Original LBP-unu', 'Filtrat LBP-unu');
title('Comparație între structurile LBP-unu original și filtrat');
grid on;

% totalitatea structurilor de tip LBP-doi -> figure 9
data1 = [w2_original; w2_filtered]';
xLabels = {'00000011', '00000110', '00001100', '00011000', '00110000', '01100000', '11000000', '10000001'};
figure(7);
bar(data1);
set(gca, 'XTickLabel', xLabels);
xlabel('Structuri LBP-doi');
ylabel('Numărul de LBP-doi');
legend('LBP-doi original', 'LBP-doi filtrat');
title('Comparație între numărul de LBP-doi în imagini originale și filtrate');
grid on;

% w1M & w2M -> figure 8
figure(8);
x = 1:1:numberOfImages;
area(x, filteredZeroLBPCount, 'FaceColor', 'c', 'EdgeColor', 'none');
hold on;
area(x, filteredOneLBPCount, 'FaceColor', 'y', 'EdgeColor', 'none');
xlabel('Număr de imagini');
ylabel('Număr de valori LBP');
title('Număr LBP din imagini filtrate');
legend('Număr LBP-zero filtrat', 'Număr LBP-unu filtrat');

