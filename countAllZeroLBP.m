function zeroLBPsNumber = countAllZeroLBP(image)
    
    [rows, cols] = size(image);
    zeroLBPsNumber = 0;

    for i = 2:rows-1
        for j = 2:cols-1
            centerPixel = image(i, j);

            pattern = zeros(1,8);
            pattern(1) = image(i-1, j-1) >= centerPixel;
            pattern(2) = image(i-1, j) >= centerPixel;
            pattern(3) = image(i-1, j+1) >= centerPixel;
            pattern(4) = image(i, j+1) >= centerPixel;
            pattern(5) = image(i+1, j+1) >= centerPixel;
            pattern(6) = image(i+1, j) >= centerPixel;
            pattern(7) = image(i+1, j-1) >= centerPixel;
            pattern(8) = image(i, j-1) >= centerPixel;
            
            lbpValue = bi2de(pattern, 'left-msb');
            
            if lbpValue == 0
                zeroLBPsNumber = zeroLBPsNumber + 1;
            end
        end
    end
end