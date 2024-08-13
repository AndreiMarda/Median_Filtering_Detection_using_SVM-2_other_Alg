function w2_original = vectorCountTwoBit1LBPs(image)

    [rows, cols] = size(image);
    w2_original = zeros(1, 8); 
    
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
            
            bitsum = sum(pattern);
            
            if bitsum == 2 % if there are 2 bits of 1
                if pattern(1) == 1 && pattern (8) == 1
                        w2_original(8) = w2_original(8)+1;
                end
                for k = 1:7 % loop all the elements from the matrix
                    if pattern(k) == 1 && pattern(k+1) == 1 
                        w2_original(k) = w2_original(k)+1;
                    end
                end
            end
        end
    end
end