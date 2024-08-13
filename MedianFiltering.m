function filteredImg = MedianFiltering(imagePath, windowNumber)
    
    RGB = imread(imagePath);
    
    if size(RGB, 3) == 1
        
        if windowNumber == 1
            filteredImg  = medfilt2(RGB); 
        end 
        
        if windowNumber == 2
            filteredImg = medfilt2(RGB, [5, 5]);
        end
        
        if windowNumber == 3
            filteredImg = medfilt2(RGB, [7, 7]);
        end

    else
        R = RGB(:, :, 1);
        G = RGB(:, :, 2);
        B = RGB(:, :, 3);
        
        R = medfilt2(R);
        G = medfilt2(G);
        B = medfilt2(B);
        
        %concatenating all three channels
        filteredImg = cat(3, R, G, B);
    end

end