function saveFig( figInput, iterCount, outputDir )
    
    if nargin == 2
        fileName = ['output' filesep  'time' int2str(iterCount)];        
    elseif nargin == 3
        mkdir_if_not_exist (outputDir);
        if outputDir(end) ~= '/', outputDir = [outputDir filesep]; end        
        fileName = [outputDir num2str(iterCount,'%06d')]; 
    end
%     figInput.PaperPositionMode = 'manual';
    saveas(figInput, fileName, 'jpeg')

end



