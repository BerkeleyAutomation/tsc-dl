function saveFig( figInput, iterCount )
    
    fileName = ['output' filesep  'time' int2str(iterCount)];
    saveas(figInput, fileName, 'jpeg')

end

