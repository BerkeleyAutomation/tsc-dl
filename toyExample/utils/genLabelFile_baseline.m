function [ ] = genLabelFile_baseline( frameLabels, outputDir, fileName )

    if nargin <2
        fprintf('saving labels file in ./output/ \n')        
        fileName = ['output' filesep 'frameLabels'];
    elseif nargin == 2
        mkdir_if_not_exist (outputDir);
        if outputDir(end) ~= '/', outputDir = [outputDir filesep]; end
        fileName = [outputDir 'frameLabels'];        
    elseif nargin == 3
        mkdir_if_not_exist (outputDir);
        if outputDir(end) ~= '/', outputDir = [outputDir filesep]; end
        fileName = [outputDir fileName];        
    end
    
    % make textFile
    fid = fopen( [fileName '.txt'], 'wt' );
    
    % write to textFile    
    for i = 1:size(frameLabels,1)               
        fprintf(fid, '%d %d %s \n', frameLabels(i,1), frameLabels(i,2), ['G' num2str(i,'%02d')]);        
    end
    fclose(fid);
        
end