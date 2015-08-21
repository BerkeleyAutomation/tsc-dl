function [ ] = genLabelFile( frameLabels, outputDir, fileName )

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
    t= 1;
    label = {};
    while t < size(frameLabels,2)      
      if frameLabels(:, t) == frameLabels(:, t+1)%if all values are same assign rotation
          if t == 1, label{t} = 'G2'; end
          label{t+1} = 'G2';%rotation
      elseif frameLabels(3, t) == frameLabels(3, t+1)
          if t == 1, label{t} = 'G1'; end
          label{t+1} = 'G1';%movement
      elseif frameLabels(1:2, t) == frameLabels(1:2, t+1)          
          if t == 1, label{t} = 'G2'; end
          label{t+1} = 'G2';%rotation 
      else
         fprintf('Can only have two labels--check data \n')
         disp(frameLabels(:,t))
      end 
      t = t+1;
    end    
    
    startFrame = 1; 
    for i = 1:length(label)-1
        if ~strcmp(label{i}, label{i+1})          
            %write to file
            fprintf(fid, '%d %d %s \n', startFrame, i, label{startFrame});
            startFrame = i+1; %re-init counter           
        elseif i == length(label)-1 %for last surgeme
            fprintf(fid, '%d %d %s \n', startFrame, i+1, label{startFrame});
        end            
    end

    fclose(fid);
        
end