clc
clear all
close all

%%
numExamples = 5;
numPoints = 5;
stepSize = 0.75;
seed_RNG = rng(10,'twister');
listX = random('unif', -10, 10, 2, numPoints );
c = ['b', 'r', 'c', 'm' , 'y'];
a = 50; %markersize
noiseLevel = [10, 25, 50, 75, 100];% as percent of noise

delete(['baseline' filesep '*.*']); %clear the output folder

%% Without Noise
xcurr = listX(:,1);
xTraj_clean  = xcurr;
scatter(xcurr(1), xcurr(2),a, 'filled', 'MarkerEdgeColor', c(1), 'MarkerFaceColor', c(1))
hold on
axis([-12 12 -12 12])
axis equal

xTranscriptions = [];
for i = 1: numPoints-1        
    x2 = listX(:,i+1);        
    diff = x2 - xcurr;
    xTranscriptions(i, 1) = size(xTraj_clean, 2); %start frame
    while (norm(diff)>0)
        diff = x2 - xcurr;        
        if norm(diff)>stepSize    
            xcurr = xcurr +  stepSize*diff./norm(diff);                            
        else%last step
            xcurr = xcurr+  diff;                  
            xTranscriptions(i, 2) = size(xTraj_clean, 2); %end frame
            if i == numPoints-1 %for last iteration so the traj length is same 
                xTranscriptions(i, 2) = xTranscriptions(i, 2)+1;
            end
        end 
        scatter(xcurr(1), xcurr(2),a, 'filled', 'MarkerEdgeColor', c(i), 'MarkerFaceColor', c(i))
        xTraj_clean = [xTraj_clean, xcurr];
    end    
end

hold off
fileName = ['baseline' filesep 'simpleExpt' '_noiseLevel000'];
save([fileName '.mat'], ...
         'listX', 'xTraj_clean')
save([fileName '-transcriptions' '.mat'], 'xTranscriptions')
 saveas(gcf, fileName, 'jpeg');


%% With Noise
for n = 1: length(noiseLevel)
    for e = 1:numExamples
        xcurr = listX(:,1);
        xTraj_noise = xcurr;        
        figure()
        scatter(xcurr(1), xcurr(2),a, 'filled', 'MarkerEdgeColor', c(1), 'MarkerFaceColor', c(1))
        hold on
        axis([-12 12 -12 12])
        axis equal
        xTranscriptions = [];
        for i = 1: numPoints-1        
            x2 = listX(:,i+1);        
            diff = x2 - xcurr;
            xTranscriptions(i, 1) = size(xTraj_noise, 2); %start frame
            while (norm(diff)>0)
                diff = x2 - xcurr;        
                if norm(diff)>stepSize    
                    xcurr = xcurr +  stepSize*diff./norm(diff) + ...
                        0.01*noiseLevel(n)*random('unif', -stepSize, stepSize, 2, 1);                            
                else%last step
                    xcurr = xcurr+  diff;                  
                    xTranscriptions(i, 2) = size(xTraj_noise, 2); %end frame
                    if i == numPoints-1 %for last iteration so the traj length is same 
                        xTranscriptions(i, 2) = xTranscriptions(i, 2)+1;
                    end
                end 
                scatter(xcurr(1), xcurr(2),a, 'filled', 'MarkerEdgeColor', c(i), 'MarkerFaceColor', c(i))
                xTraj_noise = [xTraj_noise, xcurr];
            end    
        end

        hold off
        fileName = ['baseline' filesep 'simpleExpt' '_noiseLevel' ...
            num2str(noiseLevel(n),'%03d') '_' num2str(e,'%02d')];
        save([fileName '.mat'], ...
             'listX', 'xTraj_noise')   
         save([fileName '-transcriptions' '.mat'], 'xTranscriptions')
         saveas(gcf, fileName, 'jpeg');
    end
end