clc
clear all
close all

%%
numPoints = 5;
stepSize = 0.75;
seed_RNG = rng(10,'twister');
listX = random('unif', -10, 10, 2, numPoints );
c = ['b', 'r', 'c', 'm' , 'y'];
a = 50; %markersize
noiseLevel = [10, 25, 50, 75, 100];% as percent of noise

%% Without Noise
xcurr = listX(:,1);
xTraj_clean  = xcurr;
scatter(xcurr(1), xcurr(2),a, 'filled', 'MarkerEdgeColor', c(1), 'MarkerFaceColor', c(1))
hold on
axis([-12 12 -12 12])
axis equal

for i = 1: numPoints-1        
    x2 = listX(:,i+1);        
    diff = x2 - xcurr;
    while (norm(diff)>0)
        diff = x2 - xcurr;        
        if norm(diff)>stepSize    
            xcurr = xcurr +  stepSize*diff./norm(diff);                            
        else%last step
            xcurr = xcurr+  diff;                  
        end 
        scatter(xcurr(1), xcurr(2),a, 'filled', 'MarkerEdgeColor', c(i), 'MarkerFaceColor', c(i))
        xTraj_clean = [xTraj_clean; xcurr];
    end    
end

hold off
fileName = ['baseline' filesep 'simpleExpt' '_noiseLevel000'];
save([fileName '.mat'], ...
         'listX', 'xTraj_clean')     
 saveas(gcf, fileName, 'jpeg');


%% With Noise
for n = 1: length(noiseLevel)
    
    xcurr = listX(:,1);
    xTraj_noise = xcurr;
    figure()
    scatter(xcurr(1), xcurr(2),a, 'filled', 'MarkerEdgeColor', c(1), 'MarkerFaceColor', c(1))
    hold on
    axis([-12 12 -12 12])
    axis equal

    for i = 1: numPoints-1        
        x2 = listX(:,i+1);        
        diff = x2 - xcurr;
        while (norm(diff)>0)
            diff = x2 - xcurr;        
            if norm(diff)>stepSize    
                xcurr = xcurr +  stepSize*diff./norm(diff) + ...
                    0.01*noiseLevel(n)*random('unif', -stepSize, stepSize, 2, 1);                            
            else%last step
                xcurr = xcurr+  diff;                  
            end 
            scatter(xcurr(1), xcurr(2),a, 'filled', 'MarkerEdgeColor', c(i), 'MarkerFaceColor', c(i))
            xTraj_noise = [xTraj_noise; xcurr];
        end    
    end

    hold off
    fileName = ['baseline' filesep 'simpleExpt' '_noiseLevel' num2str(noiseLevel(n),'%03d')];
    save([fileName '.mat'], ...
         'listX', 'xTraj_noise')     
     saveas(gcf, fileName, 'jpeg');
end