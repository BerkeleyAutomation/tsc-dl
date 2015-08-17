clear all
close all
% clc

%% Initialize
seed_RNG = rng(10,'twister');
pause('on')

numTargets = 5;
%state representation is [x,y,theta (radian)]
x_curr = [0;0;0];
x_traj = x_curr;
target_buffer = zeros(2, numTargets);
minStep = 1;

figHandle = figure();
axHandle = axes('parent',figHandle);
hold (axHandle, 'on')
axisLim = [-12 12 -12 12];
axis(axHandle, axisLim);
axis(axHandle, 'manual')

delete(['output' filesep '*.jpg']); %clear the output folder

targetCount = 1;
iterCount = 1;

%% Iterate
while targetCount <= numTargets 
    %% sample new target
    target_curr = genTarget();
    target_buffer(:,targetCount) = target_curr;
    if targetCount>1
       delete(objHandles) 
    end
    objHandles = plotState(x_curr, target_curr, axHandle );         
    saveFig( figHandle, iterCount );
            
    %% search for target
    diff = target_curr - x_curr(1:2);
    headingToTarget = atan2 (diff(2), diff(1));
   
    sDir = -pi:pi/18:pi;
    sDir = sort([sDir, headingToTarget, x_curr(3)]);
    ind_curr = find(sDir == x_curr(3) );
    ind_target = find(sDir == headingToTarget );
    
    if ind_curr >= ind_target
        searchArray = [sDir(ind_curr:end), sDir(1:ind_target)];
    else
        searchArray = sDir(ind_curr:ind_target);
    end
    
    % searchArray = searchArray + x_curr(3)*ones(1, length(searchArray));
        
    for s = 1:length(searchArray)        
        x_curr(3) = searchArray (s);
        delete(objHandles) 
        objHandles = plotState(x_curr, target_curr, axHandle );         
        x_traj = [x_traj, x_curr]; %record trajectory                   
        
        iterCount = iterCount+1;
        saveFig( figHandle, iterCount );
    end
    
    %% go to target in equisized steps.
    
    while (norm(diff)>0)
        diff = target_curr - x_curr(1:2);        
        if  norm(diff)>minStep
            x_curr(1:2)= x_curr(1:2) + diff./norm(diff);
            delete(objHandles) 
            objHandles = plotState(x_curr, target_curr, axHandle );             
            x_traj = [x_traj, x_curr]; %record trajectory        
            trajHandle = plotTraj( x_traj, axHandle );
            
            iterCount = iterCount+1;
            saveFig( figHandle, iterCount );

        else 
            x_curr(1:2) = x_curr(1:2)+ diff; %last step to reach to target
            delete(objHandles) 
            objHandles = plotState(x_curr, target_curr, axHandle );             
            x_traj = [x_traj, x_curr]; %record trajectory        
            trajHandle = plotTraj( x_traj, axHandle );
            
            iterCount = iterCount+1;
            saveFig( figHandle, iterCount );

        end
    end
    
    %% increment target counter
    targetCount = targetCount +1;
end

save(['output' filesep 'kinematics.mat'],'target_buffer','x_traj')

