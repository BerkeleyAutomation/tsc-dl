clear all
close all
% clc
addpath(genpath(pwd))

%% Experiment specifc details
%flags for experiment params
varyRobotInit = false;
noisyDynamics = true; 
varyTargetInit = false;
targetNoise = true;

exptSetup = [num2str(varyRobotInit) num2str(noisyDynamics)...
    num2str(varyTargetInit) num2str(targetNoise)];
% output_[string to specify current params]
mkdir_if_not_exist ('output');
outputDirPrefix = ['output' filesep exptSetup];

numDemos = 3; 
flag_plotTraj = true; 

%% Initialize
for expt = 1:numDemos
    %% init params for loop
    outputDir = [outputDirPrefix filesep num2str(expt,'%02d') ];
    mkdir_if_not_exist (outputDir);
    if outputDir(end) ~= '/', outputDir = [outputDir filesep]; end
    % fileName = [outputDir 'time' int2str(iterCount)];
    
    if expt ==1
        if (~varyTargetInit)
            %seed_RNG = rng(1,'twister');
            seed_RNG = rng(10,'twister');%for exptsetup only 0101
        else
            seed_RNG = rng;
        end
    end

    pause('on')

    numTargets = 3;
    %state representation is [x,y,theta (radian)]
    if ~varyRobotInit    
        x_curr = [0;0;0];
    else
        x_curr = [random('unif',-10,10, 2,1); random('unif',-pi, pi) ];
    end

    x_traj = x_curr;
    target_buffer = zeros(2, numTargets);
    minStep = 1;
    % dynamics noise is used as a percent of minStep
    if noisyDynamics, dynamicsNoise = 0.5; end
    % target noise level absolute values.
    if targetNoise, targetNoiseLevel = 1; end

    figHandle = figure();
    axHandle = axes('parent',figHandle);
    hold (axHandle, 'on')
    axisLim = [-12 12 -12 12];
    axis(axHandle, axisLim);
    axis(axHandle, 'manual')
    axis(axHandle, 'off')
    % removes the axes without removing the whitebox
    % axis off results in plotting over a grey area
%     set(axHandle,'YColor',[1 1 1],'XColor',[1 1 1],...
%         'xtick',[],'ytick',[])

    delete([outputDir '*.*']); %clear the output folder

    targetCount = 1;
    iterCount = 1;

    % This models as 
    % x(t+1) = x(t) + u(t)
    
    %% Iterate
    while targetCount <= numTargets 
        % sample new target
        target_curr = genTarget();
        
        if targetNoise 
            if expt ==1 
                target_mean(:,targetCount) = target_curr;%save the means from first time around
            elseif expt>1
                target_curr =  target_mean(:, targetCount) +...
                    random('unif',-targetNoiseLevel,targetNoiseLevel, 2,1);
            end
            if sum(abs(target_curr)>10)>=1
                target_curr(abs(target_curr)>10) = 10*sign(target_curr(abs(target_curr)>10));
            end            
        end
        
        target_buffer(:,targetCount) = target_curr;
        if targetCount>1
           delete(objHandles) 
        end
        objHandles = plotState(x_curr, target_curr, axHandle );         
        saveFig( figHandle, iterCount, outputDir );

        %% search for target -- No Noise in rotation
        diff = target_curr - x_curr(1:2);
        headingToTarget = atan2 (diff(2), diff(1));

        sDir = -pi:pi/18:pi;
        sDir = sort([sDir, headingToTarget, x_curr(3)]);
        ind_curr = find(sDir == x_curr(3) );
        ind_target = find(sDir == headingToTarget );

        %to handle cyclic nature of rotations.
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
            saveFig( figHandle, iterCount, outputDir );
        end

        %% go to target in equisized steps.

        while (norm(diff)>0)
            diff = target_curr - x_curr(1:2);        
            if  norm(diff)>minStep            
                if ~noisyDynamics % Perfect update
                    x_curr(1:2)= x_curr(1:2) + diff./norm(diff);             
                else%noisy update based on an attracter
                    x_curr(1:2)= x_curr(1:2) + diff./norm(diff) +...
                        dynamicsNoise*random('unif',0,minStep,2,1);
                end
                delete(objHandles) 
                objHandles = plotState(x_curr, target_curr, axHandle );             
                x_traj = [x_traj, x_curr]; %record trajectory        
                if flag_plotTraj
                    trajHandle = plotTraj( x_traj, axHandle );
                end

                iterCount = iterCount+1;
                saveFig( figHandle, iterCount, outputDir );

            else %last step to reach to target
                x_curr(1:2) = x_curr(1:2)+ diff; 
                delete(objHandles) 
                objHandles = plotState(x_curr, target_curr, axHandle );             
                x_traj = [x_traj, x_curr]; %record trajectory        
                if flag_plotTraj
                    trajHandle = plotTraj( x_traj, axHandle );
                end

                iterCount = iterCount+1;
                saveFig( figHandle, iterCount, outputDir);

            end
        end

        %% increment target counter
        targetCount = targetCount +1;
    end

    save([outputDir 'kinematics.mat'],'target_buffer','x_traj')
    genLabelFile (x_traj, outputDir)
    close all
end
