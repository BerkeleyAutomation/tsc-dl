clear all
close all
% clc
addpath(genpath(pwd))

%% Experiment specifc details
%flags for experiment params
varyRobotInit = false;
noisyDynamics = true; 
varyTargetInit = false;
targetNoise = false;

% In order of increasing difficulty
% exptList = ['0001', '1000', '0100', '0101', '1001', '1101', '1110'];

exptSetup = [num2str(varyRobotInit) num2str(noisyDynamics)...
    num2str(varyTargetInit) num2str(targetNoise)];

fprintf('Running Sim for [varyRobotInit, noisyDynamics, varyTargetInit, targetNoise]:%s \n',...
    exptSetup);
%make correct directory structure
mkdir_if_not_exist ('output');

outputDirPrefix = ['output' filesep exptSetup];
mkdir_if_not_exist (outputDirPrefix);

kinDIR = [outputDirPrefix filesep exptSetup '_kinematics'];
mkdir_if_not_exist (kinDIR);

vidTrans= [outputDirPrefix  filesep exptSetup '_video' filesep 'transcriptions'];
mkdir_if_not_exist (vidTrans);
vidFrames = [outputDirPrefix filesep exptSetup '_video' filesep 'frames'];
mkdir_if_not_exist (vidFrames);

numDemos = 5; 
flag_plotTraj = true; 

% initialize random number generator
seed_RNG = rng(1,'twister');
% if strcmp(exptSetup, '0101')
%     seed_RNG = rng(10,'twister');%for exptsetup only 0101
% end

%% Initialize
for expt = 1:numDemos
    fprintf('%s: %02d \n', exptSetup, expt);
    
    %% init params for loop
    outputDir = [vidFrames filesep exptSetup '_' num2str(expt,'%02d') '_capture1'];
    mkdir_if_not_exist (outputDir);
    if outputDir(end) ~= '/', outputDir = [outputDir filesep]; end

%     pause('on') %for save fig
       
    numTargets = 3;
    
    %to use the same targets as first iteration
    if expt==1 
        target_list = random('unif', -10, 10, 2, numTargets);
    end    
    target_buffer = [];%this buffer list is updated online as targets are reached
    
    %generate new targets for every expt run
    if expt>1 && varyTargetInit
        target_list = random('unif', -10, 10, 2, numTargets);
    end
       
    %state representation is [x,y,theta (radian)]
    if ~varyRobotInit    
        x_curr = [0;0;0];
    else
        %rng(100,'twister');
        x_curr = [random('unif',-10,10, 2,1); random('unif',-pi, pi) ];
        %reset the seed after generating random robot position so that we
        %get same target positions
        %rng(seed_RNG);
    end    
    
    x_traj = x_curr;        
    
    maxStep = 1;%only works for maxStep greater than one
    rotStepSize = pi/18;
    % dynamics noise is used as a percent of maxStep
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

    delete([outputDir '*.*']); %clear the output folder
    
    iterCount = 1;

    % This models as  x(t+1) = x(t) + u(t)
    
    %% Iterate
    for targetCount = 1:numTargets 
        % sample new target    
        % target_curr = genTarget();        
        target_curr = target_list(:, targetCount);
        
        if targetNoise 
            target_curr =  target_curr+ ...
                random('unif',-targetNoiseLevel,targetNoiseLevel, 2,1);
            % make sure target is in [-10,10]x[-10,10]
            if sum(abs(target_curr)>10)>=1
                target_curr(abs(target_curr)>10) = ...
                    10*sign(target_curr(abs(target_curr)>10));
            end            
        end
        
        target_buffer = [target_buffer , target_curr];
               
        if targetCount>1
           delete(objHandles) 
        end
        objHandles = plotState(x_curr, target_curr, axHandle );         
        saveFig( figHandle, iterCount, outputDir );

        %% search for target -- No Noise in rotation
        diff = target_curr - x_curr(1:2);
        headingToTarget = atan2 (diff(2), diff(1));

        sDir = -pi:rotStepSize:pi;
        sDir = sort([sDir(2:end), headingToTarget, x_curr(3)]);
        ind_curr = find(sDir == x_curr(3) );
        ind_target = find(sDir == headingToTarget );

        %to handle cyclic nature of rotations.
        %add ind_curr+1 to not include current state as one step.
        if min(ind_curr) >= ind_target
            searchArray = [sDir(max(ind_curr)+1:end), sDir(1:ind_target)];
        else
            searchArray = sDir(max(ind_curr)+1:ind_target);
        end        

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
            if  norm(diff)>maxStep            
                if ~noisyDynamics % Perfect update
                    x_curr(1:2)= x_curr(1:2) + maxStep*diff./norm(diff);             
                else%noisy update based on an attracter
                    x_curr(1:2)= x_curr(1:2) + maxStep*diff./norm(diff) +...
                        dynamicsNoise*random('unif',-maxStep,maxStep,2,1);
                end

            else %last step to reach to target
                x_curr(1:2) = x_curr(1:2)+ diff; 
            end
            
            delete(objHandles) 
            objHandles = plotState(x_curr, target_curr, axHandle );             
            x_traj = [x_traj, x_curr]; %record trajectory        
            if flag_plotTraj
                trajHandle = plotTraj( x_traj, axHandle );
            end

            iterCount = iterCount+1;
            saveFig( figHandle, iterCount, outputDir );
            
        end

        %% increment target counter
        %targetCount = targetCount +1;
    end
    
    save([kinDIR filesep exptSetup '_' num2str(expt,'%02d') '.mat'],...
        'target_buffer','x_traj')    
    genLabelFile (x_traj, vidTrans, [exptSetup '_' num2str(expt,'%02d')] );
    
    close all
end
