clear all
close all
% clc
addpath(genpath(pwd))

%% Experiment specifc details
%flags for experiment params
noisyDynamics = false; 
targetNoise = true;
observationNoise = true;

flag_plotTraj = false; 
% In order of increasing difficulty
% exptList = ['0001', '1000', '0100', '0101', '1001', '1101', '1110'];

exptSetup = [num2str(noisyDynamics) num2str(targetNoise) num2str(observationNoise)];

fprintf('Running Sim for [noisyDynamics, targetNoise]:%s \n',...
    exptSetup);
%make correct directory structure
mkdir_if_not_exist ('baseline2');
delete(['baseline2' filesep '*.*']); %clear the output folder

outputDirPrefix = ['baseline2' filesep exptSetup];
mkdir_if_not_exist (outputDirPrefix);

kinDIR = [outputDirPrefix filesep exptSetup '_kinematics'];
mkdir_if_not_exist (kinDIR);

vidTrans= [outputDirPrefix  filesep exptSetup '_video' filesep 'transcriptions'];
mkdir_if_not_exist (vidTrans);
vidFrames = [outputDirPrefix filesep exptSetup '_video' filesep 'frames'];
mkdir_if_not_exist (vidFrames);

%% 

numDemos = 5; 
maxStep = 1;    
xVel = 0.1;

% dynamics noise is used as a percent of maxStep
if noisyDynamics, dynamicsNoise = 0.25; end
% observation noise is used as a percent of maxStep
if observationNoise, obsNoise = 0.25; end
% target noise level absolute values.
if targetNoise, targetNoiseLevel = 1; end

% initialize random number generator
seed_RNG = rng(10,'twister');
numTargets = 5;
% listX = random('unif', -10, 10, 2, numTargets  );
target_list = [1, 1; 
              3, 10;
              5, 2;
              7, 9;
              10, 1]';

if targetNoise
    target_list = target_list + random('unif', -maxStep, maxStep, 2, numTargets);
end
          
%% Initialize
for expt = 1:numDemos
    fprintf('%s: %02d \n', exptSetup, expt);
    
    %% init params for loop
    outputDir = [vidFrames filesep exptSetup '_' num2str(expt,'%02d') '_capture1'];
    mkdir_if_not_exist (outputDir);
    if outputDir(end) ~= '/', outputDir = [outputDir filesep]; end
       
    target_buffer = [];%this buffer list is updated online as targets are reached
    
    %state representation is [x,y]
    x_curr = target_list (:,1);
    
    x_traj = x_curr;                

    figHandle = figure();
    axHandle = axes('parent',figHandle);
    hold (axHandle, 'on')
    axisLim = [-5 15 -5 15];
    axis(axHandle, axisLim);
    axis(axHandle, 'manual')
    axis(axHandle, 'off')

    delete([outputDir '*.*']); %clear the output folder
    
    iterCount = 1;
    xTranscriptions = [];
    % This models as  x(t+1) = x(t) + u(t)    
    %% Iterate
    for targetCount = 2:numTargets 
        % sample new target    
        % target_curr = genTarget();        
        target_curr = target_list(:, targetCount);
        target_buffer = [target_buffer , target_curr];
               
        if targetCount>2
           delete(objHandles) 
        end
        objHandles = plotState_baseline(x_curr, target_curr, axHandle );         
        saveFig( figHandle, iterCount, outputDir );
        
        xTranscriptions(targetCount, 1) = size(x_traj, 2); %start frame
        %% go to target in equisized steps.
        diff = target_curr - x_curr(1:2);
        
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
                xTranscriptions(targetCount, 2) = size(x_traj, 2); %end frame
                    if targetCount == numTargets %for last iteration so the traj length is same 
                        xTranscriptions(targetCount, 2) = xTranscriptions(targetCount, 2)+1;
                    end
            end
            
            delete(objHandles) 
            objHandles = plotState_baseline(x_curr, target_curr, axHandle );             
            if ~observationNoise
                x_traj = [x_traj, x_curr]; %record trajectory        
            else
                 %Only record trajectory with noise, vision sees clean traj
                x_traj = [x_traj, ...
                    x_curr + obsNoise*random('unif',-maxStep,maxStep,2,1)];
            end
            
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
    genLabelFile_baseline (xTranscriptions, vidTrans, [exptSetup '_' num2str(expt,'%02d')] );
    
    close all
end
