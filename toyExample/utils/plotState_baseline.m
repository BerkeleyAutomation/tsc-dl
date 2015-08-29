function [ figOutput ] = plotState_baseline( x_curr, target_curr, fAx, background)
   if nargin < 2
       fprintf('plotState needs at least 2 arguments')
       
   elseif nargin == 2
        if ~exist('x_curr', 'var')
            x_curr = [];
        end
        if ~exist('target_curr', 'var')
            target_curr = [1,1];
        end

    elseif nargin == 3
        if ~exist('fAx', 'var')%Input axis object to plot on
            figInput = figure();
            fAx = axes('parent',figInput);
            hold (fAx, 'on')
            axesLim = [-10 10 -10 10];
            axis(fAx, axesLim);
            hold on
        end
    elseif nargin ==4
        if ~exist('background', 'var')
            % background = load('images/whiteBG.jpg');
            set(figInput,'color','w');
        elseif background == 'default'
            background = 'images/living-room.jpg';
            bg_Image(background, 1, 1, figInput)
        else
            bg_Image(['images' filesep background], 1, 1, figInput)
        end   
   end

    %% plot target as flash light
    hTarget = plot(fAx,target_curr(1),target_curr(2), ...
        'Marker','o', ...
        'MarkerEdgeColor', 'r', ... 
        'MarkerFaceColor', 'y', ...
        'MarkerSize',40 );        
%     hTarget.MarkerHandle.EdgeColorData = uint8(255*[1;0;0;0.5]);
    
    %% plot robot as oriented traingle
    hRobot_body = plot(fAx,x_curr(1),x_curr(2), ...
        'Marker','o', ... 
        'MarkerFaceColor', 'b',...
        'MarkerEdgeColor', 'b', ...
        'MarkerSize',25 );        
%     hRobot.MarkerHandle.EdgeColorData = uint8(255*[0;0;1;0.5]);
%     hRobot.MarkerHandle.FaceColorData = uint8(255*[0;0;1;0.5]);    
    
    %%    
    figOutput = [hTarget, hRobot_body ]; %return figure handle
    
   
end



