function [ trajHandle ] = plotTraj( x_traj, axHandle )
    
%     persistent trajHandle
%     delete(trajHandle)
    trajHandle = plot(axHandle, x_traj(1,end), x_traj(2,end),':g*', 'LineWidth', 1);
    uistack(trajHandle, 'bottom')

end

