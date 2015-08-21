function [x_next] = dynamics( x_curr, u_curr)

% x = [x; xdot] 
% u = [accel in nDim]

mass = ;
f = 10; %deceleration due to friction 
nDim = length(x_curr)/2;

xddot = [u_curr-f*ones(nDim)];

x_next = [];
% s = s0 + u.t + 1/2 a.t^2 with t = 1;
x_next(1:nDim) = x_curr(1:nDim) + x_curr(nDim+1:2*nDim) + 0.5*xddot; 
% v = u + a.t
x_next(nDim+1:2*nDim) = x_curr(nDim+1:2*nDim) + xddot;

end