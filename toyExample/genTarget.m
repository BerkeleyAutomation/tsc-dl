function [ targetNext ] = genTarget(  )
    % Load previous generator settings.
%     rng(s);

    pd = makedist('Uniform','lower',-10,'upper',10);   
    targetNext = random(pd, 2,1);
    
end

