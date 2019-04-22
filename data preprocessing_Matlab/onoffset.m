function [ ind ] = onoffset( interval,mode )
%Function calculates on/off set of QRS complexe
slope = [];
for i = 2:length(interval)-1
    slope(end+1) = interval(i+1)-interval(i-1);
end
%   using MIN_SLOPE determine onset placement
if strcmp(mode,'on')
    [m,ind] = min(abs(slope));
    %display('onset detected');
elseif strcmp(mode,'off')
    slope_th = 0.2*max(abs(slope));
    slope_s = find(abs(slope)>=slope_th);
    ind = slope_s(1);
else
    display('wrong input, please select on/off set')
end

end

