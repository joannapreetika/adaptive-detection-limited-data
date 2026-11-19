function xc = first_crossing(x, y, level, direction)
%FIRST_CROSSING  First x at which y crosses a level (linear interpolation).
%   xc = first_crossing(x, y, level, 'below')   first x where y drops below level
%   xc = first_crossing(x, y, level, 'above')   first x where y reaches level
%   NaN if it never happens; x(1) if it already holds at the first point.

    if strcmp(direction, 'below'), i = find(y < level, 1);
    else,                          i = find(y >= level, 1);
    end
    if isempty(i), xc = NaN;
    elseif i == 1, xc = x(1);
    else,          xc = x(i-1) + (level - y(i-1)) * (x(i) - x(i-1)) / (y(i) - y(i-1));
    end
end
