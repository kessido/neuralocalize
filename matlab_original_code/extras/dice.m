function d = dice(x,y)
% d = dice(x,y)
%
% x should be N*nx
% y should be N*ny
%
% then d is nx*ny


[N,nx]    = size(x);
[test,ny] = size(y);

if(N~=test)
    error('x and y incompatible');
end

xx=repmat(sum(x,1),ny,1)';
yy=repmat(sum(y,1),nx,1);

d = 2 * (double(x)'*double(y)) ./ (xx+yy);