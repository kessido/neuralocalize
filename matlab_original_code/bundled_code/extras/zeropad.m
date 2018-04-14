function ret=zeropad(x,n)
% str = zeropad(x,n)
% x can be a numerical or a string
% S Jbabdi 

if(ischar(x))
  x=str2num(x);
end
ret=dec2base(x,10,n);