function ret=zeropad(x,n)
% out = zeropad(in,n)
% in = text or num

if(isstr(x))
  x=str2num(x);
end
ret=dec2base(x,10,n);