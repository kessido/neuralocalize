function [ output_args ] = ciftisave(cifti,filename,N)
%  ciftisave(cifti,filename,caret7command,N)
% Save a CIFTI file as a GIFTI external binary and then convert it to CIFTI

caret7command='wb_command';

if(nargin<3)
    N=size(cifti.cdata,1);
end
% Fix Header (Saad Jbabdi)
str=cifti.private.data{1}.metadata(1).value;
str=strrep(str,num2str(N),num2str(size(cifti.cdata,1)));
cifti.private.data{1}.metadata(1).value=str;

tic
save(cifti,[filename '.gii'],'ExternalFileBinary')
toc

tic
unix([caret7command ' -cifti-convert -from-gifti-ext ' filename '.gii ' filename]);
toc

unix([' rm ' filename '.gii ' filename '.dat']);

end

