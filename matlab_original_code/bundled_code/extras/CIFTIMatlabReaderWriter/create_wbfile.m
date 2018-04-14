function create_wbfile(data,fname,type,hemi)
% create_wbfile(data,fname,type,hemi)
%
% S Jbabdi 04/13
addpath /home/fs0/saad/matlab/CIFTIMatlabReaderWriter_old

d='/vols/Scratch/saad/example_gifti_cifti';
switch(lower(type))
    case 'func'
        if(~strcmp(fname(end-7:end),'func.gii'))
            fname=[fname '.func.gii'];
        end
        gii=gifti([d '/' hemi '.func.gii']);
        gii.cdata=data;
        save(gii,fname);
    case 'dconn'
        if(~strcmp(fname(end-8:end),'dconn.nii'))
            fname=[fname '.dconn.nii'];
        end
        cifti=myciftiopen([d '/' hemi '.dconn.nii']);
        N=size(cifti.cdata,1);
        cifti.cdata=M;
        myciftisave(cifti,fname,N);
    case 'dtseries'        
        if(~strcmp(fname(end-11:end),'dtseries.nii'))
            fname=[fname '.dconn.nii'];
        end
        cifti=myciftiopen([d '/' hemi '.dconn.nii']);
        N=size(cifti.cdata,1);
        cifti.cdata=M;
        myciftisave(cifti,fname,N);
    case 'surf'
        if(~strcmp(fname(end-7:end),'surf.gii'))
            fname=[fname '.surf.gii'];
        end
        gii=gifti([d '/' hemi '.surf.gii']);
        gii.vertices=data.vertices;
        gii.faces=data.faces;        
        save(gii,fname);        
    otherwise
        error(['unknown type ' type]);        
end

function cifti=myciftiopen(fname)
cmd='/vols/Scratch/saad/workbench/bin_linux64/wb_command';
tmpfile=['/tmp/grotdtseries' zeropad(round(rand*9999),5) '.gii'];
unix([cmd ' -cifti-convert -to-gifti-ext ' fname ' ' tmpfile]);
cifti = gifti(tmpfile);
if(strcmp(hemi,'R'))    
    str=cifti.private.data{1}.metadata(1).value;
    str=strrep(str,'LEFT','RIGHT');
    cifti.private.data{1}.metadata(1).value=str;       
end



function myciftisave(cifti,fname)
cmd='~saad/scratch/workbench/bin_linux64/wb_command';
if(size(cifti.cdata,1)~=32942)
    % Fix Header (Saad Jbabdi)
    str=cifti.private.data{1}.metadata(1).value;
    str=strrep(str,'32942',num2str(size(cifti.cdata,1)));
    cifti.private.data{1}.metadata(1).value=str;
end
save(cifti,[fname '.gii'],'ExternalFileBinary')
unix([cmd ' -cifti-convert -from-gifti-ext ' fname '.gii ' fname]);




