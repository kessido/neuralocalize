function data = cifti_extract_data(cifti,structure,BM,All)
% data = cifti_extract_data(cifti,structure,BM,[All=0])
% if All=1, keep all surface vertices
% if All=0, remove cut area
%
% S.Jbabdi 04/2016
if(nargin<4)
    All=0;
end


if(strcmp(structure,'L'))
    if(All==0)
        data = cifti.cdata( BM{1}.DataIndices, :);
    else
        data = zeros(32492,size(cifti.cdata,2));
        data(BM{1}.SurfaceIndices,:) = cifti.cdata( BM{1}.DataIndices, :);
    end
elseif(strcmp(structure,'R'))
    if(All==0)
        data = cifti.cdata( BM{2}.DataIndices, :);
    else
        data = zeros(32492,size(cifti.cdata,2));
        data(BM{2}.SurfaceIndices,:) = cifti.cdata( BM{2}.DataIndices, :);
    end
elseif(strcmp(structure,'SC'))
    data=[];
    for i=3:length(BM)
        y = cifti.cdata( BM{i}.DataIndices, :);  
        data = [data;y];
    end    
else
    hit=0;
    for i=3:21
        if(strcmp(structure,BM{i}.BrainStructure))            
            data = cifti.cdata( BM{i}.DataIndices, :);       
            hit=1;
        end
    end
    if(~hit)
        error('structure either L, R or SC or a CIFTI_STRUCTURE_*');
    end
end
