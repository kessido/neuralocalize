function [s,BM]=open_wbfile(fname)
% [struct BM]=open_wbfile(fname)
%
% Assumes workbench is installed
%
% S Jbabdi 04/13


[type,fname]=detect_type(fname);
switch(lower(type))
    case 'gifti'
        s=gifti(fname);
    case 'cifti'
        [s BM]=myciftiopen(fname);
    otherwise
        error('unknown file type');
end

    function [t,f]=detect_type(fname)
        [~,~,e]=fileparts(fname);
        if(strcmp(e,'.gii'))
            t='gifti';f=fname;
        elseif(strcmp(e,'.nii'))
            t='cifti';f=fname;
        else
            if(exist([fname '.dtseries.nii'],'file'))
                t='cifti';f=[fname '.dtseries.nii'];
            elseif(exist([fname '.dconn.nii'],'file'))
                t='cifti';f=[fname '.dconn.nii'];
            elseif(exist([fname '.func.gii'],'file'))
                t='gifti';f=[fname '.func.gii'];
            elseif(exist([fname '.surf.gii'],'file'))
                t='gifti';f=[fname '.surf.gii'];
            elseif(exist([fname '.shape.gii'],'file'))
                t='gifti';f=[fname '.shape.gii'];
            end
        end
    end

    function [cifti BM]=myciftiopen(fname)
        cmd='wb_command';
        rng('shuffle');
        tmpfile=['/tmp/grotdtseries' zeropad(round(rand*9999),5) '.gii'];
        unix([cmd ' -cifti-convert -to-gifti-ext ' fname ' ' tmpfile]);
        cifti = gifti(tmpfile);
        BM=restructure_data(cifti.private.data{1}.metadata.value ,cifti.cdata);
        unix(['rm -f ' tmpfile ' ' tmpfile '.data']);
    end

    function BM=restructure_data(str,data)
        %%
        tree=xmltree(str);
        
        % find transformation matrix
        bmi=[];
        IJK2XYZi=[];
        XYZ2IJKi=[];
        for ctr=1:length(tree)
            try NAME=get(tree,ctr,'name');catch, NAME=[];end
            if strcmp(NAME,'BrainModel')
                bmi=[bmi ctr];
            elseif strcmp(NAME,'TransformationMatrixVoxelIndicesIJKtoXYZ')
                IJK2XYZi=[IJK2XYZi ctr];
            elseif strcmp(NAME,'TransformationMatrixVoxelIndicesXYZtoIJK')
                XYZ2IJKi=[XYZ2IJKi ctr];
            end
        end
        if length(IJK2XYZi)>2 | length(XYZ2IJKi)>2
            error('Too many transformation matrices found, can''t with this cope')
        end
        %%
        BM=cell(length(bmi),1);
        for bm=1:length(bmi)
            %%
            attr=get(tree,bmi(bm),'attributes');
            
            for b=1:length(attr)
                KEY=attr{b}.key;
                VAL=attr{b}.val;
                if ~isstruct(BM{bm})
                    BM{bm}=struct(KEY,VAL);
                else
                    BM{bm}=setfield(BM{bm},KEY,VAL);
                end
            end
            %%
            
            try ch=get(tree,bmi(bm),'contents');catch, ch=[];end
            while length(ch)>0
                cc = ch(1);
                ch(1)=[];
                try nch=get(tree,cc,'contents');catch, nch=[];end
                ch=union(ch,nch);
                if strcmp(get(tree,cc,'type'),'element')
                    if strcmp(get(tree,cc,'name'),'VertexIndices')
                        try  ccc=get(tree,cc,'contents');catch, ccc=[];end
                        if ~isempty(ccc)
                            if strcmp(get(tree,ccc,'type'),'chardata')
                                didx=str2num(get(tree,ccc,'value'))+1;
                                if length(didx)~=str2num(BM{bm}.IndexCount)
                                    error('Data dimension does not agree with index count!')
                                end
                                BM{bm}=setfield(BM{bm},'SurfaceIndices',didx);
                                midx=str2num(BM{bm}.IndexOffset)+1:str2num(BM{bm}.IndexOffset)+str2num(BM{bm}.IndexCount);
                                BM{bm}=setfield(BM{bm},'DataIndices',midx);
                                BM{bm}=setfield(BM{bm},'Data',data(midx,:));
                            end
                        end
                    elseif strcmp(get(tree,cc,'name'),'VoxelIndicesIJK')
                        try  ccc=get(tree,cc,'contents');catch, ccc=[];end
                        if ~isempty(ccc)
                            if strcmp(get(tree,ccc,'type'),'chardata')
                                didx=str2num(get(tree,ccc,'value'))+1;
                                if length(didx)/3==str2num(BM{bm}.IndexCount)
                                    didx=reshape(didx,[3,length(didx)/3])';
                                else
                                    error('Data dimension does not agree with index count!')
                                end
                                BM{bm}=setfield(BM{bm},'VolumeIndicesIJK',didx);
                                midx=str2num(BM{bm}.IndexOffset)+1:str2num(BM{bm}.IndexOffset)+str2num(BM{bm}.IndexCount);
                                BM{bm}=setfield(BM{bm},'DataIndices',midx');
                                BM{bm}=setfield(BM{bm},'Data',data(midx,:));
                                if length(IJK2XYZi)>0
                                    XT=struct;
                                    attr=get(tree,IJK2XYZi,'attributes');
                                    for b=1:length(attr)
                                        KEY=attr{b}.key;
                                        VAL=attr{b}.val;
                                        XT=setfield(XT,KEY,VAL);
                                    end
                                    ch=get(tree,IJK2XYZi,'contents');
                                    for c=ch';
                                        if strcmp(get(tree,c,'type'),'chardata')
                                            trx=reshape(str2num(get(tree,c,'value')),[4 4]);
                                        end
                                    end
                                    XT=setfield(XT,'TransformMatrix',trx);
                                    BM{bm}=setfield(BM{bm},'IJK2XYZ',XT);
                                end
                            end
                        end
                    elseif strcmp(get(tree,cc,'name'),'VoxelIndicesXYZ')
                        try  ccc=get(tree,cc,'contents');catch, ccc=[];end
                        if ~isempty(ccc)
                            if strcmp(get(tree,ccc,'type'),'chardata')
                                didx=str2num(get(tree,ccc,'value'));
                                if length(didx)/3==str2num(BM{bm}.IndexCount)
                                    didx=reshape(didx,[3,length(didx)/3])';
                                else
                                    error('Data dimension does not agree with index count!')
                                end
                                BM{bm}=setfield(BM{bm},'VolumeIndicesXYZ',didx);
                                midx=str2num(BM{bm}.IndexOffset)+1:str2num(BM{bm}.IndexOffset)+str2num(BM{bm}.IndexCount);
                                BM{bm}=setfield(BM{bm},'DataIndices',midx');
                                BM{bm}=setfield(BM{bm},'Data',data(midx,:));
                                if length(XYZ2IJKi)>0
                                    XT=struct;
                                    attr=get(tree,XYZ2IJKi,'attributes');
                                    for b=1:length(attr)
                                        KEY=attr{b}.key;
                                        VAL=attr{b}.val;
                                        XT=setfield(XT,KEY,VAL);
                                    end
                                    ch=get(tree,XYZ2IJKi,'contents');
                                    for c=ch';
                                        if strcmp(get(tree,c,'type'),'chardata')
                                            trx=reshape(str2num(get(tree,c,'value')),[4 4]);
                                        end
                                    end
                                    XT=setfield(XT,'TransformMatrix',trx);
                                    BM{bm}=setfield(BM{bm},'XYZ2IJK',XT);
                                end
                            end
                        end
                    end
                    
                end
            end
        end
    end
end
