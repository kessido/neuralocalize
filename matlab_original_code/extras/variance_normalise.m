function yn = variance_normalise(y)
% yn = variance_normalise(y)
% y is TxN

% addpath ~steve/matlab/groupPCA/
yn=y;
[uu,ss,vv]=ss_svds(y,30);
vv(abs(vv)<2.3*std(vv(:)))=0; 
stddevs=max(std(yn-uu*ss*vv'),0.001);
yn=yn./repmat(stddevs,size(yn,1),1);
        
