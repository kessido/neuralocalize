function [cope,varcope,stats] = fsl_glm(x,y,c)
% [cope,varcope,stats] = fsl_glm(x,y,[c=Identity])
% stats.{t,p,z,dof}
%
% S. Jbabdi 03/14

if(nargin<3)
    c = eye(size(x,2));
end

beta = pinv(x)*y;
cope = c*beta;
r    = y-x*beta;
dof  = size(r,1)-rank(x);

sigma_sq = sum(r.^2)/dof;

varcope = diag((c*inv(x'*x)*c')) * sigma_sq;
t       = cope./sqrt(varcope);
% p       = tcdf(t,dof); % 1-sided p-value for t-stat
t(isnan(t))=0;
stats.t   = t;
% stats.p   = p;
stats.dof = dof;
% stats.z   = (2^0.5)*erfinv(1-2*betainc(dof./(dof+t.^2),dof/2,1/2)/2);
stats.ss = sigma_sq;
