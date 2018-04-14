function [P,Q,R1,R2,S]=reord2(A,visu,comps)
% [P,Q,R1,R2,S]=reord2(A,[,visu=0,comps=3])
% Spectral reordering
%
% S Jbabdi 2012


if(nargin<2);visu=0;end
if(nargin<3);comps=3;end

%%%%%% Spectral Ordering %%%%%%

% Row and column normalise
ti = diag(sqrt(1./sum(A,1)));
to = diag(sqrt(1./sum(A,2)));
W  = to*A*ti;

% Do the job
[U,S,V] = svd(W);
S       = diag(S);

sel = 2:comps+1;
P = to*U(:,sel) .* repmat(S(sel)',size(U,1),1);
Q = ti*V(:,sel) .* repmat(S(sel)',size(V,1),1);

[~,R1]=sort(P);
[~,R2]=sort(Q);

% calculate cost function
% cf = zeros(comps,1);
% for k=1:comps
%     for j=1:k    
%         cf(k) = cf(k) + sum(sum(A .*(repmat(P(:,j),1,size(A,2))-repmat(Q(:,j)',size(A,1),1)).^2));
%     end
% end




if(visu==1)
    [~,vals] = local_eigs([P;Q],round(sum(size(A))/10));
    cf = mean(vals,1)';


    figure
    
    subplot(2,4,1);hold on
    plot(P(:,1),1-.01,'b.');
    plot(Q(:,1),1+.01,'ro');
    axis([min([P(:,1);Q(:,1)]),max([P(:,1);Q(:,1)]),1-.5,1+.5])
    grid on
    
    subplot(2,4,2);hold on
    plot(P(:,1),P(:,2),'b.');
    plot(Q(:,1),Q(:,2),'ro');
    axis equal
    grid on
    
    subplot(2,4,3);hold on
    plot3(P(:,1),P(:,2),P(:,3),'b.');
    plot3(Q(:,1),Q(:,2),Q(:,3),'ro');
    axis equal
    grid on
    
    subplot(2,4,5)
    imagesc(A);title('un-ordered');
    
    subplot(2,4,6)
    imagesc(A(R1(:,1),R2(:,1)));title('un-normalised');
    
    subplot(2,4,7)
    imagesc(W(R1(:,1),R2(:,1)));title('normalised');

    subplot(2,4,8),hold on
%     plot(rescale(S(2:comps+1),0,1),'.-');
    plot(cf,'.-r');
    grid on
    %     for ii=1:3
%         xx=P(R1(:,ii),ii);yy=Q(R2(:,ii),ii);    
%         subplot(3,3,ii)
%         imagesc(A(R1(:,ii),R2(:,ii)));
%         line([0 length(yy)],find(gradient(sign(xx))>.5,1)*ones(1,2),'color','k','linewidth',2);
%         line(find(gradient(sign(yy))>.5,1)*ones(1,2),[0 length(xx)],'color','k','linewidth',2);
% 
%         title(['Reordered ' num2str(ii)],'FontWeight','Bold')
%     end
%     
%     X=P;
%     subplot(3,3,4)
%     plot(X(:,1),X(:,2),'.');axis equal 
%     xlabel('u1');ylabel('u2');
%     subplot(3,3,5);
%     plot(X(:,1),X(:,3),'.');axis equal 
%     xlabel('u1');ylabel('u3');
%     subplot(3,3,6);
%     plot(X(:,2),X(:,3),'.');axis equal 
%     xlabel('u2');ylabel('u3');
%     X=Q;
%     subplot(3,3,7)
%     plot(X(:,1),X(:,2),'.');axis equal 
%     xlabel('v1');ylabel('v2');
%     subplot(3,3,8);
%     plot(X(:,1),X(:,3),'.');axis equal 
%     xlabel('v1');ylabel('v3');
%     subplot(3,3,9);
%     plot(X(:,2),X(:,3),'.');axis equal 
%     xlabel('v2');ylabel('v3');
   
    set(gcf,'position',[191         388        1142         399]);


%     figure
%     subplot(1,3,1)
%     plot3(P(:,1),P(:,2),P(:,3),'.');axis equal 
%     subplot(1,3,2)
%     plot3(Q(:,1),Q(:,2),Q(:,3),'.');axis equal 
%     subplot(1,3,3)
%     plot(S(2:10),'-o');
    
%     set(gcf,'position',[187          14        1149         295]);
    
end    
    


function [dim,vals] = local_eigs(data,k)
% [dim,vals] = local_eigs(data,k)
% local PCA - find dimensionality of embedding manifold
%
% data is NxP 
% k defines k-nearest neighbours
%
% S. Jbabdi 12/14

N = size(data,1);

vals = zeros(N,size(data,2));
for i=1:N
    [~,j]     = sort(quickdist(data(i,:),data),'ascend');
    y         = data(j(1:k),:);
    e         = eig(cov(y));
    vals(i,:) = sort(e,'descend');
end
vals = vals ./ repmat(max(vals,[],2),1,size(vals,2));


[~,dim]=max(abs(gradient(mean(vals))));
dim=dim+1;

