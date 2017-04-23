function label = testTrees(data,tree)
% Slow version - pass data point one-by-one
[N, D]= size(data);
cc = [];
for T = 1:length(tree)
    for m = 1:size(data,1);
        idx = 1;
        
        while ~tree(T).node(idx).isleaf
            switch tree(T).node(idx).splitfun
                case 1
                    t = tree(T).node(idx).model.t;
                    r = tree(T).node(idx).model.r;
                    % Decision
                    if data(m,r) < t % Pass data to left node
                        idx = idx*2;
                    else
                        idx = idx*2+1; % and to right
                    end
                case 2
                    w = tree(T).node(idx).model.w;
                    r1 = tree(T).node(idx).model.r1;
                    r2 = tree(T).node(idx).model.r2;
                    % Decision
                    % if [data(m, [r1 r2]), ones(N, 1)]*w < 0 % Pass data to left node
                    if (double(data(m,r1))*w(1) + double(data(m,r2))*w(2)+ ones(N,1)*w(3)< 0)
                        idx = idx*2;
                    else
                        idx = idx*2+1; % and to right
                    end
                case 3
                    r1= tree(T).node(idx).model.r1;
                    r2= tree(T).node(idx).model.r2;
                    w= tree(T).node(idx).model.w;
                    t1= tree(T).node(idx).model.t1;
                    t2= tree(T).node(idx).model.t2;
                    %phi= [data(m, r1).*data(m, r2), data(m,r1).^2, data(m,r2).^2, data(m, r1), data(m, r2), ones(N, 1)];
                    phi= [double(data(m, r1)).*double(data(m, r2)), double(data(m,r1)).^2, double(data(m,r2)).^2, double(data(m, r1)), double(data(m, r2)), ones(N, 1)];
                    mv= phi*w;
                    % Decision
                    if mv<t2 && mv>t1 % Pass data to left node
                        idx = idx*2;
                    else
                        idx = idx*2+1; % and to right
                    end
                case 4
                    x= tree(T).node(idx).model.x;
                    t= tree(T).node(idx).model.t;
                    dsts= pdist2(data(m, 1:end-1), x);
                    if dsts < t
                        idx = idx*2;
                    else
                        idx = idx*2+1; % and to right
                    end
                otherwise
                    disp('unknown splifunction')
            end
        end
        
        leaf_idx = tree(T).node(idx).leaf_idx;
        
        if ~isempty(tree(T).leaf(leaf_idx))
            p(m,:,T) = tree(T).leaf(leaf_idx).prob;
            label(m,T) = tree(T).leaf(leaf_idx).label;
            
%             if isfield(tree(T).leaf(leaf_idx),'cc') % for clustering forest
%                 cc(m,:,T) = tree(T).leaf(leaf_idx).cc;
%             end
        end
    end
end

end

