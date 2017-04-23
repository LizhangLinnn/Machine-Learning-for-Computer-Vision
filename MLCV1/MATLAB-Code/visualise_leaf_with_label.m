function visualise_leaf_with_label(trees, leaves)
sizeOfLeaves = size(leaves,2);
for L = 1:size(leaves,2)

subplot(ceil(sizeOfLeaves^0.5),ceil(sizeOfLeaves^0.5),L);

tmp = trees(1).prob(leaves(1,L),:);
bar(tmp);
title(sprintf('%d th tree',L));
    

axis([0 10.5 0 1]);
end
end