function h = gethmatrix(v1, v2)
%3A) obtain homography matrix
n = size(v1, 1);
% Solve equations using SVD
x1 = v1(:,1)';
y1 = v1(:,2)';
x2 = v2(:,1)';
y2 = v2(:,2)';
rowofzero = zeros(3, n);
rowofxy = -[x1; y1; ones(1,n)];
hx = [rowofxy; rowofzero; x2.*x1; x2.*y1; x2];
hy = [rowofzero; rowofxy; y2.*x1; y2.*y1; y2];
h = [hx hy];
[U, D, V] = svd(h');
h = (reshape(V(:,9), 3, 3)).';
end
