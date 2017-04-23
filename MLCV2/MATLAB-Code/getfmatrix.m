function f = getfmatrix(v1, v2)
%3B) obtain fundamental matrix
n = size(v1, 1);
A = [v1(:,1).*v2(:,1) v1(:,1).*v2(:,2) v1(:,1) ...
      v1(:,2).*v2(:,1) v1(:,2).*v2(:,2) v1(:,2) ...
      v2(:,1) v2(:,2) ones(n,1)];
F = zeros(3,3);
[UA,SA,VA]=svd(A);
F(:) = VA(:,9);
[U,S,V] = svd(F);
f = U(:,1:2)*S(1:2,1:2)*V(:,1:2)';
end