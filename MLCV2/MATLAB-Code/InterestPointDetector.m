function coords = InterestPointDetector(img, t_percent, ANMS,radius)
%%
%Q1 Automatic
%2a) Harris Point Detector
maxheight = size(img, 1);
maxwidth = size(img, 2);

% ================= convert colour scale to gray scale =================
if (size(size(img),2) == 3)
    img = rgb2gray(img);
end
img = im2double(img);

%% ======== find interest points based on the type of detector used ========

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% see http://db.cs.duke.edu/courses/cps196.1/spring04/handouts/Image%20Processing.pdf
% about why using a mask is better for generalising the approach of getting
% image derivatives
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ================= obtaining image derivatives ==================
dx = [-1 0 1;-1 0 1;-1 0 1]; % Derivative masks
dy = dx'; 
Ix = conv2(img, dx, 'same'); % Approximation of Image derivatives
Iy = conv2(img, dy, 'same');   

% ====================== gaussian weighting ======================
g = [0.03 0.105 0.222 0.286 0.222 0.105 0.03];
fx = repmat(g, size(g,2), 1);
fy = fx';
g = fx.*fy;
Ix2 = conv2(Ix.^2, g, 'same'); % Smoothed squared image derivatives
Iy2 = conv2(Iy.^2, g, 'same');
Ixy = conv2(Ix.*Iy, g, 'same');

% ================ auto-correlation matrix ===============
% ================ | Ix2 Ixy | ===========================
% ================ | Ixy Iy2 | ===========================


% ===== eigenvalues of the auto-correlation matrix ======= 
lambda1 = 0.5*(Ix2+Iy2)+sqrt(Ixy.^2+(0.5*(Ix2-Iy2)).^2);
lambda2 = 0.5*(Ix2+Iy2)-sqrt(Ixy.^2+(0.5*(Ix2-Iy2)).^2);

% ================ Harris feature function ===============
alpha = 0.06; % by convention
% cim = abs(lambda1.*lambda2-alpha*(lambda1+lambda2).^2); 
k = 0.04; cim = (Ix2.*Iy2 - Ixy.^2) - k*(Ix2 + Iy2).^2;

% ================ extraing local maxima =================
if ANMS % adaptive non-maximal suppression method
    % Simply look for local maxima in the interest function
    % can lead to an uneven distribution of feature points across 
    % the image, e.g., points will be denser in regions of higher 
    % contrast. So we extract local maxima by performing a grey  
    % scale morphological dilation and then finding points in the  
    % corner strength image that match the dilated image and are  
    % also greater than the threshold.
    
    
    sze = 2*radius+1;                   % Size of mask.
    mx = ordfilt2(cim,sze^2,ones(sze)); % the sum of response value of all neighbours

    % set threshold as % of the maximum value
    threshold = t_percent*max(max(mx));
    
    cim_ones = (cim==mx)&(cim>threshold);
    cim = cim_ones.*cim;
    % sort auto-correlation values in ascending order
    [r,c,value] = find(cim);       % Find row,col coords.
    [value,idx] = sort(value,'descend');
    coords = [r(idx),c(idx)];

else % directly extract local maxima
    threshold = t_percent*max(max(cim));
    cim = (cim>threshold);
    [r,c,value] = find(cim);         
    [value,idx] = sort(value,'descend');
    coords = [r(idx),c(idx)];
end

% code from below removes all the corners that are within 32 pixels 
% from the edges of the image, this is done so that 
% in Q1,2b patches of 32x32 can be obtained with meaningful data
index = [];
for i = 1:size(coords, 1)
    if~(coords(i,1)>32 && coords(i,2)>32 && coords(i,1)<maxheight-32 && coords(i,2)<maxwidth-32)
        index(end+1) = i;
    end
end
coords(index,:)=[];

end