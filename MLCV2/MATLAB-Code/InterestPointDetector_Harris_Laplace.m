function points = InterestPointDetector_Harris_Laplace(img,t_percent)
    % Extract interest points using Harris-Laplace algorithm
    % INPUT
    % =====
    % img       : graylevel image
    % t_percent : threshold as a percentage of the maximum quality
    %             default should be 1%
    %
    %
    % OUTPUT
    % ======
    % points    : interest points extracted (sorted by reliability in 
    %             descending order)
    
    
    % Get image paramaters
    img         = double(img(:,:,1));
    img_height  = size(img,1);
    img_width   = size(img,2);

    % Set scale parameters (optimal values verified by D.Lowe)
    sigma_begin = 1.5;
    sigma_step  = 2^(1/4);
    sigma_nb    = 8;
    sigma_array = (sigma_step.^(0:sigma_nb-1))*sigma_begin;


    % Part 1 : Harris
    % ====================================================================
    harris_pts = zeros(0,4);
    for i=1:sigma_nb

        % scale (standard deviation)
        s_I = sigma_array(i);   % intgration scale
        s_D = 0.7*s_I;          % derivative scale %0.7

        % derivative mask
        x = -round(3*s_D):round(3*s_D);
        dx = x .* exp(-x.*x/(2*s_D*s_D)) ./ (s_D*s_D*s_D*sqrt(2*pi));
        dy = dx';

        % image derivatives
        Ix = conv2(img, dx, 'same');
        Iy = conv2(img, dy, 'same');
        g   = fspecial('gaussian',max(1,fix(6*s_I+1)), s_I);
        
        
        % auto-correlation matrix
        Ix2 = conv2(Ix.^2, g,  'same');
        Iy2 = conv2(Iy.^2, g,  'same');
        Ixy = conv2(Ix.*Iy, g, 'same');

        % interest point response (Original Harris measure)
        k = 0.06; cim = (Ix2.*Iy2 - Ixy.^2) - k*(Ix2 + Iy2).^2;	

        
        % find local maxima on neighborgood
        radius = 3;
        [r,c,max_local] = findLocalMaximum(cim,radius);

        % set threshold as a percentage of the maximum value
        t = t_percent*max(max_local(:));

        % find local maxima greater than threshold
        [r,c,~] = find(max_local>=t);
        value = max_local((c-1)*img_height+r);
        
        
        % Remove all the corners that are within 32*scale pixels from
        % the edges of the image, this is done so that in Q1,2b patches  
        % of (32*scale/1.5)x(32*scale/1.5) can be obtained with meaningful data
        patch_d = 32*s_I/1.5;
        index = [];
        for j = 1:size(r, 1)
            if~(r(j)>patch_d && c(j)>patch_d && r(j)<img_height-patch_d && c(j)<img_width-patch_d)
                index(end+1) = j;
            end
        end
        r(index,:)=[];
        c(index,:)=[];
        value(index,:) = [];
        
        % build interest points
        n = size(r,1);
        harris_pts(end+1:end+n,:) = [r,c,repmat(i,[n,1]),value];
        
        
    end
    
        % sort auto-correlation values in descend order
        [value,idx] = sort(harris_pts(:,4),'descend');
        harris_pts = harris_pts(idx,:);


    % Part 2 : Laplace
    % ====================================================================
    % compute scale-normalized laplacian operator
    laplace_snlo = zeros(img_height,img_width,sigma_nb);
    for i=1:sigma_nb
        s_L = sigma_array(i);   % scale
        laplace_snlo(:,:,i) = s_L*s_L*imfilter(img,fspecial('log', floor(6*s_L+1), s_L),'replicate');
    end
    % verify for each of the initial points whether the LoG attains a maximum at the scale of the point
    n   = size(harris_pts,1);
    cpt = 0;
    points = zeros(n,4);
    for i=1:n
        r = harris_pts(i,1);
        c = harris_pts(i,2);
        s = harris_pts(i,3);
        val = laplace_snlo(r,c,s);
        if s>1 && s<sigma_nb
            if val>laplace_snlo(r,c,s-1) && val>laplace_snlo(r,c,s+1)
                cpt = cpt+1;
                points(cpt,:) = harris_pts(i,:);
            end
        elseif s==1
            if val>laplace_snlo(r,c,2)
                cpt = cpt+1;
                points(cpt,:) = harris_pts(i,:);
            end
        elseif s==sigma_nb
            if val>laplace_snlo(r,c,s-1)
                cpt = cpt+1;
                points(cpt,:) = harris_pts(i,:);
            end
        end
    end
    points(cpt+1:end,:) = [];
    
    
    
    % SET SCALE TO SIGMA FOR DISPLAY
    points(:,3) = sigma_array(points(:,3));
end


function [row,col,max_local] = findLocalMaximum(val,radius)
        mask  = fspecial('disk',radius)>0;
        nb    = sum(mask(:));
        highest          = ordfilt2(val, nb, mask);
        second_highest   = ordfilt2(val, nb-1, mask);
        index            = highest==val & highest~=second_highest;
        max_local        = zeros(size(val));
        max_local(index) = val(index);
        [row,col]        = find(index==1);
end