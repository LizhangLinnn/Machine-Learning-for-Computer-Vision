function [points1_manual,points2_manual] = getManualIPs(img1, img2, num)
    figure;
    subplot(121);
    imshow(img1);
    points1_manual = fliplr(ginput(num));
    subplot(122);
    imshow(img2);
    points2_manual = fliplr(ginput(num));
end