function [] = myvisualise(pt, img, mytitle)
width = 3;     % Width in inches
height = 3;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 18;      % Fontsize
lw = 1.5;      % LineWidth
msz = 4;       % MarkerSize

imagesc(img), axis image, colormap(gray), hold on
pos = get(gcf, 'Position');
set(gcf, 'Position', [pos(1) pos(2) width*100, height*100]); % Set size
set(gca, 'FontSize', fsz, 'LineWidth', alw);

if size(pt,2) == 2 % only display coordinates
    plot(pt(:,2),pt(:,1),'ro','LineWidth',lw,'MarkerSize',msz); 
    title(mytitle);
elseif size(pt,2) > 2 % display scaled points
    for i=1:size(pt,1)
        rectangle('Position',...
            [pt(i,2)-pt(i,3),pt(i,1)-pt(i,3),4*pt(i,3),4*pt(i,3)],...
            'Curvature',[1,1],...
            'EdgeColor','r',...
            'LineWidth',1.5);
    end
    title(mytitle);
end

end