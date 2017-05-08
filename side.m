
%for tr = 12:43
    
 
I = imread(sprintf('stupid.jpg'));
I = imresize(I,[256,256]);

subplot(1,3,1);
imshow(I);

cform = makecform('srgb2lab');
% Apply the colorform
lab_he = applycform(I,cform);
subplot(1,3,2);
imshow(lab_he);
ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
display(nrows);
ncols = size(ab,2);
display(ncols);
ab = reshape(ab,nrows*ncols,2);
display(ab);
nColors = 3;
[cluster_idx cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',3);
pixel_labels = reshape(cluster_idx,nrows,ncols);
imshow(pixel_labels,[]);
       
segmented_images = cell(1,3);
% Create RGB label using pixel_labels
rgb_label = repmat(pixel_labels,[1,1,3]);

for k = 1:nColors
    colors = I;
    colors(rgb_label ~= k) = 0;
    segmented_images{k} = colors;
end
                                  
figure, subplot(3,1,1);imshow(segmented_images{1});title('Cluster 1'); subplot(3,1,2);imshow(segmented_images{2});title('Cluster 2');
subplot(3,1,3);imshow(segmented_images{3});title('Cluster 3');
set(gcf, 'Position', get(0,'Screensize'));
   
% Feature Extraction
x = inputdlg('Enter the cluster no. containing the ROI only:');
i = str2double(x);
% Extract the features from the segmented image
seg_img = segmented_images{i};
figure
imshow(seg_img);
imwrite(seg_img,sprintf('stupid.jpg'));

%end