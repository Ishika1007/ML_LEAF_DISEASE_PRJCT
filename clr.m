

I = imread('tootoo.jpg');
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

% Convert to grayscale if image is RGB
if ndims(seg_img) == 3
   img = rgb2gray(seg_img);
end
figure, imshow(img); title('Gray Scale Image');
                               

% Evaluate the disease affected area
black = im2bw(seg_img,graythresh(seg_img));
figure, imshow(black);title('Black & White Image');
m = size(seg_img,1);
n = size(seg_img,2);

zero_image = zeros(m,n); 

%G = imoverlay(zero_image,seg_img,[1 0 0]);

cc = bwconncomp(seg_img,6);
display(cc);
diseasedata = regionprops(cc,'basic');

display(diseasedata);

A1 = diseasedata.Area;
sprintf('Area of the disease affected region is : %g%',A1);
display(A1);

I_black = im2bw(I,graythresh(I));
figure, imshow(I_black);title('Black & White Image Original');
kk = bwconncomp(I,6);
display(kk);
leafdata = regionprops(kk,'basic');
A2 = leafdata.Area;
sprintf(' Total leaf area is : %g%',A2);
display(A2);

%Affected_Area = 1-(A1/A2);
Affected_Area = (A1/A2);
if Affected_Area < 0.1
    Affected_Area = Affected_Area+0.15;
end
sprintf('Affected Area is: %g%%',(Affected_Area*100))


glcms = graycomatrix(img);
display(glcms);



% Derive Statistics from GLCM
stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(seg_img);
Standard_Deviation = std2(seg_img);
Entropy = entropy(seg_img);
RMS = mean2(rms(seg_img));
%Skewness = skewness(img)
Variance = mean2(var(double(seg_img)));
a = sum(double(seg_img(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(seg_img(:)));
Skewness = skewness(double(seg_img(:)));
% Inverse Difference Movement
m = size(seg_img,1);
n = size(seg_img,2);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = seg_img(i,j)./(1+(i-j).^2);
        in_diff = in_diff+temp;
    end
end
IDM = double(in_diff);
    
 [center , radii] = imfindcircles(img, [3 4], 'ObjectPolarity', 'dark','sensitivity',0.93,'Method','twostage')

 headC = radii;
 headC(headC==0) = [];
 [head x]= size(headC);
 h = viscircles(center,radii,'EdgeColor','b');


feat_disease  = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM];



%%
% Load All The Features
load('airplane.mat')


% Put the test features into variable 'test'
test = feat_disease;
result = multisvm(A,B,test);
%disp(result);

% Visualize Results
if result == 0
    helpdlg(' Alternaria Macrospora ');
    disp(' Alternaria Macrospora ');
elseif result == 1
    helpdlg(' Bacterial Blight or Alternaria Macrospora ');
    disp(' Bacterial Blight or Alternaria Macrospora');
elseif result == 2
    helpdlg(' Grey Mildew ');
    disp('Grey Mildew');
elseif result == 3
    helpdlg(' Healthy Leaf ');
    disp('Healthy Leaf ');
end




