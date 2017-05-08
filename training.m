
A = zeros(43,13);
for y = 1:43
    y
    q{y} = imread(sprintf('%d.jpg',y));
    orig=q{y};
% Create the Gray Level Cooccurance Matrices (GLCMs)
img = rgb2gray(orig);
glcms = graycomatrix(img);



% Derive Statistics from GLCM
stats = graycoprops(glcms,'Contrast Correlation Energy Homogeneity');
Contrast = stats.Contrast;
Correlation = stats.Correlation;
Energy = stats.Energy;
Homogeneity = stats.Homogeneity;
Mean = mean2(orig);
Standard_Deviation = std2(orig);
Entropy = entropy(orig);
RMS = mean2(rms(orig));
%Skewness = skewness(img)
Variance = mean2(var(double(orig)));
a = sum(double(orig(:)));
Smoothness = 1-(1/(1+a));
Kurtosis = kurtosis(double(orig(:)));
Skewness = skewness(double(orig(:)));
% Inverse Difference Movement
m = size(orig,1);
n = size(orig,2);
in_diff = 0;
for i = 1:m
    for j = 1:n
        temp = orig(i,j)./(1+(i-j).^2);
        in_diff = in_diff+temp;
    end
end
IDM = double(in_diff);
   
 [center , radii] = imfindcircles(img, [3 4], 'ObjectPolarity', 'dark','sensitivity',0.93,'Method','twostage');

 headC = radii;
 headC(headC==0) = [];
 [head x]= size(headC);
 h = viscircles(center,radii,'EdgeColor','b');

 A(y,:) = [Contrast,Correlation,Energy,Homogeneity, Mean, Standard_Deviation, Entropy, RMS, Variance, Smoothness, Kurtosis, Skewness, IDM];

end

B = zeros(1,43);
for g = 1:15
    B(1,g) = 0;
end
   for g = 16:30
    B(1,g) = 1;
   end
   for g = 31:43
    B(1,g) = 2;
   end
  
    save airplane A B
    