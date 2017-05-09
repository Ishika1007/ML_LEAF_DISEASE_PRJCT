Zr = imread('1.jpg');
I = rgb2gray(Zr);
imshow(I);

% K = wiener2(I,[5,5]);
% figure
% imshow(K);
% R = im2bw(K);
% imshow(R);

% Iblur1 = imgaussfilt(I,2);
% Iblur2 = imgaussfilt(I,4);
% Iblur3 = imgaussfilt(I,8);
% 
% Beew = edge(Iblur1,'Canny');
% figure
% imshow(Beew);
% 
% BW = im2bw(Iblur1);
% figure
% imshow(BW);
% 
% BW2=imfill(BW,6,'holes');
% figure
% imshow(BW2);

mask = zeros(size(I));
mask(25:end-25,25:end-25) = 1;
%mask(:,:) = 1;
figure
imshow(mask)
title('Initial Contour Location')
bwi = activecontour(I,mask,300);
figure
imshow(bwi)
title('Segmented Image')

Bun = imfill(bwi,6,'holes');
figure
imshow(Bun);

Waterfall = zeros(size(Zr));
Waterfall(:,:,1) = Zr(:,:,1).*uint8(Bun);
Waterfall(:,:,2) = Zr(:,:,2).*uint8(Bun);
Waterfall(:,:,3) = Zr(:,:,3).*uint8(Bun);


figure 
imshow(uint8(Waterfall))
imwrite(uint8(Waterfall),sprintf('tootoo.jpg'));