%% **Dynamic PET images denoising with MRI information: A graph wavelet based method*%%

clear
close all

%% Noisy PET data (Abnormal)
load BrainWeb_CleanAbnormalPET.mat
load BrainWeb_NoisyAbnormalPET.mat

frame_start = 1;
zpos = frame_start:24;
x = double(ground(:,:,zpos));
y = double(DynamicPET_mlem(:,:,zpos));
N = 175;
FrameNum  = 12;
recon = x(:,:,12) ;
im = y(:,:,12);

recon = imrotate(recon,90);
im  = imrotate(im,90);
recon = recon(21:195,5:179);
im = im(21:195,5:179);

%% Noisy PET image normalization
sigma_den = 1;  
scale_range = 1;
scale_shift = (1-scale_range)/2;
maxzans = max(im(:));
minzans = min(im(:));
im = (im-minzans)/(maxzans-minzans);
sigma_den = sigma_den/(maxzans-minzans);
im = im*scale_range+scale_shift;
sigma_den = sigma_den*scale_range;


%% Parameter settings for Graph Construction

sigma =20;  
param.nnparam.k =11;
param.patch = 5; 
param.nnparam.sigma = sigma;



%% **************************************************%%
%% Construct Graph based on  Fusion image
%% Fusion image
load FusionImageData.mat
% figure
% imshow(mridatda3,[]);

[G, nopixels, nopatches] = gsp_patch_graph(mridatda3,param);
 L= G.L;
 
%% Graph  Wavelet Transform based on  Fusion image
fprintf('Measuring largest eigenvalue, lmax = ');
lmax=sgwt_rough_lmax(L); 
arange=[0,lmax];
fprintf('%g\n',lmax);
Nscales=4;
fprintf('Designing transform in spectral domain\n');
[g,t]=sgwt_filter_design(lmax,Nscales);
m=50; % order of polynomial approximation
fprintf('Computing Chebyshev polynomials of order %g for fast transform \n',m);

%% the First decomposition
for k=1:numel(g)
    c{k}=sgwt_cheby_coeff(g{k},m,m+1,arange);
end
fprintf('Computing forward transform\n');
wpall21=sgwt_cheby_op(im(:),L,c,arange);

%%  Reconstruct the first two scale coefficients of the first decomposition for later frames
for k=1:numel(wpall21)
    wpall22{k}=zeros(size(wpall21{k}));
end
wpall22{1}=wpall21{1};
wpall22{2}=wpall21{2};
imr22=sgwt_inverse(wpall22,L,c,arange);
imr22=reshape(imr22,size(im));


%% the second decomposition
scale22 = reshape(wpall21{2},[N ,N ]);
wpall_scale22=sgwt_cheby_op(scale22(:),L,c,arange);

for k=1:numel(wpall21)
    wpall23{k}=zeros(size(wpall21{k}));
end
wpall23{1}=wpall21{1}; 
wpall23{2} = wpall_scale22{1};
imr23=sgwt_inverse(wpall23,L,c,arange);
imr23=reshape(imr23,size(im)); 


%% Denormalization
im = (im-scale_shift)/scale_range;
im = im*(maxzans-minzans)+minzans;


imr22 = (imr22-scale_shift)/scale_range;
imr22 = imr22*(maxzans-minzans)+minzans;
imr23 = (imr23-scale_shift)/scale_range;
imr23 = imr23*(maxzans-minzans)+minzans;


%% SNR and RMSE

error1 = recon-im;
error22 =recon -  imr22;
error23 =recon -  imr23;

mse_imr22 = sum((recon(:)-imr22(:)).^2);
mse_imr23 = sum((recon(:)-imr23(:)).^2);
SNR_imr22  = -10*log10((mse_imr22)/sum(recon(:).^2));
SNR_imr23  = -10*log10((mse_imr23)/sum(recon(:).^2));
RMSE_imr22 = sqrt(mse_imr22 /(N*N));
RMSE_imr23 = sqrt(mse_imr23 /(N*N));


%% show results

figure;
imshow([recon,im,imr23],[])
fprintf('SNR is: %g  and RMSE is : %g for Frame %g \n',SNR_imr23,RMSE_imr23,FrameNum);
    


