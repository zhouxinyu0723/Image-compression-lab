%% get paths of images
clear
folder_path = strcat('foreman20_40_RGB');
image_paths=image_dataset(folder_path);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%     Motion encoding
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% prepare for for loop
scales = [0.07, 0.2, 0.4, 0.8, 1.0,1.5, 2, 3, 4, 4.5];%[0.07, 0.2, 0.4, 0.8, 1.0, 1.5, 2, 3, 4, 4.5];
bitPerPixel = zeros(length(scales),1);
PSNR = zeros(length(scales),1);
%% for loop
for scaleIdx = 1 : numel(scales)
    qScale = scales(scaleIdx);
    qScale_str=sprintf('%.2f',qScale);
    rec_path = strcat(folder_path,'ch5_rec_',qScale_str,'/');  
    mkdir(rec_path);
    byte_count = 0;
    PSNR_each_image = zeros(length(image_paths),1);
    images_rec = cell(length(image_paths),1);
    %% get the firt and the second images
    I = double(imread(image_paths{1}));
    P1 = double(imread(image_paths{2}));
    %% 1.1 first image intra huffman code generation
    k_I        = IntraEncode8x8(I, qScale);
    bias = 1001;
    pmf_I = stats_marg(k_I+bias, 1:5010);
    [ BinaryTreeQuantize_I, HuffCodeQuantize_I, BinCodeQuantize_I, CodelengthsQuantize_I] = buildHuffman( pmf_I );
    %% 1.2 first image encode
    bytestream_I = enc_huffman_new(k_I+bias , BinCodeQuantize_I, CodelengthsQuantize_I); %positive and not zero
    byte_count = byte_count + numel(bytestream_I);
    %% 1.3 first image decode
    k_I_rec = dec_huffman_new (bytestream_I, BinaryTreeQuantize_I, length(k_I));
    Fn1 = IntraDecode8x8(k_I_rec-bias, size(I),qScale);
    imwrite(uint8(Fn1),strcat(rec_path,int2str(1),'.bmp'));
    PSNR_each_image(1)=calcPSNR(I, Fn1);
    %% 2.1 motion prediction (motion vectors) huffman code generation
    Fn1_ycc = ictRGB2YCbCr(Fn1);
    P1_ycc = ictRGB2YCbCr(P1);
    MV = SSD8x8(Fn1_ycc(:,:,1), P1_ycc(:,:,1));
    size_MV = size(MV);
    MV_flat = reshape(MV,[],1);
    pmf_MV = stats_marg(MV_flat, 1:81);
    [ BinaryTreeQuantize_MV, HuffCodeQuantize_MV, BinCodeQuantize_MV, CodelengthsQuantize_MV] = buildHuffman( pmf_MV );
    %% 2.2 motion prediction (motion vectors) encode
    bytestream_MV = enc_huffman_new(MV_flat, BinCodeQuantize_MV, CodelengthsQuantize_MV);
    byte_count = byte_count + numel(bytestream_MV);
    %% 2.3 motion prediction (motion vectors) decode
    MV_flat_rec = dec_huffman_new (bytestream_MV, BinaryTreeQuantize_MV, length(MV_flat));
    MV_rec = reshape(MV_flat_rec,size_MV);
    mcp = SSD_rec8x8(Fn1, MV_rec);
    %% 3.1 Residual huffman code generation
    Residual   = P1 - mcp;
    k_Residual = IntraEncode8x8(Residual, qScale);
    bias;
    pmf_Residual = stats_marg(k_Residual+bias, 1:5010);
    [ BinaryTreeQuantize_Residual, HuffCodeQuantize_Residual, BinCodeQuantize_Residual, CodelengthsQuantize_Residual] = buildHuffman( pmf_Residual );
    %% 3.2 Residual encode
    bytestream_Residual = enc_huffman_new(k_Residual+bias , BinCodeQuantize_Residual, CodelengthsQuantize_Residual);
    byte_count = byte_count + numel(bytestream_Residual);
    %% 3.3 Residual decode
    k_Residual_rec = dec_huffman_new (bytestream_Residual, BinaryTreeQuantize_Residual, length(k_Residual));
    Residual_rec = IntraDecode8x8(k_Residual_rec-bias, size(Residual),qScale);   
    %% 4 Residual + motion
    Fn = mcp + Residual_rec;
    imwrite(uint8(Fn),strcat(rec_path,int2str(2),'.bmp'));
    PSNR_each_image(2)=calcPSNR(P1, Fn);
    for i = 3:length(image_paths)
        P = double(imread(image_paths{i}));
        %% 5.1 motion prediction calculation continuous
        Fn_ycc = ictRGB2YCbCr(Fn);
        P_ycc = ictRGB2YCbCr(P);
        MV = SSD8x8(Fn_ycc(:,:,1), P_ycc(:,:,1));
        size_MV = size(MV);
        MV_flat = reshape(MV,[],1);
        %% 5.2 motion prediction (motion vectors) encode continuous
        bytestream_MV = enc_huffman_new(MV_flat, BinCodeQuantize_MV, CodelengthsQuantize_MV);
        byte_count = byte_count + numel(bytestream_MV);
        %% 5.3 motion prediction (motion vectors) decode continuous
        MV_flat_rec = dec_huffman_new (bytestream_MV, BinaryTreeQuantize_MV, length(MV_flat));
        MV_rec = reshape(MV_flat_rec,size_MV);
        mcp        = SSD_rec8x8(Fn, MV_rec);
        %% 6.1 Residual calculation continuous
        Residual   = P - mcp;
        k_Residual = IntraEncode8x8(Residual, qScale);
        %% 6.2 Residual encode continous
        bytestream_Residual = enc_huffman_new(k_Residual+bias , BinCodeQuantize_Residual, CodelengthsQuantize_Residual);
        byte_count = byte_count + numel(bytestream_Residual);
        %% 6.3 Residual decode continous
        k_Residual_rec = dec_huffman_new (bytestream_Residual, BinaryTreeQuantize_Residual, length(k_Residual));
        Residual_rec = IntraDecode8x8(k_Residual_rec-bias, size(Residual),qScale);   
        %% 7 Residual + motion continous
        Fn = mcp + Residual_rec;
        imwrite(uint8(Fn),strcat(rec_path,int2str(i),'.bmp'));
        pppppp = calcPSNR(P, Fn)
        PSNR_each_image(i)=pppppp;
    end
    bitPerPixel(scaleIdx) = (byte_count*8) / (numel(I)/3) /length(image_paths);
    PSNR(scaleIdx) = mean(PSNR_each_image);  
    fprintf('QP: %.2f bit-rate: %.2f bits/pixel PSNR: %.2fdB\n', qScale, bitPerPixel(scaleIdx), PSNR(scaleIdx))
end
%% save information
info_path = strcat(folder_path,'_ch5_info_');  
mkdir(info_path);
save(strcat(info_path,'/scales.mat'),'scales');
save(strcat(info_path,'/PSNR.mat'),'PSNR');
save(strcat(info_path,'/bitPerPixel.mat'),'bitPerPixel');
%https://de.mathworks.com/matlabcentral/answers/154360-imwrite-function-and-permission-error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   individual encoding
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% prepare for for loop
scales_ind = [0.15,0.3,0.7,1.0,1.5,3,5,7,10];
bitPerPixel_ind = zeros(length(scales_ind),1);
PSNR_ind = zeros(length(scales_ind),1);
for scaleIdx = 1 : numel(scales_ind)
    I = double(imread('lena_small.tif'));%image_paths{1}
    qScale   = scales_ind(scaleIdx);
    k_train  = IntraEncode8x8(I, qScale);  
    %% use pmf of k_small to build and train huffman table
    % insert your code here
    bias = 1001;
    pmf = stats_marg(k_train+bias, 1:5010);
    [ BinaryTreeQuantize, HuffCodeQuantize, BinCodeQuantize, CodelengthsQuantize] = buildHuffman( pmf ); 
    bitPerPixel_each_image=zeros(length(image_paths),1);
    PSNR_each_image=zeros(length(image_paths),1);
    for i = 1:length(image_paths)
        %% use trained table to encode k to get the bytestream
        % insert your code here
        I = double(imread(image_paths{i}));
        k_I  = IntraEncode8x8(I, qScale);  
        bytestream = enc_huffman_new(k_I+bias , BinCodeQuantize, CodelengthsQuantize);
        bitPerPixel_each_image(i) = (numel(bytestream)*8) / (numel(I)/3);
        %% image reconstruction
        k_rec = dec_huffman_new (bytestream, BinaryTreeQuantize, length(k_I));
        I_rec = IntraDecode8x8(k_rec-bias, size(I),qScale);
        pppppp = calcPSNR(I, uint8(I_rec))
        PSNR_each_image(i) = pppppp;
    end
    bitPerPixel_ind(scaleIdx) = mean(bitPerPixel_each_image);
    PSNR_ind(scaleIdx) = mean(PSNR_each_image);
    fprintf('QP: %.2f bit-rate: %.2f bits/pixel PSNR: %.2fdB\n', qScale, bitPerPixel_ind(scaleIdx), PSNR_ind(scaleIdx))
end
%%
info_path = strcat(folder_path,'_ch4_info');  
mkdir(info_path);
save(strcat(info_path,'/scales_ind.mat'),'scales_ind');
save(strcat(info_path,'/PSNR_ind.mat'),'PSNR_ind');
save(strcat(info_path,'/bitPerPixel_ind.mat'),'bitPerPixel_ind');
figure;
motion_plot = plot(bitPerPixel,PSNR,'-*');M1 = 'Video Codec';
hold on
ind_plot = plot(bitPerPixel_ind,PSNR_ind,'-*');M2 = 'Still image Codec';
title('My Answer');
xlabel('bpp');
ylabel('PSNR [dB]');
legend(M1, M2);
xlim([0.2 4]);
hold off
%% sub functions
function pmf = stats_marg(image, range)
    pmf = hist(squeeze(reshape(image,[],1,1)),range);
    pmf = double(pmf)/sum(pmf,'all');
end
function rgb = ictYCbCr2RGB(yuv)
% Input         : yuv (Original YCbCr image)
% Output        : rgb (RGB Image after transformation)
% YOUR CODE HERE
    size_yuv = size(yuv);
    tran = [1 0 1.402;1 -0.344 -0.714;1 1.772 0];
    rgb=reshape((tran*squeeze(reshape(yuv,[],3))')',size_yuv);
end
function MSE = calcMSE(Image, recImage)
% Input         : Image    (Original Image)
%                 recImage (Reconstructed Image)
% Output        : MSE      (Mean Squared Error)
% YOUR CODE HERE
    [l,w,d] = size(Image);
    MSE = 1/d/l/w*sum((double(Image)-double(recImage)).^2,'all');
end
function yuv = ictRGB2YCbCr(rgb)
% Input         : rgb (Original RGB Image)
% Output        : yuv (YCbCr image after transformation)
% YOUR CODE HERE
    size_rgb = size(rgb);
    tran = [0.299 0.587 0.114;-0.169 -0.331 0.5;0.5 -0.419 -0.081];
    yuv=reshape((tran*squeeze(reshape(rgb,[],3))')',size_rgb);
end
function image_paths=image_dataset(path)
    % this function reads data of images from a subfolder
    % and return a cell array of image paths
    % images = {"path1" "path2" "path3" ...}'
    file_atts = dir(path);
    n = length(file_atts)-2;
    image_paths = cell(n,1);
    for i = 1:n
        image_paths(i) = {strcat(path,'/',file_atts(i+2).name)};
    end
end
function PSNR = calcPSNR(Image, recImage)
% Input         : Image    (Original Image)
%                 recImage (Reconstructed Image)
%
% Output        : PSNR     (Peak Signal to Noise Ratio)
% YOUR CODE HERE
% call calcMSE to calculate MSE
    PSNR = 10*log10((2^8-1)^2/calcMSE(Image, recImage));
end
function zz = ZigZag8x8(quant)
%  Input         : quant (Quantized Coefficients, 8x8x3)
% 
%  Output        : zz (zig-zag scaned Coefficients, 64x3)
    seed = [1;9;2;3;10;17;25;18;11;4;5;12;19;26;33;41;34;27;20;13;6;7;14;21;28;35;42;49;57;50;43;36;29;22;15;8;16;23;30;37;44;51;58;59;52;45;38;31;24;32;39;46;53;60;61;54;47;40;48;55;62;63;56;64];
    if(length(size(quant))==2)
        zz = quant(seed);
    else
        n = size(quant,3);
        zz = zeros(64,n);
        for i = 1:size(quant,3)
            quant_slice = quant(:,:,i);
            zz(:,i) = quant_slice(seed);
        end
    end
end
function rec_image = SSD_rec8x8(ref_image, motion_vectors)
%  Input         : ref_image(Reference Image, YCbCr image)
%                  motion_vectors
%
%  Output        : rec_image (Reconstructed current image, YCbCr image)
    m = 4;
    [r,c,~] = size(ref_image);  
    n = 8;
    r_shift = ceil(motion_vectors/(2*m+1));
    c_shift = mod(motion_vectors,(2*m+1));
    c_shift(c_shift == 0) = (2*m+1);
    r_shift = r_shift - m - 1;
    c_shift = c_shift - m - 1;
    rec_image = zeros(size(ref_image));
    for i = 1:r/n
        for j = 1:c/n
            rec_image(i*n-n+1:i*n,j*n-n+1:j*n,:);
            a = j*n+c_shift(i,j);
            ref_image(1:2,j*n-n+1+c_shift(i,j):j*n+c_shift(i,j),:);
            ref_image(i*n-n+1+r_shift(i,j):i*n+r_shift(i,j),1:2,:);
            rec_image(i*n-n+1:i*n,j*n-n+1:j*n,:) = ref_image(i*n-n+1+r_shift(i,j):i*n+r_shift(i,j),j*n-n+1+c_shift(i,j):j*n+c_shift(i,j),:);
        end
    end
end
function dst = ZeroRunDec_EoB8x8(src, EoB)
%  Function Name : ZeroRunDec1.m zero run level decoder
%  Input         : src (zero run encoded sequence 1xM with EoB signs)
%                  EoB (end of block sign)
%
%  Output        : dst (reconstructed zig-zag scanned sequence 1xN)
    dst = zeros(1,64);
    i = 1;
    j = 1;
    while(1)
        if(src(i)==0)
            j = j+src(i+1);
            if j > length(dst)
                dst = [dst  zeros(1,64)];
            end
            i = i+2;
            j = j+1;
        elseif src(i)==EoB
            j = ceil(j/64)*64;
            if j > length(dst)
                dst = [dst  zeros(1,64)];
            end
            i = i+1;
            j = j+1;
        else
            if j > length(dst)
                dst = [dst  zeros(1,64)];
            end
            dst(j) = src(i);
            i = i+1;
            j = j+1;
        end
        if(i>length(src))
            break;
        end
    end
end
function dct_block = DeQuant8x8(quant_block, qScale)
%  Function Name : DeQuant8x8.m
%  Input         : quant_block  (Quantized Block, 8x8x3)
%                  qScale       (Quantization Parameter, scalar)
%
%  Output        : dct_block    (Dequantized DCT coefficients, 8x8x3)
    L = [16 11 10 16 24 40 51 61;
        12 12 14 19 26 58 60 55;
        14 13 16 24 40 57 69 56;
        14 17 22 29 51 87 80 62;
        18 55 37 56 68 109 103 77;
        24 35 55 64 81 104 113 92;
        49 64 78 87 103 121 120 101;
        72 92 95 98 112 100 103 99];
    C = [17 18 24 47 99 99 99 99;
        18 21 26 66 99 99 99 99;
        24 13 56 99 99 99 99 99;
        47 66 99 99 99 99 99 99;
        99 99 99 99 99 99 99 99;
        99 99 99 99 99 99 99 99;
        99 99 99 99 99 99 99 99;
        99 99 99 99 99 99 99 99];
    dct_block = zeros(size(quant_block));
    dct_block(:,:,1) = qScale.*quant_block(:,:,1).*L;
    dct_block(:,:,2:3) = qScale.*quant_block(:,:,2:3).*C;
end
function dst = IntraDecode8x8(image, img_size, qScale)
    %  Function Name : IntraDecode.m
    %  Input         : image (zero-run encoded image, 1xN)
    %                  img_size (original image size)
    %                  qScale(quantization scale)
    %  Output        : dst   (decoded image)
    EoB = 4001;
    r = img_size(1);
    c = img_size(2);
    dst = zeros(img_size);
    dst_re = ZeroRunDec_EoB8x8(image, EoB);
    for i = 1:r/8
        for j = 1:c/8
            zz = reshape(dst_re((i-1)*c*3*8+(j-1)*8*8*3+1:(i-1)*c*3*8+j*8*8*3),[],3);
            coeffs = DeZigZag8x8(zz);
            dct_block = DeQuant8x8(coeffs, qScale);
            dct_dst((i-1)*8+1:i*8,(j-1)*8+1:j*8,:) = dct_block;
        end
    end
    idct_coeff = get_idct_coeff(8);
    col_idct_coeff = kron(eye(r/8),idct_coeff);
    row_idct_coeff = kron(eye(c/8),idct_coeff);
    for rgb = 1:3
        dst(:,:,rgb) = col_idct_coeff*dct_dst(:,:,rgb)*(row_idct_coeff');
    end
    dst = ictYCbCr2RGB(dst);
end
function dst = IntraEncode8x8(image, qScale)
    %  Function Name : IntraEncode.m
    %  Input         : image (Original RGB Image)
    %                  qScale(quantization scale)
    %  Output        : dst   (sequences after zero-run encoding, 1xN)
    EOB = 4001;
    image = ictRGB2YCbCr(image);
    [r,c,~] = size(image);
    dst = zeros(1,r*c*3/64);
    
    dct_coeff = get_dct_coeff(8);
    col_dct_coeff = kron(eye(r/8),dct_coeff);
    row_dct_coeff = kron(eye(c/8),dct_coeff);
    dct_image = zeros(size(image));
    for rgb = 1:3
        dct_image(:,:,rgb) = col_dct_coeff*image(:,:,rgb)*(row_dct_coeff');
    end
    
    for i = 1:r/8
        for j = 1:c/8
            dct_block = dct_image((i-1)*8+1:i*8,(j-1)*8+1:j*8,:);
            quant = Quant8x8(dct_block, qScale);
            zz = ZigZag8x8(quant);
            dst((i-1)*c*3*8+(j-1)*8*8*3+1:(i-1)*c*3*8+j*8*8*3) = reshape(zz,[],1);
        end
    end
    dst(dst<-1000) = -1000;
    dst(dst>4000) = 4000;
    dst = ZeroRunEnc_EoB8x8(dst, EOB);
end
function coeffs = DeZigZag8x8(zz)
%  Function Name : DeZigZag8x8.m
%  Input         : zz    (Coefficients in zig-zag order)
%
%  Output        : coeffs(DCT coefficients in original order)
    i = 1;
    j = 1;
    n = 8;
    k = 1;
    direction = 0;
    % direction = 0 means go right up
    % direction = 1 means go left down
    coeffs = zeros(8,8,size(zz,2));
    while(1)
        coeffs(i,j,:) = zz(k,:);
        k = k+1;
        if i==n&&j==n
        	break;
        end
        if direction == 0
            if j==n
                direction = 1;
                i = i+1;
            elseif i==1
                direction = 1;
                j = j+1;
            else
                i = i-1;
                j = j+1;
            end
        else
            if i==n
                direction = 0;
                j = j+1;
            elseif j==1
                direction = 0;
                i = i+1;
            else
                i = i+1;
                j = j-1;
            end        
        end
    end
end
function zze = ZeroRunEnc_EoB8x8(zz, EOB)
%  Input         : zz (Zig-zag scanned sequence, 1xN)
%                  EOB (End Of Block symbol, scalar)
%
%  Output        : zze (zero-run-level encoded sequence, 1xM)

    processing_zero = false;
    zze = zeros(1,length(zz)*2);
    j = 1;
    for i = 1:length(zz)
            if(processing_zero == false)
                if(zz(i)==0)
                    processing_zero = true;
                    last_continous_zero_position = i;
                    if mod(i,64)==0
                        zze(j) = EOB;
                        j = j+1;
                        processing_zero = false;
                    end
                else
                    zze(j) = zz(i);
                    j = j+1;
                end
            else
                if~(zz(i)==0)
                    processing_zero = false;
                    length_min_1 = i-last_continous_zero_position-1;
                    zze(j) = 0;
                    zze(j+1) = length_min_1;
                    zze(j+2) = zz(i);
                    j = j+3;
                else
                    length_min_1 = i-last_continous_zero_position-1;
                    if mod(i,64)==0
                        zze(j) = EOB;
                        j = j+1;
                        processing_zero = false;
                     end
                end
            end

    end
    zze = zze(1:j-1);
end
%--------------------------------------------------------------
%
%
%
%           %%%    %%%       %%%      %%%%%%%%
%           %%%    %%%      %%%     %%%%%%%%%
%           %%%    %%%     %%%    %%%%
%           %%%    %%%    %%%    %%%
%           %%%    %%%   %%%    %%%
%           %%%    %%%  %%%    %%%
%           %%%    %%% %%%    %%%
%           %%%    %%%%%%    %%%
%           %%%    %%%%%     %%%
%           %%%    %%%%       %%%%%%%%%%%%
%           %%%    %%%          %%%%%%%%%   BUILDHUFFMAN.M
%
%
% description:  creatre a huffman table from a given distribution
%
% input:        data              - Data to be encoded (indices to codewords!!!!
%               BinCode           - Binary version of the Code created by buildHuffman
%               Codelengths       - Array of Codelengthes created by buildHuffman
%
% returnvalue:  bytestream        - the encoded bytestream
%
% Course:       Image and Video Compression
%               Prof. Eckehard Steinbach
%
%-----------------------------------------------------------------------------------

function [bytestream] = enc_huffman_new( data, BinCode, Codelengths)

a = BinCode(data(:),:)';
b = a(:);
mat = zeros(ceil(length(b)/8)*8,1);
p  = 1;
for i = 1:length(b)
    if b(i)~=' '
        mat(p,1) = b(i)-48;
        p = p+1;
    end
end
p = p-1;
mat = mat(1:ceil(p/8)*8);
d = reshape(mat,8,ceil(p/8))';
multi = [1 2 4 8 16 32 64 128];
bytestream = sum(d.*repmat(multi,size(d,1),1),2);

end
%--------------------------------------------------------------
%
%
%
%           %%%    %%%       %%%      %%%%%%%%
%           %%%    %%%      %%%     %%%%%%%%%            
%           %%%    %%%     %%%    %%%%
%           %%%    %%%    %%%    %%%
%           %%%    %%%   %%%    %%%
%           %%%    %%%  %%%    %%%
%           %%%    %%% %%%    %%%
%           %%%    %%%%%%    %%%
%           %%%    %%%%%     %%% 
%           %%%    %%%%       %%%%%%%%%%%%
%           %%%    %%%          %%%%%%%%%   BUILDHUFFMAN.M
%
%
% description:  creatre a huffman table from a given distribution
%
% input:        bytestream        - Encoded bitstream
%               BinaryTree        - Binary Tree of the Code created by buildHuffma
%               nr_symbols        - Number of symbols to decode
%
% returnvalue:  output            - decoded data
%
% Course:       Image and Video Compression
%               Prof. Eckehard Steinbach
%
%
%-----------------------------------------------------------------------------------

function [output] = dec_huffman_new (bytestream, BinaryTree, nr_symbols)

output = zeros(1,nr_symbols);
ctemp = BinaryTree;

dec = zeros(size(bytestream,1),8);
for i = 8:-1:1
    dec(:,i) = rem(bytestream,2);
    bytestream = floor(bytestream/2);
end

dec = dec(:,end:-1:1)';
a = dec(:);

i = 1;
p = 1;
while(i <= nr_symbols)&&p<=max(size(a))
    while(isa(ctemp,'cell'))
        next = a(p)+1;
        p = p+1;
        ctemp = ctemp{next};
    end;
    output(i) = ctemp;
    ctemp = BinaryTree;
    i=i+1;
end
end
function motion_vectors_indices = SSD8x8(ref_image, image)
%  Input         : ref_image(Reference Image, size: height x width)
%                  image (Current Image, size: height x width)
%
%  Output        : motion_vectors_indices (Motion Vector Indices, size: (height/8) x (width/8) x 1 )
    m = 4;
    if(length(size(image))==3)
        [r,c,~] = size(image);
    else
        [r,c] = size(image);
    end
    n = 8;
    speed_up = true;
    if (speed_up)
        l_kern = tril(ones(r));
        r_kern = triu(ones(c));
        image_sum = padarray(l_kern * image *r_kern,[1 1 ],0,'pre');
        ref_image_sum = padarray(l_kern * ref_image *r_kern,[1 1 ],0,'pre');
        image_sum8 = image_sum(n+1:end,n+1:end) - image_sum(1:end-n,n+1:end) - image_sum(n+1:end,1:end-n) + image_sum(1:end-n,1:end-n);
        ref_image_sum8 = ref_image_sum(n+1:end,n+1:end) - ref_image_sum(1:end-n,n+1:end) - ref_image_sum(n+1:end,1:end-n) + ref_image_sum(1:end-n,1:end-n);
        motion_vectors_indices = zeros(r/8,c/8);
    end
    for i = 1:r/8
        for j = 1:c/8 
            block = image(i*8-7:i*8,j*8-7:j*8);
            best_indice = (2*m+1)*m+m+1;
            best_SSE = sum((block-ref_image(i*8-7:i*8,j*8-7:j*8)).^2,'all');
            current_indice = 0;
            for p = -m:m
                for q = -m:m
                    current_indice = current_indice+1;
                    if(i*8-7+p>=1&i*8-7+p<=r-n+1&j*8-7+q>=1&j*8-7+q<=c-n+1)
                        if(speed_up)
                            trian = 1/n/n*(abs(image_sum8(i*8-7,j*8-7) - ref_image_sum8(i*8-7+p,j*8-7+q)))^2;
                            if(trian>best_SSE)
                                continue;
                            end
                        end
                        ref_block = ref_image(i*8-7+p:i*8+p,j*8-7+q:j*8+q);
                        SSE = sum((block-ref_block).^2,'all');
                        if SSE < best_SSE
                            best_SSE = SSE;
                            best_indice = current_indice;
                        end
                    end
                end
            end
            motion_vectors_indices(i,j) = best_indice;  
        end
    end
end  
%--------------------------------------------------------------
%
%
%
%           %%%    %%%       %%%      %%%%%%%%
%           %%%    %%%      %%%     %%%%%%%%%            
%           %%%    %%%     %%%    %%%%
%           %%%    %%%    %%%    %%%
%           %%%    %%%   %%%    %%%
%           %%%    %%%  %%%    %%%
%           %%%    %%% %%%    %%%
%           %%%    %%%%%%    %%%
%           %%%    %%%%%     %%% 
%           %%%    %%%%       %%%%%%%%%%%%
%           %%%    %%%          %%%%%%%%%   BUILDHUFFMAN.M
%
%
% description:  creatre a huffman table from a given distribution
%
% input:        PMF               - probabilty mass function of the source
%
% returnvalue:  BinaryTree        - cell structure containing the huffman tree
%               HuffCode          - Array of integers containing the huffman tree
%               BinCode           - Matrix containing the binary version of the code
%               Codelengths       - Array with number of bits in each Codeword
%
% Course:       Image and Video Compression
%               Prof. Eckehard Steinbach
%
% Author:       Dipl.-Ing. Ingo Bauermann 
%               02.01.2003 (created)
%
%-----------------------------------------------------------------------------------


function [ BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman( p );
    global y

    p=p(:)/sum(p)+eps;              % normalize histogram
    p1=p;                           % working copy

    c=cell(length(p1),1);			% generate cell structure 

    for i=1:length(p1)				% initialize structure
       c{i}=i;						
    end

    while size(c)-2					% build Huffman tree
        [p1,i]=sort(p1);			% Sort probabilities
        c=c(i);						% Reorder tree.
        c{2}={c{1},c{2}};           % merge branch 1 to 2
        c(1)=[];	                % omit 1
        p1(2)=p1(1)+p1(2);          % merge Probabilities 1 and 2 
        p1(1)=[];	                % remove 1
    end

    %cell(length(p),1);              % generate cell structure
    getcodes(c,[]);                  % recurse to find codes
    code=char(y);

    [numCodes maxlength] = size(code); % get maximum codeword length

    % generate byte coded huffman table
    % code

    length_b=0;
    HuffCode=zeros(1,numCodes);
    for symbol=1:numCodes
        for bit=1:maxlength
            length_b=bit;
            if(code(symbol,bit)==char(49)) HuffCode(symbol) = HuffCode(symbol)+2^(bit-1)*(double(code(symbol,bit))-48);
            elseif(code(symbol,bit)==char(48))
            else 
                length_b=bit-1;
                break;
            end;
        end;
        Codelengths(symbol)=length_b;
    end;

    BinaryTree = c;
    BinCode = code;
    clear global y;
    %return
end
%----------------------------------------------------------------
function getcodes(a,dum)       
global y                            % in every level: use the same y
if isa(a,'cell')                    % if there are more branches...go on
         getcodes(a{1},[dum 0]);    % 
         getcodes(a{2},[dum 1]);
else   
   y{a}=char(48+dum);   
end
end
function dct_coeff = get_idct_coeff(N)
    %N = 8;
    dct_coeff = zeros(N,N);
    for k = 1:N
        for n = 1:N
            dct_coeff(k,n) = sqrt(2/N)/sqrt(1+(k==1))*cos(pi/2/N*(2*n-1)*(k-1));  
        end
    end
    dct_coeff = dct_coeff';
end
function dct_coeff = get_dct_coeff(N)
    %N = 8;
    dct_coeff = zeros(N,N);
    for k = 1:N
        for n = 1:N
            dct_coeff(k,n) = sqrt(2/N)/sqrt(1+(k==1))*cos(pi/2/N*(2*n-1)*(k-1));  
        end
    end
end
function quant = Quant8x8(dct_block, qScale)
%  Input         : dct_block (Original Coefficients, 8x8x3)
%                  qScale (Quantization Parameter, scalar)
%
%  Output        : quant (Quantized Coefficients, 8x8x3)
    L = [16 11 10 16 24 40 51 61;
        12 12 14 19 26 58 60 55;
        14 13 16 24 40 57 69 56;
        14 17 22 29 51 87 80 62;
        18 55 37 56 68 109 103 77;
        24 35 55 64 81 104 113 92;
        49 64 78 87 103 121 120 101;
        72 92 95 98 112 100 103 99];
    C = [17 18 24 47 99 99 99 99;
        18 21 26 66 99 99 99 99;
        24 13 56 99 99 99 99 99;
        47 66 99 99 99 99 99 99;
        99 99 99 99 99 99 99 99;
        99 99 99 99 99 99 99 99;
        99 99 99 99 99 99 99 99;
        99 99 99 99 99 99 99 99];
    quant = zeros(size(dct_block));
    quant(:,:,1) = round(dct_block(:,:,1)./L./qScale);
    quant(:,:,2:3) = round(dct_block(:,:,2:3)./C./qScale);
end


