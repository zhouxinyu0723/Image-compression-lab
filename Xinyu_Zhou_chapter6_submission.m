clear
close all
folder_path = strcat('foreman20_40_RGB');
image_paths=image_dataset(folder_path);
%% prepare for for loop
scales = [0.08, 0.2, 0.4, 0.8, 1.0, 1.5, 2, 3, 4, 4.5];
lambdas = [2, 2,2,2,2,2,2,2,2,2,2];
bitPerPixel = zeros(length(scales),1);
PSNR = zeros(length(scales),1);
%% for loop
for scaleIdx = 1 : numel(scales)%scaleIdx = 1;
qScale = scales(scaleIdx);
lambda = lambdas(scaleIdx);
bias = 1001;
EOB = 4001;
qScale_str=sprintf('%.2f',qScale);
rec_path = strcat(folder_path,'_CTUs_mode_LF_',qScale_str,'/');
mkdir(rec_path);
byte_count = 0;
PSNR_each_image = zeros(length(image_paths),1);
MV_bytes_each_image = zeros(length(image_paths),1);
Residual_bytes_each_image = zeros(length(image_paths),1);
tree_bytes_each_image = zeros(length(image_paths),1);
mode_bytes_each_image = zeros(length(image_paths),1);

%% get the firt and the second images
I = double(imread(image_paths{1}));
P1 = double(imread(image_paths{2}));
[r,c,~] = size(P1);
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
[Fn1,LF_h] = loop_filter(Fn1,8,qScale*16*sqrt(3),qScale*16*0.5*sqrt(3)/2 );
imwrite(uint8(LF_h),strcat(rec_path,int2str(1+19),'_LF.bmp'));
imwrite(uint8(Fn1),strcat(rec_path,int2str(1+19),'.bmp'));
PSNR_each_image(1)=calcPSNR(I, Fn1);
Residual_bytes_each_image(1) = numel(bytestream_I);
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
[Fn1,h] = loop_filter(Fn1,16,qScale*16*2*sqrt(3),qScale*16*0.5*sqrt(3) );
%% inter residual
% first propsed TUs_layers
sizes = [32 16 8 4];
modes = [1 2];
Residual = P1-mcp;
[TUslayers_mode,TUs_layers,modes_ZeroRun_codes,modes_MSEs,modes_rec_imgs] = simple_residual_analyse_with_modes(P1,mcp,modes,sizes,qScale);    
% entropy optimization
for TU_iter = 1:3
    % generat haffmann
    [TUs_tree_code_flat,mode_code_flat,ZeroRun_code_flat,TUs_ZeroRun_code_diff_layers,unit_index]=TUs_encode(TUs_layers,TUslayers_mode,modes_ZeroRun_codes);
    bias = 1001;
    BinaryTreeQuantize_Residual = cell(1,numel(sizes));
    HuffCodeQuantize_Residual = cell(1,numel(sizes));
    BinCodeQuantize_Residual = cell(1,numel(sizes));
    CodelengthsQuantize_Residual = cell(1,numel(sizes));
    for i = 1:numel(sizes)
        pmf_Residual = stats_marg(TUs_ZeroRun_code_diff_layers{i}+bias, 1:5010);
        [ BinaryTreeQuantize_Residual{i}, HuffCodeQuantize_Residual{i}, BinCodeQuantize_Residual{i}, CodelengthsQuantize_Residual{i}] = buildHuffman( pmf_Residual );
    end
    % calculate Rate
    mode_bitPerPixel_block = cell(1,numel(modes));
    for iii = 1:numel(modes)
        mode_bitPerPixel_block{iii} = cell(1,numel(sizes));
        for i = 1:numel(sizes)
            for j = 1:size(TUslayers_mode{i},1)
                mode_bitPerPixel_block{iii}{i} = zeros(size(TUslayers_mode{i},1),size(TUslayers_mode{i},2));
                for k = 1:size(TUslayers_mode{i},1)
                    bytestream_Residual = enc_huffman_new(modes_ZeroRun_codes{iii}{i}{j,k}+bias , BinCodeQuantize_Residual{i}, CodelengthsQuantize_Residual{i}); 
                    mode_bitPerPixel_block{iii}{i}(j,k) = numel(bytestream_Residual)*8/sizes(i)/sizes(i);
                end
            end
        end
    end
    [TUslayers_mode,TUs_layers] = cost_analyse_with_modes(modes_MSEs,mode_bitPerPixel_block,lambda);
end

[TUs_tree_code_flat,mode_code_flat,ZeroRun_code_flat,TUs_ZeroRun_code_diff_layers,unit_index]=TUs_encode(TUs_layers,TUslayers_mode,modes_ZeroRun_codes);
% now we get haffman table and the TUs_layers for the firs Residual image
% then encode and decode the first Residual image
% encode tree_code haffman
[TUs_tree_code_flat_bytest,maxi] = bits2bytes(TUs_tree_code_flat,8);
TUs_tree_code_flat_bytest = TUs_tree_code_flat_bytest+1;
maxi = maxi+1;
pmf_tree_code = stats_marg(TUs_tree_code_flat_bytest, 1:maxi);
[ BinaryTreeQuantize_mode_tree_code, HuffCodeQuantize_tree_code, BinCodeQuantize_tree_code, CodelengthsQuantize_tree_code] = buildHuffman( pmf_tree_code );
% encode tree_code
bytestream_tree_code = enc_huffman_new(TUs_tree_code_flat_bytest, BinCodeQuantize_tree_code, CodelengthsQuantize_tree_code);
byte_count =  byte_count + numel(bytestream_tree_code);
% decode tree_code
tree_code_flat_bytest_rec = dec_huffman_new (bytestream_tree_code, BinaryTreeQuantize_mode_tree_code, length(TUs_tree_code_flat_bytest));
tree_code_flat_bytest_rec = tree_code_flat_bytest_rec-1;
TUs_tree_code_flat_rec = bytes2bits(tree_code_flat_bytest_rec,8);

% encode mode_code haffman
[mode_code_flat_bytest,maxi] = bits2bytes(mode_code_flat-1,8);
mode_code_flat_bytest = mode_code_flat_bytest+1;
maxi = maxi+1;
pmf_mode_code = stats_marg(mode_code_flat_bytest, 1:maxi);
[ BinaryTreeQuantize_mode_code, HuffCodeQuantize_mode_code, BinCodeQuantize_mode_code, CodelengthsQuantize_mode_code] = buildHuffman( pmf_mode_code );
% encode mode code
bytestream_mode_code = enc_huffman_new(mode_code_flat_bytest, BinCodeQuantize_mode_code, CodelengthsQuantize_mode_code);
byte_count =  byte_count + numel(bytestream_mode_code);
% decode mode_code
mode_code_flat_bytest_rec = dec_huffman_new (bytestream_mode_code, BinaryTreeQuantize_mode_code, length(mode_code_flat_bytest));
mode_code_flat_bytest_rec = mode_code_flat_bytest_rec-1;
mode_code_flat_rec = bytes2bits(mode_code_flat_bytest_rec,8)+1;

% encode residual
my_byte_count = 0;
bytestream_Residual = cell(1:numel(sizes));
for i = 1:numel(sizes)
    bytestream_Residual{i} = enc_huffman_new(TUs_ZeroRun_code_diff_layers{i}+bias , BinCodeQuantize_Residual{i}, CodelengthsQuantize_Residual{i}); 
    byte_count = byte_count + numel(bytestream_Residual{i});
    my_byte_count =  my_byte_count + numel(bytestream_Residual{i});
end
fprintf('MV bytes: %d \t Residual bytes: %d \t Residual tree bytes: %d \n mode bytes: %d \n',numel(bytestream_MV), my_byte_count, numel(bytestream_tree_code), numel(bytestream_mode_code));
% decode residual
TUs_ZeroRun_code_rec = cell(1:numel(sizes));
for i = 1:numel(sizes)
    a  = dec_huffman_new (bytestream_Residual{i}, BinaryTreeQuantize_Residual{i}, length(TUs_ZeroRun_code_diff_layers{i}))-bias;
    TUs_ZeroRun_code_rec{i} = a ;
end
% reconstruct
[Fn,PUs_h,img_Unit_m] = TUs_decode(mcp,TUs_tree_code_flat_rec,mode_code_flat,TUs_ZeroRun_code_rec,sizes,qScale);
% loop filter
[Fn,LF_h] = loop_filter(Fn,16,qScale*16*2*sqrt(3),qScale*16*0.5*sqrt(3) );
% load data
imwrite(uint8(P1-mcp),strcat(rec_path,int2str(2+19),'_Residuals.bmp'));
imwrite(uint8(PUs_h*30),strcat(rec_path,int2str(2+19),'_PUs.bmp'));
imwrite(uint8(img_Unit_m),strcat(rec_path,int2str(2+19),'_modes.bmp'));
imwrite(uint8(LF_h),strcat(rec_path,int2str(2+19),'_LF.bmp'));
imwrite(uint8(Fn),strcat(rec_path,int2str(2+19),'.bmp'));
MV_bytes_each_image(2) = numel(bytestream_MV);
Residual_bytes_each_image(2) = my_byte_count;
PSNR_each_image(2)=calcPSNR(P1, Fn)
%%
for ii = 3:length(image_paths)
        P = double(imread(image_paths{ii}));
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
        mcp = SSD_rec8x8(Fn, MV_rec);
        %%
        [TUslayers_mode,TUs_layers,modes_ZeroRun_codes,modes_MSEs,modes_rec_imgs] = simple_residual_analyse_with_modes(P,mcp,modes,sizes,qScale);    
        [TUs_tree_code_flat,mode_code_flat,ZeroRun_code_flat,TUs_ZeroRun_code_diff_layers,unit_index]=TUs_encode(TUs_layers,TUslayers_mode,modes_ZeroRun_codes);
        % calculate Rate
        mode_bitPerPixel_block = cell(1,numel(modes));
        for iii = 1:numel(modes)
            mode_bitPerPixel_block{iii} = cell(1,numel(sizes));
            for i = 1:numel(sizes)
                for j = 1:size(TUslayers_mode{i},1)
                    mode_bitPerPixel_block{iii}{i} = zeros(size(TUslayers_mode{i},1),size(TUslayers_mode{i},2));
                    for k = 1:size(TUslayers_mode{i},1)
                        bytestream_Residual = enc_huffman_new(modes_ZeroRun_codes{iii}{i}{j,k}+bias , BinCodeQuantize_Residual{i}, CodelengthsQuantize_Residual{i}); 
                        mode_bitPerPixel_block{iii}{i}(j,k) = numel(bytestream_Residual)*8/sizes(i)/sizes(i);
                    end
                end
            end
        end
        [TUslayers_mode,TUs_layers] = cost_analyse_with_modes(modes_MSEs,mode_bitPerPixel_block,lambda);
        [TUs_tree_code_flat,mode_code_flat,ZeroRun_code_flat,TUs_ZeroRun_code_diff_layers,unit_index]=TUs_encode(TUs_layers,TUslayers_mode,modes_ZeroRun_codes);
        %%%%% end for
        % improve  layers with entropy coding
        %% 6.2 Residual encode continous
        % encode tree_code
        TUs_tree_code_flat_bytest = bits2bytes(TUs_tree_code_flat,8);
        TUs_tree_code_flat_bytest = TUs_tree_code_flat_bytest+1;
        bytestream_tree_code = enc_huffman_new(TUs_tree_code_flat_bytest, BinCodeQuantize_tree_code, CodelengthsQuantize_tree_code);
        byte_count =  byte_count + numel(bytestream_tree_code);
        % decode tree_code
        tree_code_flat_bytest_rec = dec_huffman_new (bytestream_tree_code, BinaryTreeQuantize_mode_tree_code, length(TUs_tree_code_flat_bytest));
        tree_code_flat_bytest_rec = tree_code_flat_bytest_rec-1;
        TUs_tree_code_flat_rec = bytes2bits(tree_code_flat_bytest_rec,8);

        % encode mode_code
        [mode_code_flat_bytest,maxi] = bits2bytes(mode_code_flat-1,8);
        mode_code_flat_bytest = mode_code_flat_bytest+1;
        bytestream_mode_code = enc_huffman_new(mode_code_flat_bytest, BinCodeQuantize_mode_code, CodelengthsQuantize_mode_code);
        byte_count =  byte_count + numel(bytestream_mode_code);
        % decode mode_code
        mode_code_flat_bytest_rec = dec_huffman_new (bytestream_mode_code, BinaryTreeQuantize_mode_code, length(mode_code_flat_bytest));
        mode_code_flat_bytest_rec = mode_code_flat_bytest_rec-1;
        mode_code_flat_rec = bytes2bits(mode_code_flat_bytest_rec,8)+1;

        % encode residual
        my_byte_count = 0;
        bytestream_Residual = cell(1:numel(sizes));
        for i = 1:numel(sizes)
            bytestream_Residual{i} = enc_huffman_new(TUs_ZeroRun_code_diff_layers{i}+bias , BinCodeQuantize_Residual{i}, CodelengthsQuantize_Residual{i});
            byte_count = byte_count + numel(bytestream_Residual{i});
            my_byte_count =  my_byte_count + numel(bytestream_Residual{i});
        end
        fprintf('MV bytes: %d \t Residual bytes: %d \t Residual tree bytes: %d \n mode bytes: %d \n',numel(bytestream_MV), my_byte_count, numel(bytestream_tree_code), numel(bytestream_mode_code));
        %% 6.3 Residual decode continous
        TUs_ZeroRun_code_rec = cell(1:numel(sizes));
        for i = 1:numel(sizes)
            a  = dec_huffman_new (bytestream_Residual{i}, BinaryTreeQuantize_Residual{i}, length(TUs_ZeroRun_code_diff_layers{i}))-bias;
            TUs_ZeroRun_code_rec{i} = a ;
        end
        %% 6.4 residual recon
        [Fn,PUs_h,img_Unit_m] = TUs_decode(mcp,TUs_tree_code_flat_rec,mode_code_flat_rec,TUs_ZeroRun_code_rec,sizes,qScale);
        imshow(uint8(PUs_h)*30);
        % loop filter
        [Fn,LF_h] = loop_filter(Fn,16,qScale*16*sqrt(3),qScale*16*0.5*sqrt(3)/2 );
        %% 7 Residual + motion continous with gauss filter
        imshow(uint8(LF_h));
        imwrite(uint8(P-mcp),strcat(rec_path,int2str(ii+19),'_Residuals.bmp'));
        imwrite(uint8(PUs_h*30),strcat(rec_path,int2str(ii+19),'_PUs.bmp'));
        imwrite(uint8(img_Unit_m),strcat(rec_path,int2str(ii+19),'_modes.bmp'));
        imwrite(uint8(LF_h),strcat(rec_path,int2str(ii+19),'_LF.bmp'));
        imwrite(uint8(Fn),strcat(rec_path,int2str(ii+19),'.bmp'));
        pppppp = calcPSNR(P, Fn)
        PSNR_each_image(ii)=pppppp;
        MV_bytes_each_image(ii) = numel(bytestream_MV);
        Residual_bytes_each_image(ii) = my_byte_count;
        tree_bytes_each_image(ii) = numel(bytestream_tree_code);
        mode_bytes_each_image(ii) = numel(bytestream_mode_code);
        code_info =[MV_bytes_each_image Residual_bytes_each_image tree_bytes_each_image mode_bytes_each_image];
        save(strcat(rec_path,int2str(ii+19),'code_info'),'code_info');
end
    bitPerPixel(scaleIdx) = (byte_count*8) / (numel(I)/3) /length(image_paths);
    PSNR(scaleIdx) = mean(PSNR_each_image);  
    fprintf('QP: %.2f bit-rate: %.2f bits/pixel PSNR: %.2fdB\n', qScale, bitPerPixel(scaleIdx), PSNR(scaleIdx))
    
end
info_path = strcat(folder_path,'_CTUs_mode_LF_info');  
mkdir(info_path);
save(strcat(info_path,'/scales.mat'),'scales');
save(strcat(info_path,'/PSNR.mat'),'PSNR');
save(strcat(info_path,'/bitPerPixel.mat'),'bitPerPixel');
figure
My_Optimization = plot(bitPerPixel,PSNR,'m^-','LineWidth',1.5);M1 = 'My Optimization';
xlim([0.2 4])
ylim([20 45])
title('RD performance of Optimization, Foreman Sequence');
xlabel('bitrate [bit/pixel]');
ylabel('PSNR [dB]');
legend(M1);
hold off
%% sub functions
function [Fn_rec,img_Unit_h,img_Unit_m] = TUs_decode(mcp,TUs_tree_code_flat,mode_code_flat,TUs_ZeroRun_code,sizes,qScale)
    global deepth
    deepth = numel(sizes);
    nn = sizes(1);
    EoB =  4001;
    [r,c,~]=size(mcp);
    % apply rgb to yuv and the dct to mcp
    mcp_ycc = ictRGB2YCbCr(mcp);
    global mcp_ycc_dcts
    mcp_ycc_dcts = cell(1,numel(sizes));
    for i = 1:numel(sizes)
        n = sizes(i);
        mcp_ycc_dcts{i} = image2dct(mcp_ycc,n);
    end
    % apply quantize on dct coefficient of mcp for mode 2
    global mcp_ycc_dct_q
    mcp_ycc_dct_q = cell(1,numel(sizes));
    for i = 1:numel(sizes)
        n = sizes(i);
        for ii = 1:r/n
            for jj = 1:c/n
                mcp_ycc_dct_q{i}((ii-1)*n+1:ii*n,(jj-1)*n+1:jj*n,:) = DeQuant_inter(Quant_inter(mcp_ycc_dcts{i}((ii-1)*n+1:ii*n,(jj-1)*n+1:jj*n,:),qScale),qScale);
            end
        end
    end
    % pre decode zero run
    global DCT_coeff_blocks
    DCT_coeff_blocks = cell(1,numel(sizes));
    for i = 1:numel(sizes)
        if~isempty(TUs_ZeroRun_code{i})
            n = sizes(i);
            zigzag_flat = ZeroRunDec_EoB(TUs_ZeroRun_code{i},EoB,sizes(i));
            DCT_coeff_blocks{i} = cell(1,numel(zigzag_flat)/(n*n*3));
            for j = 1:numel(zigzag_flat)/(n*n*3)
                zigzag = reshape(zigzag_flat((j-1)*n*n*3+1:j*n*n*3),[],3); 
                q_coeffs = DeZigZag(zigzag,n);
                dct_block = DeQuant_inter(q_coeffs,qScale);
                DCT_coeff_blocks{i}{j} = dct_block;
            end
        end
    end
    % decode
    global TUs_tree_code_flat_recursive mode_code_flat_recursive TUs_ZeroRun_code_recursive sizes_rec deepth
    TUs_tree_code_flat_recursive = TUs_tree_code_flat;
    mode_code_flat_recursive = mode_code_flat;
    TUs_ZeroRun_code_recursive = TUs_ZeroRun_code;
    sizes_rec = sizes;
    deepth = numel(sizes);
    global tree_code_p mode_code_p ZeroRun_ps
    tree_code_p = 1;
    mode_code_p = 1;
    ZeroRun_ps = ones(1,deepth);
    nn = sizes(1);
    for i = 1:size(mcp,1)/sizes(1)
        for j = 1:size(mcp,2)/sizes(1)
            [Fn_rec(i*nn-nn+1:i*nn,j*nn-nn+1:j*nn,:),img_Unit_h(i*nn-nn+1:i*nn,j*nn-nn+1:j*nn,:),img_Unit_m(i*nn-nn+1:i*nn,j*nn-nn+1:j*nn,:)]= rec_CU_with_TUs(1,i*nn-nn+1,j*nn-nn+1);
        end
    end
    % to rgb
    Fn_rec = ictYCbCr2RGB(Fn_rec);
    img_Unit_h =  ictYCbCr2RGB(img_Unit_h);
end


function [img_Unit,img_Unit_h,img_Unit_m] = rec_CU_with_TUs(level,i,j)
    global mcp_ycc_dcts mcp_ycc_dct_q DCT_coeff_blocks
    global TUs_tree_code_flat_recursive mode_code_flat_recursive TUs_ZeroRun_code_recursive sizes_rec
    global tree_code_p mode_code_p ZeroRun_ps sizes_rec
    sizes = sizes_rec;
    if(level==numel(sizes))
        mode = mode_code_flat_recursive(mode_code_p);
        mode_code_p = mode_code_p + 1;
        if(mode == 1)
            img_Unit = DCT_coeff_blocks{level}{ZeroRun_ps(level)}+mcp_ycc_dcts{level}(i:i-1+sizes(level),j:j-1+sizes(level),:);
            img_Unit = IDCT(img_Unit);
            img_Unit_h = IDCT(DCT_coeff_blocks{level}{ZeroRun_ps(level)});
            img_Unit_m = ones(size(img_Unit))*0;
            ZeroRun_ps(level) = ZeroRun_ps(level)+1;
        end
        if(mode==2)
            img_Unit = DCT_coeff_blocks{level}{ZeroRun_ps(level)}+mcp_ycc_dct_q{level}(i:i-1+sizes(level),j:j-1+sizes(level),:);
            img_Unit = IDCT(img_Unit);
            img_Unit_h = IDCT(DCT_coeff_blocks{level}{ZeroRun_ps(level)});
            img_Unit_m = ones(size(img_Unit))*128;
            ZeroRun_ps(level) = ZeroRun_ps(level)+1;
        end
        img_Unit_h(1,:,:)=255;img_Unit_h(end,:,:)=255;img_Unit_h(:,1,:)=255;img_Unit_h(:,end,:)=255;
        img_Unit_m(1,:,:)=255;img_Unit_m(end,:,:)=255;img_Unit_m(:,1,:)=255;img_Unit_m(:,end,:)=255;
    else 
        if(TUs_tree_code_flat_recursive(tree_code_p)==1)
            tree_code_p = tree_code_p+1;
            n = sizes(level);
            img_Unit = zeros(n,n,3);
            img_Unit_h = zeros(n,n,3);
            [img_Unit(1:n/2,1:n/2,:),img_Unit_h(1:n/2,1:n/2,:),img_Unit_m(1:n/2,1:n/2,:)]     = rec_CU_with_TUs(level+1,i,j);
            [img_Unit(1:n/2,n/2+1:n,:),img_Unit_h(1:n/2,n/2+1:n,:),img_Unit_m(1:n/2,n/2+1:n,:)]   = rec_CU_with_TUs(level+1,i,j+n/2);
            [img_Unit(n/2+1:n,1:n/2,:),img_Unit_h(n/2+1:n,1:n/2,:),img_Unit_m(n/2+1:n,1:n/2,:)]   = rec_CU_with_TUs(level+1,i+n/2,j);
            [img_Unit(n/2+1:n,n/2+1:n,:),img_Unit_h(n/2+1:n,n/2+1:n,:),img_Unit_m(n/2+1:n,n/2+1:n,:)] = rec_CU_with_TUs(level+1,i+n/2,j+n/2);
        else
            tree_code_p = tree_code_p+1;
            mode = mode_code_flat_recursive(mode_code_p);
            mode_code_p = mode_code_p + 1;
            if(mode == 1)
                img_Unit = DCT_coeff_blocks{level}{ZeroRun_ps(level)}+mcp_ycc_dcts{level}(i:i-1+sizes(level),j:j-1+sizes(level),:);
                img_Unit = IDCT(img_Unit);
                img_Unit_h = IDCT(DCT_coeff_blocks{level}{ZeroRun_ps(level)});
                img_Unit_m = ones(size(img_Unit))*0;
                ZeroRun_ps(level) = ZeroRun_ps(level)+1;
            end
            if(mode==2)
                img_Unit = DCT_coeff_blocks{level}{ZeroRun_ps(level)}+mcp_ycc_dct_q{level}(i:i-1+sizes(level),j:j-1+sizes(level),:);
                img_Unit = IDCT(img_Unit);
                img_Unit_h = IDCT(DCT_coeff_blocks{level}{ZeroRun_ps(level)});
                img_Unit_m = ones(size(img_Unit))*128;
                ZeroRun_ps(level) = ZeroRun_ps(level)+1;
            end
            img_Unit_h(1,:,:)=255;img_Unit_h(end,:,:)=255;img_Unit_h(:,1,:)=255;img_Unit_h(:,end,:)=255;
            img_Unit_m(1,:,:)=255;img_Unit_m(end,:,:)=255;img_Unit_m(:,1,:)=255;img_Unit_m(:,end,:)=255;
         end
    end
end
function [Fn_filtered,h] = loop_filter(Fn,n,alpha,beta)
    [r c,~] = size(Fn);
    Fn_filtered = Fn;
    h =Fn;
    for i = 1:r/n
        for j = 1:c/n
            for k = 1:n/4
                if (j-1)*n+(k-1)*4-3>=1 & (j-1)*n+(k-1)*4+4<=c
                    p_col = (j-1)*n+(k-1)*4+(1:4);
                    q_col = (j-1)*n+(k-1)*4-(0:3);
                    if(k==1)
                        for l = 1:n
                            row = (i-1)*n+l;
                            p0 = Fn(row,p_col(1),:);
                            p1 = Fn(row,p_col(2),:);
                            p2 = Fn(row,p_col(3),:);
                            p3 = Fn(row,p_col(4),:);
                            q0 = Fn(row,q_col(1),:);
                            q1 = Fn(row,q_col(2),:);
                            q2 = Fn(row,q_col(3),:);
                            q3 = Fn(row,q_col(4),:);
                            if(sum((p0-q0).^2)<alpha&sum((p0-p1).^2)<alpha&sum((q1-q0).^2)<beta)
                                Fn_filtered(row,p_col(1),:)=(p2+2*p1+2*p0+2*q0+q1+4)/8;
                                Fn_filtered(row,p_col(2),:)=(p2+p1+p0+q0+2)/4;
                                Fn_filtered(row,p_col(3),:)=(2*p3+3*p2+p1+p0+q0+4)/8;
                                Fn_filtered(row,q_col(1),:)=(q2+2*q1+2*q0+2*p0+p1+4)/8;
                                Fn_filtered(row,q_col(2),:)=(q2+q1+q0+p0+2)/4;
                                Fn_filtered(row,q_col(3),:)=(2*q3+3*q2+q1+q0+p0+4)/8;
                                h(row,p_col(1),:)= 0;
                                h(row,q_col(1),:)= 0;
                            end
                        end
                    else
                        for l = 1:n
                            row = (i-1)*n+l;
                            p0 = Fn(row,p_col(1),:);
                            p1 = Fn(row,p_col(2),:);
                            p2 = Fn(row,p_col(3),:);
                            q0 = Fn(row,q_col(1),:);
                            q1 = Fn(row,q_col(2),:);
                            q2 = Fn(row,q_col(3),:);
                            if(sum((p0-q0).^2)<alpha&sum((p0-p1).^2)<alpha&sum((q1-q0).^2)<beta)
                                Fn_filtered(row,p_col(1),:)=p0+((q0-p0)*4+(p1-q1)+4)/8;
                                Fn_filtered(row,p_col(2),:)=p1+(p2+(p0+q0+1)/2-2*p1)/2;
                                Fn_filtered(row,q_col(1),:)=q0+((p0-q0)*4+(q1-p1)+4)/8;
                                Fn_filtered(row,q_col(2),:)=q1+(q2+(q0+p0+1)/2-2*q1)/2;
                                h(row,p_col(1),:) = 0;
                                h(row,q_col(1),:) = 0;
                            end
                        end
                    end
                end
                if (i-1)*n+(k-1)*4-3>=1 & (i-1)*n+(k-1)*4+4<=r
                    p_row = (i-1)*n+(k-1)*4+(1:4);
                    q_row = (i-1)*n+(k-1)*4-(0:3);
                    if(k==1)
                        for l = 1:n
                            col = (j-1)*n+l;
                            p0 = Fn(p_row(1),col,:);
                            p1 = Fn(p_row(2),col,:);
                            p2 = Fn(p_row(3),col,:);
                            p3 = Fn(p_row(4),col,:);
                            q0 = Fn(q_row(1),col,:);
                            q1 = Fn(q_row(2),col,:);
                            q2 = Fn(q_row(3),col,:);
                            q3 = Fn(q_row(4),col,:);
                            if(sum((p0-q0).^2)<alpha&sum((p0-p1).^2)<alpha&sum((q1-q0).^2)<beta)
                                Fn_filtered(p_row(1),col,:)=(p2+2*p1+2*p0+2*q0+q1+4)/8;
                                Fn_filtered(p_row(2),col,:)=(p2+p1+p0+q0+2)/4;
                                Fn_filtered(p_row(3),col,:)=(2*p3+3*p2+p1+p0+q0+4)/8;
                                Fn_filtered(q_row(1),col,:)=(q2+2*q1+2*q0+2*p0+p1+4)/8;
                                Fn_filtered(q_row(2),col,:)=(q2+q1+q0+p0+2)/4;
                                Fn_filtered(q_row(3),col,:)=(2*q3+3*q2+q1+q0+p0+4)/8;
                                h(row,p_col(1),:) = 0;
                                h(row,q_col(1),:) = 0;
                            end
                        end
                    else
                        for l = 1:n
                            col = (j-1)*n+l;
                            p0 = Fn(p_row(1),col,:);
                            p1 = Fn(p_row(2),col,:);
                            p2 = Fn(p_row(3),col,:);
                            q0 = Fn(q_row(1),col,:);
                            q1 = Fn(q_row(2),col,:);
                            q2 = Fn(q_row(3),col,:);
                            if(sum((p0-q0).^2)<alpha&sum((p0-p1).^2)<alpha&sum((q1-q0).^2)<beta)
                                Fn_filtered(p_row(1),col,:)=p0+((q0-p0)*4+(p1-q1)+4)/8;
                                Fn_filtered(p_row(2),col,:)=p1+(p2+(p0+q0+1)/2-2*p1)/2;
                                Fn_filtered(q_row(1),col,:)=q0+((p0-q0)*4+(q1-p1)+4)/8;
                                Fn_filtered(q_row(2),col,:)=q1+(q2+(q0+p0+1)/2-2*q1)/2;
                                h(row,p_col(1),:) = 0;
                                h(row,q_col(1),:) = 0;
                            end
                        end
                    end
                end
            end
        end
    end
end
function [TUslayers_mode,TUs_layers,modes_ZeroRun_codes,modes_MSEs,modes_rec_imgs] = simple_residual_analyse_with_modes(target_img,mcp,modes,sizes,qScale)
    modes_pointer = 1;
    % apply rgb to yuv and the dct to both target image and mcp
    [r,c,~] = size(mcp);
    EoB = 4001;
    target_img_ycc = ictRGB2YCbCr(target_img);
    mcp_ycc = ictRGB2YCbCr(mcp);
    target_ycc_dcts = cell(1,numel(sizes));
    mcp_ycc_dcts = cell(1,numel(sizes));
    for i = 1:numel(sizes)
        n = sizes(i);
        target_ycc_dcts{i} = image2dct(target_img_ycc,n);
        mcp_ycc_dcts{i} = image2dct(mcp_ycc,n);
    end
    if any(modes(:) == 1)
    % simulate mode 1 (the defualt mode) to get the code and reconstructed image
        res_ycc_dcts = cell(1,numel(sizes));
        for i = 1:numel(sizes)
            res_ycc_dcts{i} = target_ycc_dcts{i} - mcp_ycc_dcts{i};
        end
        rec_imgs = cell(1,numel(sizes));
        ZeroRun_codes = cell(1,numel(sizes));
        for i = 1:numel(sizes)
            n = sizes(i);
            ZeroRun_codes{i} = cell(r/n,c/n);
            dct_rec = zeros(r,c,3);
            for j = 1:r/n
                for k = 1:c/n
                    Q_dct_block = Quant_inter(res_ycc_dcts{i}(j*n-n+1:j*n,k*n-n+1:k*n,:),qScale);

                    zz =ZigZag_slow(Q_dct_block,n);
                    dst = reshape(zz,[],1);
                    dst(dst<-1000) = -1000;
                    dst(dst>4000) = 4000;

                    ZeroRun_codes{i}{j,k} = ZeroRunEnc_EoB(dst,EoB,n);
                    zeroRunDecoded = ZeroRunDec_EoB(ZeroRun_codes{i}{j,k},EoB,sizes(i));
                    zz = reshape(zeroRunDecoded,[],3); 
                    coeffs = DeZigZag(zz,n);
                    dct_rec(j*n-n+1:j*n,k*n-n+1:k*n,:) = DeQuant_inter(coeffs,qScale) + mcp_ycc_dcts{i}(j*n-n+1:j*n,k*n-n+1:k*n,:);
                end
            end
            rec_imgs{i} = ictYCbCr2RGB(dct2image(dct_rec,n));
        end
        modes_ZeroRun_codes{modes_pointer} = ZeroRun_codes;
        modes_rec_imgs{modes_pointer} =  rec_imgs;
        modes_pointer = modes_pointer+1;
    end
    if any(modes(:) == 2)
    % mode 2 : quantize mcp
    % apply quantize on dct coefficient of mcp
        mcp_ycc_dct_q = cell(1,numel(sizes));
        for i = 1:numel(sizes)
            n = sizes(i);
            for ii = 1:r/n
                for jj = 1:c/n
                    mcp_ycc_dct_q{i}((ii-1)*n+1:ii*n,(jj-1)*n+1:jj*n,:) = DeQuant_inter(Quant_inter(mcp_ycc_dcts{i}((ii-1)*n+1:ii*n,(jj-1)*n+1:jj*n,:),qScale),qScale);
                end
            end
        end
        % simulate mode 2 to get the code, the reconstructed image
        res_ycc_dcts_q = cell(1,numel(sizes));
        for i = 1:numel(sizes)
            res_ycc_dcts_q{i} = target_ycc_dcts{i} - mcp_ycc_dct_q{i};
        end
        rec_imgs_q = cell(1,numel(sizes));
        ZeroRun_codes_q = cell(1,numel(sizes));
        for i = 1:numel(sizes)
            n = sizes(i);
            ZeroRun_codes_q{i} = cell(r/n,c/n);
            dct_rec = zeros(r,c,3);
            for j = 1:r/n
                for k = 1:c/n
                    Q_dct_block = Quant_inter(res_ycc_dcts_q{i}(j*n-n+1:j*n,k*n-n+1:k*n,:),qScale);

                    zz =ZigZag_slow(Q_dct_block,n);
                    dst = reshape(zz,[],1);
                    dst(dst<-1000) = -1000;
                    dst(dst>4000) = 4000;

                    ZeroRun_codes_q{i}{j,k} = ZeroRunEnc_EoB(dst,EoB,n);
                    zeroRunDecoded = ZeroRunDec_EoB(ZeroRun_codes_q{i}{j,k},EoB,sizes(i));
                    zz = reshape(zeroRunDecoded,[],3); 
                    coeffs = DeZigZag(zz,n);
                    dct_rec(j*n-n+1:j*n,k*n-n+1:k*n,:) = DeQuant_inter(coeffs,qScale) + mcp_ycc_dct_q{i}(j*n-n+1:j*n,k*n-n+1:k*n,:);
                end
            end
            rec_imgs_q{i} = ictYCbCr2RGB(dct2image(dct_rec,n));
        end
        modes_ZeroRun_codes{modes_pointer} = ZeroRun_codes_q;
        modes_rec_imgs{modes_pointer} =  rec_imgs_q;
        modes_pointer = modes_pointer+1;
    end
    %% a simple layers
    
    modes_MSEs = cell(1,numel(modes));
    for ii = 1:numel(modes)
        for i = 1:numel(sizes)
            modes_MSEs{ii}{i} = calcBlockMSE(modes_rec_imgs{ii}{i},target_img,sizes(i));
        end  
    end
    
    MSE_with_best_mode = cell(1,numel(sizes));
    TUslayers_mode= cell(1,numel(sizes));% mode 1 use normal mode 2 use quantized mcp
    for i = 1:numel(sizes)
        clear modes_MSE 
        for ii = 1:numel(modes)
            modes_MSE(:,:,ii) = modes_MSEs{ii}{i};
        end
        [MSE_with_best_mode{i},TUslayers_mode{i}]=min(modes_MSE,[],3);
    end
    
    TUs_layers = cell(1,numel(sizes)-1);
    MSE_from_above_layers = MSE_with_best_mode{numel(sizes)};
    for i = numel(sizes)-1:-1:1
        [rr,cc] = size(MSE_from_above_layers);
        MSE_from_above_layers = kron(eye(rr/2),[1/2 1/2])*MSE_from_above_layers*kron(eye(cc/2),[1/2;1/2]);
        MSE_this_layer = MSE_with_best_mode{i};
        TUs_layers{i} = MSE_this_layer > MSE_from_above_layers ;
        MSE_from_above_layers = min(MSE_from_above_layers,MSE_this_layer);
    end    
end
function quant = Quant_inter(dct_block, qScale)
    %  Input         : dct_block (Original Coefficients, 8x8x3)
    %                  qScale (Quantization Parameter, scalar)
    %
    %  Output        : quant (Quantized Coefficients, 8x8x3)
%     quant_Default4x4 =[16,16,16,16
%                        16,16,16,16
%                        16,16,16,16
%                        16,16,16,16];              
%     quant_Default8x8 =[16,16,16,16,17,18,20,24
%                        16,16,16,17,18,20,24,25
%                        16,16,17,18,20,22,25,28
%                        16,17,18,20,24,25,28,33
%                        17,18,20,24,25,28,33,41
%                        18,20,24,25,28,33,41,54
%                        20,24,25,28,33,41,54,71
%                        24,25,28,33,41,54,71,91];
%     quant_Default16x16 = kron(quant_Default8x8,ones(2,2));
%     quant_Default32x32 = kron(quant_Default8x8,ones(4,4));
%     if(size(dct_block,1)==4)
%         quant = round(dct_block./quant_Default4x4./qScale);
%     end
%     if(size(dct_block,1)==8)
%         quant = round(dct_block./quant_Default8x8./qScale);
%     end
%     if(size(dct_block,1)==16)
%         quant = round(dct_block./quant_Default16x16./qScale);
%     end
%     if(size(dct_block,1)==32)
%         quant = round(dct_block./quant_Default32x32./qScale);
%     end
    
    
    quant_Default4x4L =[16,16,24,51
                       16,24,61,61
                       24,61,99,99
                       51,99,99,99];    
    quant_Default4x4C =[16,24,99,99
                       24,24,99,99
                       24,99,99,99
                       99,99,99,99];  
    L8x8 = [16 11 10 16 24 40 51 61;
        12 12 14 19 26 58 60 55;
        14 13 16 24 40 57 69 56;
        14 17 22 29 51 87 80 62;
        18 55 37 56 68 109 103 77;
        24 35 55 64 81 104 113 92;
        49 64 78 87 103 121 120 101;
        72 92 95 98 112 100 103 99];
    C8x8 = [17 18 24 47 99 99 99 99;
        18 21 26 66 99 99 99 99;
        24 13 56 99 99 99 99 99;
        47 66 99 99 99 99 99 99;
        99 99 99 99 99 99 99 99;
        99 99 99 99 99 99 99 99;
        99 99 99 99 99 99 99 99;
        99 99 99 99 99 99 99 99];
    L16x16 = kron(L8x8,ones(2,2));
    L32x32 = kron(L8x8,ones(4,4));
    C16x16 = kron(C8x8,ones(2,2));
    C32x32 = kron(C8x8,ones(4,4));
    if(size(dct_block,1)==4)
        quant(:,:,1) = round(dct_block(:,:,1)./quant_Default4x4L./qScale);
        quant(:,:,2:3) = round(dct_block(:,:,2:3)./quant_Default4x4C./qScale);
    end
    if(size(dct_block,1)==8)
        quant(:,:,1) = round(dct_block(:,:,1)./L8x8./qScale);
        quant(:,:,2:3) = round(dct_block(:,:,2:3)./C8x8./qScale);
    end
    if(size(dct_block,1)==16)
        quant(:,:,1) = round(dct_block(:,:,1)./L16x16./qScale);
        quant(:,:,2:3) = round(dct_block(:,:,2:3)./C16x16./qScale);
    end
    if(size(dct_block,1)==32)
        quant(:,:,1) = round(dct_block(:,:,1)./L32x32./qScale);
        quant(:,:,2:3) = round(dct_block(:,:,2:3)./C32x32./qScale);
    end
end
function [TUs_tree_code_flat,mode_code_flat,ZeroRun_code_flat,ZeroRun_code_flat_different_layers,unit_index] = TUs_encode(TUs_layers,TUslayers_mode,modes_ZeroRun_codes)
    global TUs_levels_recursive TUslayers_mode_recursive modes_ZeroRun_codes_recursive deepth 
    TUs_levels_recursive = TUs_layers;
    TUslayers_mode_recursive = TUslayers_mode;
    modes_ZeroRun_codes_recursive = modes_ZeroRun_codes;
    deepth = size(TUslayers_mode,2);
    global TUs_tree_code_flat_recursive mode_code_flat_recursive ZeroRun_code_flat_recursive ZeroRun_code_flat_different_layers_recursive unit_index_recursive
    TUs_tree_code_flat_recursive = [];
    mode_code_flat_recursive = [];
    ZeroRun_code_flat_recursive = [];
    ZeroRun_code_flat_different_layers_recursive = cell(1,size(TUslayers_mode,2));
    unit_index_recursive = [];
    for i = 1 : size(TUs_layers{1},1)
        for j = 1 : size(TUs_layers{1},2) 
            TU_encode_recursive(1,i,j);
        end
    end
    TUs_tree_code_flat = TUs_tree_code_flat_recursive;
    mode_code_flat = mode_code_flat_recursive;
    unit_index = unit_index_recursive;
    ZeroRun_code_flat = ZeroRun_code_flat_recursive;
    ZeroRun_code_flat_different_layers = ZeroRun_code_flat_different_layers_recursive;
end
function  TU_encode_recursive(level,i,j)
    global TUs_levels_recursive TUslayers_mode_recursive modes_ZeroRun_codes_recursive deepth
    global TUs_tree_code_flat_recursive mode_code_flat_recursive ZeroRun_code_flat_recursive ZeroRun_code_flat_different_layers_recursive unit_index_recursive
    if level == deepth
        %TUs_tree_code_block = [];
        mode_code_flat_recursive = [mode_code_flat_recursive TUslayers_mode_recursive{level}(i,j)];
        unit_index_recursive = [unit_index_recursive [level;i;j]];
        ZeroRun_code_flat_recursive = [ZeroRun_code_flat_recursive modes_ZeroRun_codes_recursive{TUslayers_mode_recursive{level}(i,j)}{level}{i,j}];
        ZeroRun_code_flat_different_layers_recursive{level} = [ZeroRun_code_flat_different_layers_recursive{level} modes_ZeroRun_codes_recursive{TUslayers_mode_recursive{level}(i,j)}{level}{i,j}];
    else
        if TUs_levels_recursive{level}(i,j) == 1
            TUs_tree_code_flat_recursive = [TUs_tree_code_flat_recursive 1];
            TU_encode_recursive(level+1,2*i-1,2*j-1);
            TU_encode_recursive(level+1,2*i-1,2*j);
            TU_encode_recursive(level+1,2*i,2*j-1);
            TU_encode_recursive(level+1,2*i,2*j);
        else
            TUs_tree_code_flat_recursive = [TUs_tree_code_flat_recursive 0];
            mode_code_flat_recursive = [mode_code_flat_recursive TUslayers_mode_recursive{level}(i,j)];
            unit_index_recursive = [unit_index_recursive [level;i;j]];
            ZeroRun_code_flat_recursive = [ZeroRun_code_flat_recursive modes_ZeroRun_codes_recursive{TUslayers_mode_recursive{level}(i,j)}{level}{i,j}];
            ZeroRun_code_flat_different_layers_recursive{level} = [ZeroRun_code_flat_different_layers_recursive{level} modes_ZeroRun_codes_recursive{TUslayers_mode_recursive{level}(i,j)}{level}{i,j}];
        end
    end
end
function dct_block = DeQuant_inter(quant_block, qScale)
%  Function Name : DeQuant8x8.m
%  Input         : quant_block  (Quantized Block, 8x8x3)
%                  qScale       (Quantization Parameter, scalar)
%
%  Output        : dct_block    (Dequantized DCT coefficients, 8x8x3)
    quant_Default4x4L =[16,16,24,51
                       16,24,61,61
                       24,61,99,99
                       51,99,99,99];    
    quant_Default4x4C =[16,24,99,99
                       24,24,99,99
                       24,99,99,99
                       99,99,99,99];               
    L8x8 = [16 11 10 16 24 40 51 61;
        12 12 14 19 26 58 60 55;
        14 13 16 24 40 57 69 56;
        14 17 22 29 51 87 80 62;
        18 55 37 56 68 109 103 77;
        24 35 55 64 81 104 113 92;
        49 64 78 87 103 121 120 101;
        72 92 95 98 112 100 103 99];
    C8x8 = [17 18 24 47 99 99 99 99;
        18 21 26 66 99 99 99 99;
        24 13 56 99 99 99 99 99;
        47 66 99 99 99 99 99 99;
        99 99 99 99 99 99 99 99;
        99 99 99 99 99 99 99 99;
        99 99 99 99 99 99 99 99;
        99 99 99 99 99 99 99 99];
    L16x16 = kron(L8x8,ones(2,2));
    L32x32 = kron(L8x8,ones(4,4));
    C16x16 = kron(C8x8,ones(2,2));
    C32x32 = kron(C8x8,ones(4,4));
    if(size(quant_block,1)==4)
        dct_block(:,:,1) = qScale.*quant_Default4x4L.*quant_block(:,:,1);
        dct_block(:,:,2:3) = qScale.*quant_Default4x4C.*quant_block(:,:,2:3);
    end
    if(size(quant_block,1)==8)
        dct_block(:,:,1) = qScale.*L8x8.*quant_block(:,:,1);
        dct_block(:,:,2:3) = qScale.*C8x8.*quant_block(:,:,2:3);
    end
    if(size(quant_block,1)==16)
        dct_block(:,:,1) = qScale.*L16x16.*quant_block(:,:,1);
        dct_block(:,:,2:3) = qScale.*C16x16.*quant_block(:,:,2:3);
    end
    if(size(quant_block,1)==32)
        dct_block(:,:,1) = qScale.*L32x32.*quant_block(:,:,1);
        dct_block(:,:,2:3) = qScale.*C32x32.*quant_block(:,:,2:3);
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


function [ BinaryTree, HuffCode, BinCode, Codelengths] = buildHuffman( p )

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
function zze = ZeroRunEnc_EoB(zz, EOB,n)
%  Input         : zz (Zig-zag scanned sequence, 1xN)
%                  EOB (End Of Block symbol, scalar)
%
%  Output        : zze (zero-run-level encoded sequence, 1xM)
    n_2 = n*n;
    processing_zero = false;
    zze = zeros(1,length(zz)*2);
    j = 1;
    for i = 1:length(zz)
            if(processing_zero == false)
                if(zz(i)==0)
                    processing_zero = true;
                    last_continous_zero_position = i;
                    if mod(i,n_2)==0
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
                    if mod(i,n_2)==0
                        zze(j) = EOB;
                        j = j+1;
                        processing_zero = false;
                     end
                end
            end

    end
    zze = zze(1:j-1);
end
function [TUslayers_mode,TUs_layers] = cost_analyse_with_modes(modes_MSE_block,mode_bitPerPixel_block,lambda)
    
    %% a simple layers    
    deepth = size(modes_MSE_block{1},2);
    COST_with_best_mode = cell(1,deepth);
    TUslayers_mode= cell(1,deepth);% mode 1 use normal mode 2 use quantized mcp
    for i = 1:deepth
        clear modes_COSTs 
        for ii = 1:size(modes_MSE_block,2)
            modes_COSTs(:,:,ii) = modes_MSE_block{ii}{i}+lambda*mode_bitPerPixel_block{ii}{i};
        end
        [COST_with_best_mode{i},TUslayers_mode{i}]=min(modes_COSTs,[],3);
    end
    
    TUs_layers = cell(1,deepth-1);
    COST_from_above_layers = COST_with_best_mode{deepth};
    for i = (deepth-1):-1:1
        [rr,cc] = size(COST_from_above_layers);
        COST_from_above_layers = kron(eye(rr/2),[1/2 1/2])*COST_from_above_layers*kron(eye(cc/2),[1/2;1/2]);
        COST_this_layer = COST_with_best_mode{i};
        TUs_layers{i} = COST_this_layer > COST_from_above_layers ;
        COST_from_above_layers = min(COST_from_above_layers,COST_this_layer);
    end    
    

end
function coeffs = DeZigZag(zz,n)
%  Function Name : DeZigZag8x8.m
%  Input         : zz    (Coefficients in zig-zag order)
%
%  Output        : coeffs(DCT coefficients in original order)
    i = 1;
    j = 1;
    k = 1;
    direction = 0;
    % direction = 0 means go right up
    % direction = 1 means go left down
    coeffs = zeros(n,n,size(zz,2));
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
function dst = ZeroRunDec_EoB(src, EoB, n)
%  Function Name : ZeroRunDec1.m zero run level decoder
%  Input         : src (zero run encoded sequence 1xM with EoB signs)
%                  EoB (end of block sign)
%
%  Output        : dst (reconstructed zig-zag scanned sequence 1xN)
    n_2 = n*n;
    dst = zeros(1,n_2);
    i = 1;
    j = 1;
    while(1)
        if(src(i)==0)
            j = j+src(i+1);
            if j > length(dst)
                dst = [dst  zeros(1,n_2)];
            end
            i = i+2;
            j = j+1;
        elseif src(i)==EoB
            j = ceil(j/n_2)*n_2;
            if j > length(dst)
                dst = [dst  zeros(1,n_2)];
            end
            i = i+1;
            j = j+1;
        else
            if j > length(dst)
                dst = [dst  zeros(1,n_2)];
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
function PSNR = calcPSNR(Image, recImage)
% Input         : Image    (Original Image)
%                 recImage (Reconstructed Image)
%
% Output        : PSNR     (Peak Signal to Noise Ratio)
% YOUR CODE HERE
% call calcMSE to calculate MSE
    PSNR = 10*log10((2^8-1)^2/calcMSE(Image, recImage));
end
function MSE = calcBlockMSE(Image, recImage,n)
% Input         : Image    (Original Image)
%                 recImage (Reconstructed Image)
% Output        : MSE      (Mean Squared Error)
% YOUR CODE HERE
    [r,c,~] = size(Image);
    MSE = zeros(r/n,c/n);
    for i = 1:r/n
        for j = 1:c/n 
            MSE(i,j) = calcMSE(Image(i*n-n+1:i*n,j*n-n+1:j*n,:),recImage(i*n-n+1:i*n,j*n-n+1:j*n,:));
        end
    end
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
function image = dct2image(dct,n)
    [r,c,~] = size(dct);
    idct_coeff = get_idct_coeff(n);
    col_idct_coeff = kron(eye(r/n),idct_coeff);
    row_idct_coeff = kron(eye(c/n),idct_coeff);
    image = zeros(size(dct));
    for yuv = 1:3
        image(:,:,yuv) = col_idct_coeff*dct(:,:,yuv)*(row_idct_coeff');
    end
end
function dct = image2dct(image,n)
    [r,c,~] = size(image);
    dct_coeff = get_dct_coeff(n);
    col_dct_coeff = kron(eye(r/n),dct_coeff);
    row_dct_coeff = kron(eye(c/n),dct_coeff);
    dct = zeros(size(image));
    for yuv = 1:3
        dct(:,:,yuv) = col_dct_coeff*image(:,:,yuv)*(row_dct_coeff');
    end
end
function yuv = ictRGB2YCbCr(rgb)
% Input         : rgb (Original RGB Image)
% Output        : yuv (YCbCr image after transformation)
% YOUR CODE HERE
    size_rgb = size(rgb);
    tran = [0.299 0.587 0.114;-0.169 -0.331 0.5;0.5 -0.419 -0.081];
    yuv=reshape((tran*squeeze(reshape(rgb,[],3))')',size_rgb);
end
function MSE = calcMSE(Image, recImage)
% Input         : Image    (Original Image)
%                 recImage (Reconstructed Image)
% Output        : MSE      (Mean Squared Error)
% YOUR CODE HERE
    [l,w,d] = size(Image);
    MSE = 1/d/l/w*sum((double(Image)-double(recImage)).^2,'all');
end
function rgb = ictYCbCr2RGB(yuv)
% Input         : yuv (Original YCbCr image)
% Output        : rgb (RGB Image after transformation)
% YOUR CODE HERE
    size_yuv = size(yuv);
    tran = [1 0 1.402;1 -0.344 -0.714;1 1.772 0];
    rgb=reshape((tran*squeeze(reshape(yuv,[],3))')',size_yuv);
end
function [bytes,maxi] = bits2bytes(bits,n)
    bytes = zeros(1,ceil(numel(bits)/n));
    for i = 1:n
        g = bits(i:n:end);
        g = [g zeros(1,numel(bytes)-numel(g))];
        bytes = bytes+g*2^(n-i);
    end
    maxi = 0;
    for i = 1:n
        maxi = maxi+2^(i-1);
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
function bits = bytes2bits(bytes,n)

    bits = zeros(1,numel(bytes)*n);
    for i = 1:n
        bits(i:n:end) = bytes>=2^(n-i);
        bytes = bytes-2^(n-i).*bits(i:n:end);
    end


end
function pmf = stats_marg(image, range)
    pmf = hist(squeeze(reshape(image,[],1,1)),range);
    pmf = double(pmf)/sum(pmf,'all');
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
function zz = ZigZag_slow(quant,n)
%  Input         : quant (Quantized Coefficients, 8x8x3)
% 
%  Output        : zz (zig-zag scaned Coefficients, 64x3)
    i = 1;
    j = 1;
    k = 1;
    direction = 0;
    % direction = 0 means go right up
    % direction = 1 means go left down
    zz = zeros(size(quant,1)*size(quant,2),size(quant,3));
    while(1)
        zz(k,:) = quant(i,j,:);
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
function block = IDCT(coeff)
%  Function Name : IDCT8x8.m
%  Input         : coeff (DCT Coefficients) 8*8*3
%  Output        : block (original image block) 8*8*3
    if length(size(coeff)) ==3
        block = zeros(size(coeff));
        for i  = 1:size(coeff,3)
            block(:,:,i) = idct(idct(coeff(:,:,i))')';     
        end
    else
        block = idct(idct(coeff)')';
    end
end

% get_dct_coeff(N)
function coeff = DCT(block)
    %  Input         : block    (Original Image block, 8x8x3)
    %
    %  Output        : coeff    (DCT coefficients after transformation, 8x8x3)
    if length(size(block)) ==3
        coeff = zeros(size(block));
        for i  = 1:size(block,3)
            coeff(:,:,i) = dct(dct(block(:,:,i))')';     
        end
    else
        coeff = dct(dct(block)')';
    end
end



