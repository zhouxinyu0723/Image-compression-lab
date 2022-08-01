1. folder structure:
chapter 5:
-folder
--Xinyu_Zhou_chapter5_submission.m
--lena_small.tif
--foreman20_40_RGB
---foreman0020.bmp
---foreman0022.bmp
          .
          .
          .

chapter 6:
-folder
--Xinyu_Zhou_chapter6_submission.m
--foreman20_40_RGB
---foreman0020.bmp
---foreman0022.bmp
          .
          .
          .

2. to run the code, run directly

3. To change the path to the folder of the sequences:
Xinyu_Zhou_chapter5_submission.m:
edit line 3:  folder_path = strcat('foreman20_40_RGB');
edit line 117: I = double(imread('lena_small.tif'));%image_paths{1}

Xinyu_Zhou_chapter6_submission.m:
edit line 3:  folder_path = strcat('foreman20_40_RGB');

4. to calculate the bpp in chapter 6. A variable byte_count is used.
It will be accumulated in
line 37 for the encode of the first image
line 56 for the encode of the first motion vector
line 109 for the encode of the first PU trees
line 123 for the encode of the first mode code
line 134 for the encode of the first residual
line 168 for the encode of the rest of motion vector
line 199 for the encode of the first PU trees
line 209 for the encode of the first mode code
line 200 for the encode of the first residual

5. PSNR is calculated in
line 44 for the first image
line 159 for the second image
line 244 for the rest images

6. when the code is running, it will create some folders to store the reconstructed images in and other images which helps debug.

7. My previous Submission of chapter 5 has some errors. I have correct them. Especially I did not use lena_small.tif to train the haffman code for the individual coding.
