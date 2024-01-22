clc; clear all; close all;
%% READ Images
I = imread("./left000012.png");
%% Edge Detection Using Fuzzy Logic

% Create Fuzzy Inference System (FIS)
edgeFIS = mamfis('Name', 'edgeDetection');

% Specify inputs: image gradients are used as inputs
edgeFIS = addInput(edgeFIS, [-3.5,3.5], 'Name', 'Ix');
edgeFIS = addInput(edgeFIS, [-3.5,3.5], 'Name', 'Iy');

% Specify membership function for each input
sx = 0.1; % std
mux = 0; % mean
sy = 0.1; % std
muy = 0; % mean
edgeFIS = addMF(edgeFIS, 'Ix', 'gaussmf', [sx mux], 'Name', 'zero');
edgeFIS = addMF(edgeFIS, 'Iy', 'gaussmf', [sy muy], 'Name', 'zero');

% Specify output
edgeFIS = addOutput(edgeFIS, [0,1], 'Name', 'Iout');

% Specify membership function for output
% Original
whiteval = [0.1, 0.7, 1];
blackval = [0, 0, 0.7];
% Switch black and white
% whiteval = [0.3, 1, 1];
% blackval = [0, 0, 0.9];

edgeFIS = addMF(edgeFIS, 'Iout', 'trimf', whiteval, 'Name', 'edge');
edgeFIS = addMF(edgeFIS, 'Iout', 'trimf', blackval, 'Name', 'non-edge');


tr = tiledlayout(1,2);

nexttile
plotmf(edgeFIS, 'input', 1)
set(gca,'FontName','times','FontSize',12,'FontWeight','bold')

nexttile
plotmf(edgeFIS, 'output', 1)
set(gca,'FontName','times','FontSize',12,'FontWeight','bold')

tr.TileSpacing = 'compact';
tr.Padding = 'compact';

% title('Iout')

% Specify Rules
% r1 = "If Ix is zero and Iy is zero then Iout is white";
% r2 = "If Ix is not zero or Iy is not zero then Iout is black";
% r1 = "Ix==zero & Iy==zero => Iout=white";
% r2 = "Ix~=zero | Iy~=zero => Iout=black";
r1 = "Ix==zero & Iy==zero => Iout=non-edge";
r2 = "Ix~=zero | Iy~=zero => Iout=edge";
edgeFIS = addRule(edgeFIS, [r1, r2]);
edgeFIS.Rules



%% Preprocessing - modified
imagefiles=dir(fullfile(pwd,'*.png'));
nfiles = length(imagefiles);

%h = waitbar(0,' Time Loop: Fuzzy based edge detection');

 for i=1:nfiles
   currentfilename = fullfile(pwd, imagefiles(i).name);
   currentimage = imread(currentfilename);
   Irgb = currentimage;

    % Convert to Grayscale from RGB
    Igray = rgb2gray(Irgb);
    
    GF = sqrt(12);
    
    % Convert to double-precision data
    I = im2double(Igray);
    
    % Gaussian filter
    I = imgaussfilt(I, GF);
    
    % Obtain Image Gradient
    [Ix, Iy] = imgradientxy(I,'sobel');
    % Other filters can be used to obtain image gradients
    % Such as: Sobel, Prewitt
    % Functions: imfilter, imgradientxy, imgradient
    % Evaluate FIS
    Ieval = zeros(size(I));
    for ii = 1:size(I,1)
        Ieval(ii,:) = evalfis(edgeFIS, [(Ix(ii,:));(Iy(ii,:))]');
    end
%     save_filename = fullfile(pwd,strcat(imagefiles(i).name(1:end-3),'png'));
%     imwrite(Ieval,save_filename);
    
%    waitbar(i/nfiles,h);
 end
%%
%% Preprocessing - modified
imagefiles=dir(fullfile(pwd,'*.png'));
nfiles = length(imagefiles);

%h = waitbar(0,' Time Loop: Fuzzy based edge detection');

 for i=1:nfiles
     i
   currentfilename = fullfile(pwd, imagefiles(i).name);
   currentimage = imread(currentfilename);
   Irgb = currentimage;

    % Convert to Grayscale from RGB
    Igray = rgb2gray(Irgb);
    
    GF = sqrt(12);
    
    % Convert to double-precision data
    I = im2double(Igray);
    
    % Gaussian filter
    I = imgaussfilt(I, GF);
    
    % Obtain Image Gradient
    [Ix, Iy] = imgradientxy(I,'sobel');
    % Other filters can be used to obtain image gradients
    % Such as: Sobel, Prewitt
    % Functions: imfilter, imgradientxy, imgradient
    % Evaluate FIS
    Ieval = zeros(size(I));
    for ii = 1:size(I,1)
        Ieval(ii,:) = evalfis(edgeFIS, [(Ix(ii,:));(Iy(ii,:))]');
    end
%     save_filename = fullfile(pwd,strcat(imagefiles(i).name(1:end-3),'png'));
%     imwrite(Ieval,save_filename);
    
%    waitbar(i/nfiles,h);
 end

%%
% Ieval = imread("./calibration/after/results/saved_pypylon_img_zlns.png");
% I = Ieval;

%%


radiiRange = [8 40]; % 예를 들어, 20에서 50 픽셀 사이의 반지름을 가진 원을 찾습니다.

%% gray 영상 변환이후, 원 찾고, 영상 cropping 및 warping
% 원 찾기
[centers, radii] = imfindcircles(Ieval, radiiRange, 'ObjectPolarity','bright', 'Sensitivity',0.8);

% 찾은 원 표시
figure;
imshow(Ieval);
title('Fuzzy Image');% 원의 반지름 범위 설정
viscircles(centers, radii,'EdgeColor','b');
hold on;plot(centers(:,1), centers(:,2), 'r.', 'MarkerSize', 10);
title('Detected Circles');

%% raidus 값을 바탕으로 filtering
mean_radius = mean(radii);
idx = (radii >= mean_radius*0.80) & (radii <=mean_radius*1.20);
filteredCenters = centers(idx,:);
filteredRadii = radii(idx);

figure;
imshow(Ieval);
title('Fuzzy Image');% 원의 반지름 범위 설정
viscircles(filteredCenters, filteredRadii,'EdgeColor','b');
hold on;plot(filteredCenters(:,1), filteredCenters(:,2), 'r.', 'MarkerSize', 10);
title('Detected Circles');


%% 두 centers사이의 거리가 가까운 75개를 filtering
distances = pdist(filteredCenters);
% 거리 행렬을 squareform으로 변환하여 각 점 사이의 거리를 2차원 행렬로 변환
distMatrix = squareform(distances);

% distMatrix의 대각선은 0이므로, 대각선 요소를 Inf로 설정하여 최소값 계산에서 제외
distMatrix(logical(eye(size(distMatrix)))) = Inf;

% 각 점에 대해 가장 가까운 점까지의 거리를 찾음
[minDistances, ~] = min(distMatrix, [], 2);

% minDistances를 기준으로 centers와 radii를 정렬
[sortedDistances, sortIndex] = sort(minDistances);

% 가장 가까운 점들부터 상위 75개를 선택
topCenters = filteredCenters(sortIndex(1:75), :);
topRadii = filteredRadii(sortIndex(1:75));

% 선택된 원들을 원본 이미지에 그리기
figure;
imshow(Ieval); % 원본 이미지를 표시
viscircles(topCenters, topRadii, 'EdgeColor', 'r'); % 선택된 원들을 빨간색으로 표시
% 각 원에 번호를 부여
for i = 1:length(topCenters)
    text(topCenters(i,1), topCenters(i,2), num2str(i), 'Color', 'y', ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
end

hold off; % 축 유지를 해제

%% 원의 순서를 sorting
% step 1: X, Y 좌표계 값에 따라 정량 (처음 5개정도는 sorting이 됨)->step 2를 위하여 rotation angle 계산하는 용도
% centers 배열을 x 좌표에 따라 정렬
[~, sortIdx] = sortrows(topCenters, [2, 1]); % 여기서 [2, 1]은 먼저 y 좌표에 따라 정렬하고, 그 다음 x 좌표에 따라 정렬
sortedCenters = topCenters(sortIdx, :);

% 이미지 표시
imshow(Ieval);


% 원 그리기
viscircles(sortedCenters, topRadii(sortIdx), 'EdgeColor', 'r');

% 번호 텍스트 오버레이
for i = 1:length(sortedCenters)
    text(sortedCenters(i,1), sortedCenters(i,2), sprintf('%d', i), 'Color', 'y', ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
end



% step 2: 영상 회전후 sorting
% 정렬된 원 중심들 중 처음 5개를 선택하여 영상 회전 (STEP 2-1)
firstFiveCenters = sortedCenters(1:5, :);

% 처음 5개의 원 중심에 대한 선형 회귀 수행
p = polyfit(firstFiveCenters(:,1), firstFiveCenters(:,2), 1);

% 기울기로부터 각도 계산 (아크탄젠트 사용)
angleRad = atan(p(1));
angleDeg = rad2deg(angleRad);

% 회전 행렬 생성
R = [cos(-angleRad) -sin(-angleRad); sin(-angleRad) cos(-angleRad)];

% 모든 원 중심 좌표 변환 (회전)
rotatedCenters = (R * (sortedCenters - mean(firstFiveCenters))')' + mean(firstFiveCenters);

% 이미지 표시
figure;
imshow(Ieval);
hold on;

% 원 그리기
viscircles(rotatedCenters, topRadii(sortIdx), 'EdgeColor', 'r');

% 번호 텍스트 오버레이
for i = 1:length(rotatedCenters)
    text(rotatedCenters(i,1), rotatedCenters(i,2), sprintf('%d', i), 'Color', 'y', ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
end

hold off;

% STEP 2-2: 회전된 영상을 바탕으로 SORTING
% 임계값 설정 (이 값을 조절하여 줄을 구분)
[sortedY, sortIndex] = sort(rotatedCenters(:,2), 'ascend');
y_threshold1 = (sortedY(25) + sortedY(26)) / 2; % 첫 번째와 두 번째 줄을 구분하는 y 좌표의 값
y_threshold2 = (sortedY(50) + sortedY(51)) / 2; % 두 번째와 세 번째 줄을 구분하는 y 좌표의 값

% 각 줄의 원들을 분리
firstRow = rotatedCenters(rotatedCenters(:,2) < y_threshold1, :);
secondRow = rotatedCenters(rotatedCenters(:,2) >= y_threshold1 & rotatedCenters(:,2) < y_threshold2, :);
thirdRow = rotatedCenters(rotatedCenters(:,2) >= y_threshold2, :);

% 각 줄을 x 좌표에 따라 정렬
[~, firstOrder] = sort(firstRow(:, 1), 'ascend');
[~, secondOrder] = sort(secondRow(:, 1), 'ascend');
[~, thirdOrder] = sort(thirdRow(:, 1), 'ascend');

firstSorted = firstRow(firstOrder, :);
secondSorted = secondRow(secondOrder, :);
thirdSorted = thirdRow(thirdOrder, :);

% 모든 줄의 원들을 하나의 배열로 병합
sortedCenters = [firstSorted; secondSorted; thirdSorted];

% 이미지 표시
figure;
imshow(Ieval);
hold on;

% 병합된 배열로 원 그리기 및 번호 텍스트 오버레이
for i = 1:size(sortedCenters, 1)
    viscircles(sortedCenters(i,:), topRadii(i), 'EdgeColor', 'r');
    text(sortedCenters(i,1), sortedCenters(i,2), sprintf('%d', i), 'Color', 'y', ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
end

hold off;


%% 다시 rotation을 회복시켜서 결과 정리, 최종 결과를 excel창에 기록
R_inverse = R';

% 평균 중심점으로 이동하기 전의 원래 위치로 되돌리기
originalCenters = (R_inverse * (sortedCenters - mean(firstFiveCenters))')' + mean(firstFiveCenters);

% 이미지 표시
figure;
imshow(Ieval);
hold on;

% 원래 위치에 원 그리기 및 번호 텍스트 오버레이
for i = 1:size(originalCenters, 1)
    viscircles(originalCenters(i,:), topRadii(i), 'EdgeColor', 'r');
    text(originalCenters(i,1), originalCenters(i,2), sprintf('%d', i), 'Color', 'y', ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
end

hold off;
csvwrite('./file.csv', [originalCenters, topRadii]);


