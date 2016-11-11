%%
%get mfcc coeffs

clear
addpath feature_extraction mfcc
listing_music = dir('music_wav');
listing_speech = dir('speech_wav');
listing_music(1:2) = []
listing_speech(1:2) = []

music_allMFCCs = zeros(13*1500, 64);
speech_allMFCCs = zeros(13*1500, 64);

for i=1:64    
    filename1 = strcat('music_wav/',listing_music(i).name);
    filename2 = strcat('speech_wav/',listing_speech(i).name);

    Tw = 20;           % analysis frame duration (ms)
    Ts = 20;           % analysis frame shift (ms)
    alpha = 0.97;      % preemphasis coefficient
    R = [ 300 3700 ];  % frequency range to consider
    M = 20;            % number of filterbank channels
    C = 13;            % number of cepstral coefficients
    L = 22;            % cepstral sine lifter parameter
    
    % hamming window (see Eq. (5.2) on p.73 of [1])
    hamming = @(N)(0.54-0.46*cos(2*pi*(0:N-1).'/(N-1)));
    
    % Read speech samples, sampling rate and precision from file
    [ speech, fs] = audioread(filename1);
    
    % Feature extraction (feature vectors as columns)
    [ music_allMFCCs_temp, FBEs, frames ] = ...
        mfcc( speech, fs, Tw, Ts, alpha, hamming, R, M, C, L );
    music_allMFCCs(:,i) = reshape(music_allMFCCs_temp,[13*1500,1]);
    [ speech, fs] = audioread(filename2);
    [ speech_allMFCCs_temp, FBEs, frames ] = ...
        mfcc( speech, fs, Tw, Ts, alpha, hamming, R, M, C, L );
    speech_allMFCCs(:,i) = reshape(speech_allMFCCs_temp,[13*1500,1]);
    
    
    
    % Plot cepstrum over time
%     figure('Position', [30 100 800 200], 'PaperPositionMode', 'auto', ...
%         'color', 'w', 'PaperOrientation', 'landscape', 'Visible', 'on' );
%     
%     imagesc( 1:size(MFCCs,2), (0:C-1), MFCCs );
%     axis( 'xy' );
%     xlabel( 'Frame index' );
%     ylabel( 'Cepstrum index' );
%     title( 'Mel frequency cepstrum' );
%     
    
    %FF = computeAllStatistics(filename, Tw*10^3, Ts*10^3);

end

%%
%training 8-fold cross-validation

%cl =[];
species = zeros(128/8,8);
scram = randperm(128);
Y = ones(128,1);
Y(64:end) = -1;
X = [music_allMFCCs' ; speech_allMFCCs'];
Xs = X(scram,:);
Ys = Y(scram);
error = 0;
k_fold = 8;
for i = 1:k_fold
    param = 128/k_fold;
    Y_train = [Ys(1:(i-1)*(param)); Ys(param*(i)+1:end)];
    Y_test = Ys((i-1)*param+1:i*param);

    X_train = [Xs(1:(i-1)*(param),:); Xs(param*(i)+1:end,:)];
    X_test = Xs((i-1)*param+1:i*param,:);
    
    cl = svmtrain(X_train,Y_train);
    species(:,i) = svmclassify(cl,X_test);
    error = error+ sum(Y_test(:)~=species(:,i))/length(Y_test);
end
error/k_fold