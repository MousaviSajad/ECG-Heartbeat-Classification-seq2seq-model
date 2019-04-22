clear all
clc
tic
addr = '.\mitbihdb';
Files=dir(strcat(addr,'\*.mat'));

%% Translate PhysioNet classification results to AAMI and AAMI2 labling schemes
% AAMI Classes:
% % N = N, L, R, e, j
% % S = A, a, J, S
% % V = V, E
% % F = F
% % Q = /, f, Q

% AAMI2 Classes:
% % N = N, L, R, e, j
% % S = A, a, J, S
% % V = V, E, F
% % Q = /, f, Q
% https://github.com/ehendryx/deim-cur-ecg/blob/master/DS1_MIT_CUR_beat_classification.m
AAMI_annotations = {'N' 'S' 'V' 'F' 'Q'};
AAMI2_annotations = {'N' 'S' 'V_hat' 'Q'};

index = 1;
beat_len = 280;
n_cycles = 0;
featuresSeg = [];
groupN = [];
groupV = [];
groupS = [];
groupF = [];
groupQ = [];
N_class = 0;V_class=0;F_class=0;Q_class=0;S_class=0;
for i=1:length(Files) % Files names3signals
    
    %% load the files
    % load ('100m.mat')          % the signal will be loaded to "val" matrix
    % val = (val - 1024)/200;    % you have to remove "base" and "gain"
    % ECGsignal = val(1,1:1000); % select the lead (Lead I)
    % Fs = 360;                  % sampling frequecy
    % t = (0:length(ECGsignal)-1)/Fs;  % time
    % plot(t,ECGsignal)
    %
    
    [pathstr,name,ext] = fileparts(Files(i).name);
    nsig = 1;
    
    [tm,ecgsig,ann,Fs,sizeEcgSig,timeEcgSig] = loadEcgSig([addr filesep name]);
    
    signal = ecgsig(nsig,:);
    
    
    
    %%
    %     rPeaks = rDetection(signal, Fs);
    %     rPeaks = get_rpeaks(signal, Fs);
    rPeaks  = cell2mat(ann(3))+1;
    n_cycles = n_cycles + length(rPeaks);
    %     [R_i,R_amp,S_i,S_amp,T_i,T_amp,Q_i,Q_amp] = peakdetect(signal,Fs);
    %      rPeaks =  R_i;
    rPeaks = double(rPeaks);
    
    peaks = qsPeaks(signal, rPeaks, Fs);
    tpeaks = peaks(:,7);  
    
    %    %% Plot P Q R S T points
    %     N = length(signal);
    %     tm = 1/Fs:1/Fs:N/Fs;
    %     figure;plot(tm,signal);hold on
    %     scatter(peaks(:,1)/Fs,signal(peaks(:,1)),'g*') % P points
    %     scatter(peaks(:,3)/Fs,signal(peaks(:,3)),'k+') % Q points
    %     scatter(peaks(:,4)/Fs,signal(peaks(:,4)),'ro') % R points
    %     scatter(peaks(:,5)/Fs,signal(peaks(:,5)),'c^') % S points
    %     scatter(peaks(:,7)/Fs,signal(peaks(:,7)),'mo') % T points
    %     xlabel('Seconds'); ylabel('Amplitude')
    %     title('ECG peaks detection')
    %     legend('Raw signal','P','Q','R','S','T')
    %     hold off
    %
    
    %% grouping
    % gourp 0: N(normal and bundle branch block beats); group 2: V(ventricular
    %ectopic beats); group 1: S(supraventricular ectopic beats); group 3: F (fusion of N and V beats)
    % group Q:4 unknown beat
    % consider just absolute features, where each row of extraxted features is
    % related to one segment
    
    annots_list = ['N','L','R','e','j','S','A','a','J','V','E','F','/','f','Q'];
    
    annot  = cell2mat(ann(4));
    indices  = ismember(rPeaks,peaks(:,4));
    annot = annot(indices);
    % rps = peaks(:,4);
    
    % AAMI Classes:
    % % N = N, L, R, e, j
    % % S = A, a, J, S
    % % V = V, E
    % % F = F
    % % Q = /, f, Q
    
    seg_values = {};
    seg_labels =[];  
    
    ind_seg = 1;
    % normalize
    signal = normalize(signal);
    for ind=1:length(annot)
        if ~ismember(annot(ind),annots_list)
            continue;
        end
        
        N_g = ['N', 'L', 'R', 'e', 'j'];%0
        S_g = ['A', 'a', 'J', 'S'];%1
        V_g = ['V', 'E'];%2
        F_g = ['F'];%3
        Q_g = [' /', 'f', 'Q'];%4
        if(ismember(annot(ind),N_g))
              lebel = 'N';
%              if(N_class >8031) %(N_class >8031)
%                 continue
%              end
          
            
        elseif(ismember(annot(ind),S_g))
            lebel = 'S';
        elseif(ismember(annot(ind),V_g))
            lebel = 'V';
        elseif(ismember(annot(ind),F_g))
            lebel = 'F';
        elseif(ismember(annot(ind),Q_g))
            lebel = 'Q';
        else
            throw("No label! :(")
            
        end
        
        if ind==1
            
            seg_values{ind_seg} = signal(1:tpeaks(ind)-1)';
            t_sig = imresize(seg_values{ind_seg}(1:min(Fs,length(seg_values{ind_seg}))), [beat_len 1]);
            seg_values{ind_seg} = t_sig;
            seg_labels(ind_seg) = lebel;
            % plot(cell2mat(seg_values(ind_seg)))
            ind_seg = ind_seg+1;
            continue;
        end
        t_sig = imresize(signal(tpeaks(ind-1):tpeaks(ind)-1)', [beat_len 1]);
        seg_values{ind_seg} =t_sig ;
        %     figure;
        %      plot(cell2mat(seg_values(ind_seg)))
        % determine the label
        
        
        seg_labels(ind_seg) =  lebel;
        ind_seg = ind_seg+1;
        
    end
    s2s_mitbih(i).seg_values = seg_values';
    s2s_mitbih(i).seg_labels = char(seg_labels);
    % featuresSeg = [featuresSeg; peakSegFeats(N_inds,:),repmat(0,length(N_inds),1)];
    
    
    % group N:0
    % N = N, L, R, e, j
    N_inds = find(annot=='N');
    N_inds = [N_inds;find(annot=='L')];
    N_inds = [N_inds;find(annot=='R')];
    N_inds = [N_inds;find(annot=='e')];
    N_inds = [N_inds;find(annot=='j')];
    N_class = N_class + length(N_inds);
    
    % group S:1
    % S = A, a, J, S
    S_inds = find(annot=='S');
    S_inds = [S_inds;find(annot=='A')];
    S_inds = [S_inds;find(annot=='a')];
    S_inds = [S_inds;find(annot=='J')];
    S_class = S_class + length(S_inds);
    
    % group V:2
    % V = V, E
    V_inds = find(annot=='V');
    V_inds = [V_inds;find(annot=='E')];
    V_class = V_class + length(V_inds);
    
    % featuresSeg = [featuresSeg; peakSegFeats(V_inds,:),repmat(2,length(V_inds),1)];
    
    % group F:3
    % F = F
    F_inds = find(annot=='F');
    F_class = F_class + length(F_inds);
    
    % group Q:4
    % Q = /, f, Q
    Q_inds = find(annot=='/');
    Q_inds = [Q_inds;find(annot=='f')];
    Q_inds = [Q_inds;find(annot=='Q')];
    Q_class = Q_class + length(Q_inds);
    
end

% % calucualte the mean length of all beats in the dataset: it is 280
% sizes = [];
% for ind=1:length(s2s_mitbih)
%     sizes= [sizes;cellfun(@length,s2s_mitbih(ind).seg_values)];
% end
% beat_len = floor(mean(sizes))

save s2s_mitbih_aami.mat s2s_mitbih
toc
F_class
N_class
Q_class
S_class
V_class
F_class+N_class+Q_class+S_class+V_class
disp('Successfully generated :)')
