clear all; close all; clc;
for sub=1:54
if sub<10
    load(['WHERE\IS\DATA\sess01_subj0',num2str(sub),'_EEG_MI.mat']);
else
    load(['WHERE\IS\DATA\sess01_subj',num2str(sub),'_EEG_MI.mat']);
end

%% predefined filter-bank
    %% pre-processing
channel_index = [8,33,9,10,34,11,35,13,36,14,37,15,38,18,39,19,40,20,41,21];
numband=33;
band=zeros(numband,2);
for aa=1:numband
    band(aa,1)=aa+3;
    band(aa,2)=aa+7;
end

time_interval = [500 2500]; 
CSPFilter=3;

EEG_MI_train_ch = prep_selectChannels(EEG_MI_train, {'Index', channel_index});
CNT_off =prep_filterbank2(EEG_MI_train_ch , {'frequency', band});
SMT_off = prep_segmentation_filterbank(CNT_off, {'interval', time_interval});

[SMT, CSP_W, CSP_D]=func_csp_filterbank(SMT_off,{'nPatterns', CSPFilter});
FT=func_featureExtraction_filterbank(SMT, {'feature','logvar'});

    %% mutual information
for iii=1:numband
    mutual_info_FT{1,iii}=func_mutual_information2(FT{1,iii},FT{1,iii}.x,FT{1,iii}.y_dec);
    max_mut(iii)=max(mutual_info_FT{1,iii});
end

band_m=find(max_mut==max(max_mut));
[out_m,idx_m] = sort(max_mut,'descend');
sorted_band_pre=zeros(numband,2);

for bb=1:numband
    sorted_band_pre(bb,:)=band(idx_m(1,bb),:);
end
clear idx_m max_mut

%% subject-dependent filter optimization
    %% pre-processing
numband=30;
gauss=normrnd(0,1,[numband+10,2]); %k=20
band=zeros(numband+10,2);

k=1;
for aa=1:numband+10
    band_n1=sorted_band_pre(1,1)+gauss(aa,1);
    band_n2=sorted_band_pre(1,2)+gauss(aa,2);
    if (band_n1>0.5) && (band_n2>0.5) && (band_n2-band_n1>1)
        band(k,1)=band_n1;
        band(k,2)=band_n2;
        k=k+1;
    end
end
band=band(1:numband,:);
  
EEG_MI_train_ch = prep_selectChannels(EEG_MI_train, {'Index', channel_index});
CNT_off =prep_filterbank2(EEG_MI_train_ch , {'frequency', band});
SMT_off = prep_segmentation_filterbank(CNT_off, {'interval', time_interval});

[SMT, CSP_W, CSP_D]=func_csp_filterbank(SMT_off,{'nPatterns', CSPFilter});
FT=func_featureExtraction_filterbank(SMT, {'feature','logvar'});

    %% mutual information
for iii=1:numband
    mutual_info_FT{1,iii}=func_mutual_information2(FT{1,iii},FT{1,iii}.x,FT{1,iii}.y_dec);
    max_mut(iii)=max(mutual_info_FT{1,iii});
end

band_m=find(max_mut==max(max_mut));
[out_m,idx_m] = sort(max_mut,'descend');
sorted_band_opt=zeros(numband,2);

for bb=1:numband
    sorted_band_opt(bb,:)=band(idx_m(1,bb),:);
end

%% filter-set
sorted_band=zeros(40,2);
for finband=1:40
    if mod(finband,4)==0
        sorted_band(finband,1)=sorted_band_opt(finband/4,1);
        sorted_band(finband,2)=sorted_band_opt(finband/4,2);
    else
        sorted_band(finband,1)=sorted_band_pre(finband-floor(finband/4),1);
        sorted_band(finband,2)=sorted_band_pre(finband-floor(finband/4),2);
    end
end

%% save
if sub<10
    save(['filterset_sess01_subj0',num2str(sub),'.mat'],'sorted_band')
else
    save(['filterset_sess01_subj',num2str(sub),'.mat'],'sorted_band')
end
clearvars -except sub
end
