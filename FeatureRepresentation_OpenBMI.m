clear all; close all; clc;

num_sp=25;
chnum=20;
train_data=zeros(100,chnum,chnum,num_sp);
test_data=zeros(100,chnum,chnum,num_sp);
    
load(['WHERE\IS\DATA\electrode_position.mat']);
data.nfo.pos_3d=pos_3d.data';
MNT.clab=pos_3d.textdata';
MNT.pos_3d=pos_3d.data';

for sub=1:54
    %% load data
    if sub<10
        load(['WHERE\IS\DATA\sess01_subj0',num2str(sub),'_EEG_MI.mat']);
    else
        load(['WHERE\IS\DATA\sess01_subj',num2str(sub),'_EEG_MI.mat']);
    end
    %% load filter-set    
    if sub<10
        load(['WHERE\IS\DATA\filterset_sess01_subj0',num2str(sub),'.mat'])
    else
        load(['WHERE\IS\DATA\filterset_sess01_subj',num2str(sub),'.mat'])
    end

    %% stack nscm
    for sp=1:num_sp
        
        freq1=sorted_band(sp,1);
        freq2=sorted_band(sp,2);
        startp=500;
        endp=2500;
        
       %% filtering and segmentation
        [b,a]= butter(2, [freq1 freq2]/(1000/2));
        CNT_T = prep_selectChannels(EEG_MI_train, {'Index', [8,33,9,10,34,11,35,13,36,14,37,15,38,18,39,19,40,20,41,21]});
        CNT_T.x = filter(b,a,CNT_T.x);
        train = prep_segmentation(CNT_T, {'interval', [startp endp]});

        CNT_E= prep_selectChannels(EEG_MI_test, {'Index', [8,33,9,10,34,11,35,13,36,14,37,15,38,18,39,19,40,20,41,21]});
        CNT_E.x = filter(b,a,CNT_E.x);
        test = prep_segmentation(CNT_E, {'interval', [startp endp]});

       %% local average reference (training data)
        MNT_T=MNT;
        MNT_T.fs=train.fs;
        MNT_T.pos=train.t;
        train.clab=MNT_T.clab;
        train.x=permute(train.x,[1,3,2]);
        train= proc_localAverageReference(train, MNT_T, 'Radius',0.4);

        %% nscm (training data)
        train_nscm=zeros(size(train.x,3),chnum,chnum);
        for trial=1:size(train.x,3)    
           X = squeeze(train.x(:,:,trial));
           X2 = (X./sqrt(repmat(diag(X*X'),1,chnum)));
           train_nscm(trial,:,:) = (chnum/size(train.x,1))*(X2')*X2;
        end
        
       %% local average reference (test data)
        MNT_E=MNT;
        MNT_E.fs=test.fs;
        MNT_E.pos=test.t;
        test.clab=MNT_E.clab;
        test.x=permute(test.x,[1,3,2]);
        test= proc_localAverageReference(test, MNT_E, 'Radius',0.4);

       %% nscm (test data)
        test_nscm=zeros(size(test.x,3),chnum,chnum);
        for trial=1:size(test.x,3)    
           X = squeeze(test.x(:,:,trial));
           X2 = (X./sqrt(repmat(diag(X*X'),1,chnum)));
           test_nscm(trial,:,:) = (chnum/size(test.x,1))*(X2')*X2;
        end
        
       %%
        train_labels= train.y_logic';
        test_labels= test.y_logic';
        train_data(:,:,:,sp)=train_nscm;
        test_data(:,:,:,sp)=test_nscm;

       %% save
        if sp==num_sp
            if sub<10
                save(['feature_representation_sess01_subj0',num2str(sub),'.mat'],'test_data', 'test_labels', 'train_data','train_labels')
            else
                save(['feature_representation_sess01_subj',num2str(sub),'.mat'],'test_data', 'test_labels', 'train_data','train_labels')
            end
        end

    end
end

