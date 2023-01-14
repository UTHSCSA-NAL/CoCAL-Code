% clc;
clear;
close all;

addpath(genpath('nmfv1_4'))

a = dataset('XLSFile','PrAD.csv'); % Input datafile. Each row contains the features of a subject.
a = dataset2table(a);
start_col = 3; end_col = 85;% Specify the features used. 
sbjFea = table2array(a(1:end-1, start_col: end_col)); %, a(:, 266:523)double(brain);

deno = table2array(a(1:end-1, 2));
denom = repmat(deno, 1, size(sbjFea, 2));
denom(denom==0) = eps;
nSbjFea = sbjFea./ denom;

[numSbj, numFea] = size(nSbjFea);

sn0 = 2^-9.5; % Decided by noise level

load 'SMtr_SC_2FC_3_nonrescale.mat'; %To form coefficient distribution
SSMtr = sort(SMtr, 'descend');
expec = mean(SSMtr, 2);
Var_fea_observe = var(SSMtr,0, 2);

Kcv = 2;  % number of subject sub-cluster
Krv = 3; % number of feature sub-cluster

for i = 1:length(Kcv)
    for j = 1:length(Krv)
        
        close all;
        Kc = Kcv(i); Kr = Krv(j);
        
        option.orthogonal = [1,1];
        option.iter = 300000;
        option.dis = 0;
        option.residual = 1e-5;
        option.tof = 1e-5;
        
        FileName = ['SMtr_SC_' num2str(Kcv(i)) 'FC_' num2str(Krv(j)) '_nonrescale.mat'];
        load(FileName);
        SSMtr = sort(SMtr, 'descend');
        expec = mean(SSMtr, 2);
        Var_fea_observe = var(SSMtr,0, 2);
        
        for inneriter = 1:1
            
            disp('Performing Collaborative Clustering ...');
            tic;
            rand('twister',37);
            [A,S,Y,numIter,tElapsed,finalResidual] = orthnmfrule_mod(nSbjFea', Kc, Kr, option);
            toc;
            
            %%  Normalize Dictionaries
            sY = sqrt(sum(Y.^2,2));
            sA = sqrt(sum(A.^2));
            Y = Y ./ repmat(sY,1,size(Y,2));
            A = A ./ repmat(sA,size(A,1),1);
            S = repmat(sA',1, size(S,2)) .* S .* repmat(sY',size(S,1),1);
            
            %%  Shrink S
            [SS, inds] = sort(S(:), 'descend');
            

            scale = numSbj * numFea/(Kc*Kr);
            sn = sn0*scale;
            Var_fea = max(Var_fea_observe-sn, eps);
            S_bar = SS - expec;
            shrkS = sn0 * sqrt(2 * numSbj * numFea ./ Var_fea);
            SSr = expec + max(abs(S_bar)-shrkS, 0).* sign(S_bar);%(Var_fea.*SS + sn.*expec) ./ Var_fea_observe;
            St = S;
            St(inds) = SSr;
            Sr = reshape(St, size(S));
            
            
            SbjFea_new = A*Sr*Y;

            nSbjFea2 = SbjFea_new';

            [A,S,Y,numIter,tElapsed,finalResidual] = orthnmfrule_mod(nSbjFea2', Kc, Kr, option);
            
        end
        
        kLab = litekmeans(Y',Kc,'Replicates',100);
        
        [~,aLab] = max(A,[],2);
        
        [skLab,skInd] = sort(kLab);
        [saLab,saInd] = sort(aLab);
        
        dSbjFea = Y'*S';
        minDsbj = repmat(min(dSbjFea),size(dSbjFea,1),1);
        maxDsbj = repmat(max(dSbjFea),size(dSbjFea,1),1);
        ndSbjFea = (dSbjFea-minDsbj) ./ max(eps,maxDsbj-minDsbj);
        
        %% plot clustering results
        figure; colormap('parula');hold on; box on;
        imagesc(nSbjFea2(skInd,saInd)); axis image; colorbar; caxis([min(nSbjFea2(:)),max(nSbjFea2(:))]); xlabel('Feature ID'); ylabel('Sbj ID');
        cpt = find(diff(skLab)~=0) + 1;
        for cpi=1:length(cpt)
            plot([0,size(nSbjFea2,2)+1],[cpt(cpi),cpt(cpi)],'r--','LineWidth',2);
        end
        
        cpa = find(diff(saLab)~=0);
        for cpi=1:length(cpa)
            plot([cpa(cpi),cpa(cpi)],[1,size(nSbjFea2,1)],'r--','LineWidth',2);
        end
        saveas(gcf, ['cluster_result_nonrescaled_Kc',num2str(Kc),'_Kr',num2str(Kr) '.png']);
        
		%%Print results
        a(end, start_col:end_col) = array2table(aLab');
        a(1:end-1, end) = array2table(kLab);
        af = [a(:, 1:2), a(:, start_col:end)];
        writetable(af,['acc_result_','Kc',num2str(Kc),'_Kr',num2str(Kr),'.csv']);
        
		%%Save results for further analysis
        outName = ['cc_result_nonrescaled','Kc',num2str(Kc),'_Kr',num2str(Kr),'.mat'];
        save(outName,'nSbjFea','ndSbjFea','aLab','kLab');
        
    end
end

disp('Finished.');
