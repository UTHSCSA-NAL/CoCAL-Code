% clc;
clear;
close all;

addpath(genpath('nmfv1_4'))

fn_txt        =   'randidx_cocal.txt';
fd_txt        =   fopen( fullfile(fn_txt), 'at');

a = dataset('XLSFile','PrAD.csv');
a = dataset2table(a);
start_col = 3; end_col = 85;%
sbjFea = table2array(a(1:end-1, start_col: end_col)); %, a(:, 266:523)double(brain);

deno = table2array(a(1:end-1, 2));
denom = repmat(deno, 1, size(sbjFea, 2));
denom(denom==0) = eps;
sbjFea = sbjFea./ denom;

[numSbj, numFea] = size(sbjFea);
NS = round( 1 * numSbj ) ; %0.8
sn0 = 2^-9.5;

%% clustering
Kcv = 2:5;   % number of subject sub-cluster
Krv = 3:11; % number of feature sub-cluster
Nrep = 100;


for i = 1:length(Kcv)
    
    Kc = Kcv(i);
    fprintf(fd_txt, '\n%d Clusters\n', Kc);
    
    for j = 1:length(Krv)
        
        close all;
        Kr = Krv(j);
        coefile = ['SMtr_SC_' num2str(Kc) 'FC_' num2str(Kr) '_nonrescale.mat'];
        SMtrL = load(coefile);
        SSMtr = sort(SMtrL.SMtr, 'descend');
        expec = mean(SSMtr, 2);
        Var_fea_observe = var(SSMtr,0, 2);
        clustermatr1 = zeros(Nrep, NS);
        
        for kk = 1:Nrep
            
            randid = randperm(numSbj);
            nSbjFea2 = sbjFea(randid(1:NS), :);
            nSbjFea2(:,std(nSbjFea2)==0) = [];
            
            option.orthogonal = [1,1];
            option.iter = 10000;%40000
            option.dis = 0;
            option.residual = 1e-7;
            option.tof = 1e-7;
            
            
            for inneriter = 1:1
                
                disp('Performing Collaborative Clustering ...');
                tic;
                %             rand('twister',7);
                [A,S,Y,numIter,tElapsed,finalResidual] = orthnmfrule_mod(nSbjFea2', Kc, Kr, option);
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
                %                 nSbjFea2 = SbjFea_new';
                %%
                %             rand('twister',17);
                [A,S,Y,numIter,tElapsed,finalResidual] = orthnmfrule_mod(SbjFea_new, Kc, Kr, option);
                %
            end
            
            %         rand('twister',37);
            kLab = litekmeans(Y',Kc,'Replicates',30);
            %kLab = litekmeans(nSbjFea,Kc,'Replicates',20);  % clustering based on original features
            
            [~,aLab] = max(A,[],2);
            
            clustermatr1(kk, randid(1:NS)) = kLab;
            
        end
        
        rind1 = [];
        clustermatr1(clustermatr1==0)=Kr+1;
        for id1 = 1:Nrep-1
            for id2 = id1+1:Nrep
                rind1 = [ rind1, RandIndex( clustermatr1(id1, :), clustermatr1(id2, :) ) ];
            end
        end
        
        rind_vec1 = mean(rind1);
        rind_vec01 = median(rind1);
        
        fprintf(fd_txt, '(%2.3f,%2.3f)\t', rind_vec1, rind_vec01);
        
    end
    
end

fclose(fd_txt);
disp('Finished.');
