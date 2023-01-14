% clc;
clear;
close all;

addpath(genpath('nmfv1_4'))

a = dataset('XLSFile','PrAD.csv');
a = dataset2table(a);
start_col = 3; end_col = 85;%
sbjFea = table2array(a(1:end-1, start_col: end_col)); %, a(:, 266:523)double(brain);

deno = table2array(a(1:end-1, 2));
denom = repmat(deno, 1, size(sbjFea, 2));
denom(denom==0) = eps;
nSbjFea = sbjFea./ denom;

[numSbj, numFea] = size(nSbjFea);
NS = round( 1.3 * numSbj ) - numSbj;

%% clustering
Kcv = 2:4;   % number of subject sub-cluster
Krv = 3:11; % number of feature sub-cluster

for i = 1:length(Kcv)
    for j = 1:length(Krv)
        
        SMtr = [];
        Kc = Kcv(i); Kr = Krv(j);
        
        for k = 1:150
            
            randid = randperm(numSbj);
            addfea = nSbjFea(randid(1:NS), :);
            addfea = addfea + 0.001*randn(size(addfea));
            featmp = [nSbjFea; addfea];
            
            randid = randperm(numSbj+NS);
            nSbjFea2 = featmp(randid(1:numSbj), :);
            
            close all;
                        
            lambda = 2^-9.5;
            option.orthogonal = [1,1];
            option.iter = 300000;%40000
            option.dis = 0;
            option.residual = 1e-5;
            option.tof = 1e-5;
            
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
                
                %%  
                [SS, inds] = sort(S(:), 'descend');
                SMtr = [SMtr SS];
                
            end

        end
        
        save(['SMtr_SC_' num2str(Kc) 'FC_' num2str(Kr) '_nonrescale.mat'],'SMtr');

    end
end

disp('Finished.');
