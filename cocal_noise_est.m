
% clc;
% clear;

addpath(genpath('nmfv1_4'))


a = dataset('XLSFile','PrAD.csv');
a = dataset2table(a);
start_col = 3; end_col = 85;%
sbjFea = table2array(a(1:end-1, start_col: end_col)); %, a(:, 266:523)double(brain);

deno = table2array(a(1:end-1, 2));
denom = repmat(deno, 1, size(sbjFea, 2));
denom(denom==0) = eps;
nSbjFea = sbjFea./ denom;

rind_vec1 = []; rind_vec2 = [];
[numSbj, numFea] = size(nSbjFea);
% [h, w] = size(nSbjFea);
ornSbjFea = nSbjFea;
NS = round( 0.8 * numSbj );%

load 'SMtr_SC_2FC_9_nonrescale.mat';
SSMtr = sort(SMtr, 'descend');
%%
expec = median(SSMtr, 2);
%%

Var_fea_observe = var(SSMtr,0, 2);

Kcv = 2;  %4 :5 number of subject sub-cluster
Krv = 9; %7:21 number of feature sub-cluster
numiter = 30;


for i = 1:length(Kcv)
    for j = 1:length(Krv)
        
        for loglambda = -11:0.5:-3
            
            sn0 = 2^loglambda;
            
            for kk = 1:numiter
                
                randid = randperm(numSbj);
                addfea = nSbjFea(randid(1:NS), :);
                addfea = addfea + 0.01*randn(size(addfea));
                featmp = [nSbjFea; addfea];
                
                randid = randperm(numSbj+NS);
                nSbjFea2 = featmp(randid(1:numSbj), :);
                
                close all;
                Kc = Kcv(i); Kr = Krv(j);
                
                option.orthogonal = [1,1];
                option.iter = 10000;
                option.dis = 0;
                option.residual = 1e-5;
                option.tof = 1e-5;
                
                disp('Performing Collaborative Clustering ...');
                tic;
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
                nSbjFea = SbjFea_new';
                [A,S,Y,numIter,tElapsed,finalResidual] = orthnmfrule_mod(nSbjFea', Kc, Kr, option);
                

                
                kLab = litekmeans(Y',Kc,'Replicates',30);
                [~,aLab] = max(A,[],2);
                clustermatr1(kk, :) = aLab;
                clustermatr2(kk, randid(1:numSbj)) = kLab;

            end
            
            
            rind1 = [];
            rind2 = [];
            clustermatr2(clustermatr2==0)=Krv+1;
            for id1 = 1:numiter-1
                for id2 = id1+1:numiter
                    rind1 = [ rind1, RandIndex( clustermatr1(id1, :), clustermatr1(id2, :) ) ];
                    rind2 = [ rind2, RandIndex( clustermatr2(id1, :), clustermatr2(id2, :) ) ];
                end
            end
            rind_vec1 = [rind_vec1 mean(rind1)];
            rind_vec2 = [rind_vec2 mean(rind2)];
        end
        
        save('rind_vec1.mat','rind_vec1');
        save('rind_vec2.mat','rind_vec2');
        
        
        
    end
end

disp('Finished.');
