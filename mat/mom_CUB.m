fprintf('Checking for required package\n')
% if required package (Diffusion-CVPR17) is not there, then download and add to path
if ~exist('knngraph')
	system('wget https://github.com/ahmetius/diffusion-retrieval/archive/master.zip');
	system('unzip master.zip');
	addpath('diffusion-retrieval-master');
end

data_folder = '../data/';

fprintf('Checking for input data\n')
if ~exist(sprintf('%s/googlenet_rmac_ms_pcaw512_CUB.mat', data_folder))
	% if descriptors are not there, extract them
	run extract_CUB_descriptors_pretrained.m
end
load(sprintf('%s/googlenet_rmac_ms_pcaw512_CUB.mat', data_folder));

fprintf('kNN graph construction\n');	
[knn, s] = knn_wrap(V, V, 30); % 30 nearest neighbors used in CVPR18. x3 faster if yael_nn is available
G = knngraph(knn, s .^ 3); 		 % similarity^3 as in CVPR18 and CVPR17

% seed selection
fprintf('seed select \n');	
ids = 1:size(V,2);
anc_idx = 1:size(G, 2); % all images used as anchors

fprintf('pool select \n');	
L = speye(size(G)) - 0.99 * transition_matrix(G); % Laplacian, alpha = 0.99 used in CVPR18

k = 50;  % parameter k in CVPR18, equation (6)
maxpoolsize = 50;  % keep not more than
[pos, prest] = posmine(V, anc_idx, L, k, maxpoolsize); 

k = 100; % parameter k in CVPR18, equation (6) setup for fine-grained recognition
maxpoolsize = 50;  % keep not more than
[neg, nrest] = negmine(V, anc_idx, L, k, maxpoolsize);

% map ids back to the original ones
anc_idx  = ids(anc_idx);
pos  = cellfun(@(x) ids(x), pos, 'un', 0);
neg  = cellfun(@(x) ids(x), neg, 'un', 0);

% save to text files, will be loaded from python
dlmwrite(sprintf('%s/anchors.txt', data_folder), anc_idx');
fid = fopen(sprintf('%s/pos.txt', data_folder), 'w'); for i = 1:numel(anc_idx), fprintf(fid,'%d, ', pos{i}(1:end-1)); fprintf(fid,'%d\n', pos{i}(end)); end; fclose(fid);
fid = fopen(sprintf('%s/neg.txt', data_folder), 'w'); for i = 1:numel(anc_idx), fprintf(fid,'%d, ', neg{i}(1:end-1)); fprintf(fid,'%d\n', neg{i}(end)); end; fclose(fid);
fid = fopen(sprintf('%s/posw.txt', data_folder), 'w'); for i = 1:numel(anc_idx), fprintf(fid,'%.6f, ', prest{i}.sm(1:end-1)); fprintf(fid,'%.6f\n', prest{i}.sm(end)); end; fclose(fid);
fid = fopen(sprintf('%s/negw.txt', data_folder), 'w'); for i = 1:numel(anc_idx), fprintf(fid,'%.6f, ', nrest{i}.sm(1:end-1)); fprintf(fid,'%.6f\n', nrest{i}.sm(end)); end; fclose(fid);