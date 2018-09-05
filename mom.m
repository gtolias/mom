fprintf('Checking for required package\n')
% if required package (Diffusion-CVPR17) is not there, then download and add to path
if ~exist('knngraph')
	system('wget https://github.com/ahmetius/diffusion-retrieval/archive/master.zip');
	system('unzip master.zip');
	addpath('diffusion-retrieval-master');
end

fprintf('Checking for input data\n')
if ~exist('vgg_rmac_1M_mom.mat')
	% download and load the input descriptors (extracted with vgg-imagenet-rmac)
	system('wget http://cmp.felk.cvut.cz/~toliageo/ext/mom/vgg_rmac_1M_mom.mat');
	load('vgg_rmac_1M_mom.mat');
end
datetime

% Na = Inf;   % all images used as anchors: setup for fine-grained recognition in CVPR18
Na = 1500;  % choose based on random walk on graph: setup for retrieval in CVPR18

fprintf('kNN graph construction\n');	
[knn, s] = knn_wrap(V, V, 30); % 30 nearest neighbors used in CVPR18. x3 faster if yael_nn available
G = knngraph(knn, s .^ 3); 		 % similarity^3 as in CVPR18 and CVPR17
datetime

% seed selection
fprintf('seed select \n');	
ids = 1:size(V,2);

if isinf(Na)
	cc = largecc(G, 1, 'rank'); 	% keep only the biggest connected component
	G = G(cc, cc);
	V = V(:, cc);
 	ids = ids(cc);

	p = powiter(spdiags(1 ./ full(sum(G,2)), 0, size(G,1), size(G,1)) * G);  % power-iteration for random-walk

	[lmx, lmxp] = graphlmax(G, p); % local max on graph

	% choose the strongest nodes from the local max ones
	[~, sort_ids] = sort(lmxp, 'descend');
	anc_idx = lmx(sort_ids(1:min(Na, numel(lmx))));
else
	anc_idx = 1:size(G, 2);
end
datetime

fprintf('pool select \n');	
L = speye(size(G)) - 0.99 * transition_matrix(G); % Laplacian, alpha = 0.99 used in CVPR18

k = 50;  % parameter k in CVPR18, equation (6)
maxpoolsize = 50;  % keep not more than
[pos, prest] = posmine(V, anc_idx, L, k, maxpoolsize); 
datetime

k = 10000; % parameter k in CVPR18, equation (6)
maxpoolsize = 50;  % keep not more than
[neg, nrest] = negmine(V, anc_idx, L, k, maxpoolsize);
datetime

% map ids back to the original ones
anc_idx  = ids(anc_idx);
pos  = cellfun(@(x) ids(x), pos, 'un', 0);
neg  = cellfun(@(x) ids(x), neg, 'un', 0);
