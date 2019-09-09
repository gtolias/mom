fprintf('Checking for required package\n')
% if required package (Diffusion-CVPR17) is not there, then download and add to path
if ~exist('knngraph')
	if ~exist('diffusion-retrieval-master')
		system('wget https://github.com/ahmetius/diffusion-retrieval/archive/master.zip');
		system('unzip master.zip');
	end
	addpath('diffusion-retrieval-master');
end

data_folder = '../data/';

fprintf('Checking for input data\n')
if ~exist(sprintf('%s/vgg_rmac_1M_mom.mat', data_folder))
	% download and load the input descriptors (extracted with vgg-imagenet-rmac)
	system(sprintf('wget http://cmp.felk.cvut.cz/~toliageo/ext/mom/vgg_rmac_1M_mom.mat -P %s', data_folder));
end
load(sprintf('%s/vgg_rmac_1M_mom.mat', data_folder));

% Na = Inf;   % all images used as anchors: setup for fine-grained recognition in CVPR18
Na = 1500;  % choose based on random walk on graph: setup for retrieval in CVPR18

fprintf('kNN graph construction (it will take few hours)\n');	
[knn, s] = knn_wrap(V, V, 30); % 30 nearest neighbors used in CVPR18. x3 faster if yael_nn is available
G = knngraph(knn, s .^ 3); 		 % similarity^3 as in CVPR18 and CVPR17

% seed selection
fprintf('seed select \n');	
ids = 1:size(V,2);

if ~isinf(Na)
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

fprintf('pool select \n');	
fprintf('hint: if parfor available - uncomment in posmine.m and negmine.m for speed-up\n');	
L = speye(size(G)) - 0.99 * transition_matrix(G); % Laplacian, alpha = 0.99 used in CVPR18

k = 50;  % parameter k in CVPR18, equation (6)
maxpoolsize = 50;  % keep not more than
[pos, prest] = posmine(V, anc_idx, L, k, maxpoolsize); 

k = 10000; % parameter k in CVPR18, equation (6) setup for retrieval 
% k = 100; % parameter k in CVPR18, equation (6) setup for fine-grained recognition
maxpoolsize = 50;  % keep not more than
[neg, nrest] = negmine(V, anc_idx, L, k, maxpoolsize);

% map ids back to the original ones
anc_idx  = ids(anc_idx);
pos  = cellfun(@(x) ids(x), pos, 'un', 0);
neg  = cellfun(@(x) ids(x), neg, 'un', 0);


% show examples from the constructed training set
fprintf('Save some examples on disk\n');

% load image thumbnails (used only to show some examples)
if ~exist(sprintf('%s/img_1M_thumbgray.mat', data_folder))
	% download and load the input descriptors (extracted with vgg-imagenet-rmac)
	system(sprintf('wget http://cmp.felk.cvut.cz/~toliageo/ext/mom/img_1M_thumbgray.mat -P %s', data_folder));
	load(sprintf('%s/img_1M_thumbgray.mat', data_folder));
end

% show some of the chosen anchors (meaningful only in the case of random-walk)
h = figure('Visible', 'off');
for i = 1:25, subplot(5,5, i), imshow(img{anc_idx(i)}); end
saveas(h, sprintf('%s/top25_anchors.png', data_folder));
h = figure('Visible', 'off');
for i = 1:25, subplot(5,5, i), imshow(img{anc_idx(500+i)}); end
saveas(h, sprintf('%s/top501-525_anchors.png', data_folder));
h = figure('Visible', 'off');
for i = 1:25, subplot(5,5, i), imshow(img{anc_idx(1000+i)}); end
saveas(h, sprintf('%s/top1001-1025_anchors.png', data_folder));

% show some positives
h = figure('Visible', 'off');
subplot(3,3,1)
imshow(img{anc_idx(1)});
for i = 1:min(8,numel(pos{1})), subplot(3,3,i+1); imshow(img{pos{1}(i)}); end
saveas(h, sprintf('%s/pos_anchor1.png', data_folder));
h = figure('Visible', 'off');
subplot(3,3,1)
imshow(img{anc_idx(30)});
for i = 1:min(8,numel(pos{30})), subplot(3,3,i+1); imshow(img{pos{30}(i)}); end
saveas(h, sprintf('%s/pos_anchor30.png', data_folder));
h = figure('Visible', 'off');
subplot(3,3,1)
imshow(img{anc_idx(100)});
for i = 1:min(8,numel(pos{100})), subplot(3,3,i+1); imshow(img{pos{100}(i)}); end
saveas(h, sprintf('%s/pos_anchor100.png', data_folder));

% show some negatives
h = figure('Visible', 'off');
subplot(3,3,1)
imshow(img{anc_idx(1)});
for i = 1:min(8,numel(neg{1})), subplot(3,3,i+1); imshow(img{neg{1}(i)}); end
saveas(h, sprintf('%s/neg_anchor1.png', data_folder));
h = figure('Visible', 'off');
subplot(3,3,1)
imshow(img{anc_idx(30)});
for i = 1:min(8,numel(neg{30})), subplot(3,3,i+1); imshow(img{neg{30}(i)}); end
saveas(h, sprintf('%s/neg_anchor30.png', data_folder));
h = figure('Visible', 'off');
subplot(3,3,1)
imshow(img{anc_idx(100)});
for i = 1:min(8,numel(neg{100})), subplot(3,3,i+1); imshow(img{neg{100}(i)}); end
saveas(h, sprintf('%s/neg_anchor100.png', data_folder));
