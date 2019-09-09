% ----------------------------------------------------------------------------
% mine pool of negatives per anchor
% Input
% 	V          : descriptors
% 	anc_idx    : anchor indices
% 	L          : graph Laplacian
% 	k          : k nearest neighbors (Euclidean and manifold ones)
% 	maxpoolsize: maximum size of the pool
% returns
% 	pos	       : indices for negative vectors
% 	rest       : additional information for each negative vector
function [neg, rest] = negmine(V, anc_idx, L, k, maxpoolsize)

	% parfor i = 1:numel(anc_idx)
	for i = 1:numel(anc_idx)
		[neg{i}, rest{i}] = negmine_single(V, single(anc_idx(i)), L, k, maxpoolsize);
	end

% ----------------------------------------------------------------------------
function [neg, rest] = negmine_single(V, q, L, k, maxpoolsize)
	
	k = min(ceil(.5 * size(V, 2)), k); % in case too large wrt graph size
	
	if exist('yael_nn')
		[re, se] = yael_nn(V, -V(:, q), k, 16); se = -se;  % Euclidean neighbors
	else
		[se, re] = sort(V(:,q)'* V, 'descend');
		se(k+1:end) = []; re(k+1:end) = [];
	end
	sm = dfs(L, ei(size(V,2), q), 1e-10, 20);  				 % manifold similarity
	[~, rm] = sort(sm, 'descend');  									 % manifold ranking
	for i=1:numel(sm), irm(rm(i)) = i; end   					 % rank id 
	f = find(irm(re) >  k);  													 % Euclidean neighbors & manifold non-neighbors
	[~, idx] = sort(se(f), 'descend'); 								 % sort selected ones by euclidean similarity

	% negative pool
	idx = idx(1:min(maxpoolsize, numel(f)));
	neg = re(f(idx))';

	% additional info each negative
	rest.irm = irm(re(f(idx)));   % manifold rank id
	rest.sm = sm(re(f(idx)));			% manifold similarity
	rest.ire = f(idx);						% Euclidean rank id
	rest.se = se(f(idx));				  % Euclidean similarity

% -----------------------------------------------------------------------------
function e = ei(n, i)
	e = zeros(n, 1); 
	e(i) = 1;