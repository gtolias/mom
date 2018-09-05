% ----------------------------------------------------------------------------
% mine pool of positives per anchor
% Input
% 	V 			   : descriptors
% 	anc_idx    : anchor indices
% 	L          : graph Laplacian
% 	k          : k nearest neighbors (Euclidean and manifold ones)
% 	maxpoolsize: maximum size of the pool
% returns
% 	pos        : indices for positive vectors
% 	rest       : additional information for each positive vector
function [pos, rest] = posmine(V, anc_idx, L, k, maxpoolsize)

	parfor i = 1:numel(anc_idx)
		[pos{i}, rest{i}] = posmine_single(V, single(anc_idx(i)), L, k, maxpoolsize);
	end

% ----------------------------------------------------------------------------
function [pos, rest] = posmine_single(V, q, L, k, maxpoolsize)

	if exist('yael_nn')
		[re, se] = yael_nn(V, -V(:, q), size(V, 2), 16); se = -se;  % Euclidean ranking & similarity
	else
		[se, re] = sort(V(:,q)'* V, 'descend');
	end
	sm = dfs(L, ei(size(V,2), q), 1e-10, 20);    							  % manifold similarity
	if exist('yael_kmax')
		[sm, rm] = yael_kmax(single(sm), k);  										  % manifold neighbors
	else
		[sm, rm] = sort(single(sm), 'descend');
		sm(k+1:end) = []; rm(k+1:end) = [];
	end
	for i=1:numel(se), ire(re(i)) = i; end   										% rank id
	f = find(ire(rm) >  k);  						 										    % Euclidean non-neighbors & manifold neighbors
	[~, idx] = sort(sm(f), 'descend');       										% sort the selected ones by manifold similarity
	
	% positive pool
	idx = idx(1:min(maxpoolsize, numel(f)));
	pos = rm(f(idx))';

	% additional info each positive
	rest.irm = f(idx); 					% manifold rank id 					
	rest.sm = sm(f(idx));				% manifold similarity
	rest.ire = ire(rm(f(idx)));	% Euclidean rank id
	rest.se = se(rm(f(idx)));		% Euclidean similarity

% -----------------------------------------------------------------------------
function e = ei(n, i)
	e = zeros(n, 1); 
	e(i) = 1;