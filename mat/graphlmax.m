% find nodes (with importance p) on graph G that are local max of an l-order  neighborhood
% returns ids of nodes and their importance
function [lmx, lmxp] = graphlmax(G, p, l)
	if ~exist('l'), l = 1; end

	N = size(G, 1);     % graph size
	islmx = zeros(1, N);  % is local-max

	% find edges here, to speed-up later
	edges = getfield(getfield(graph(G), 'Edges'), 'EndNodes');
 	edges1 = accumarray(edges(:,1), edges(:,2), [size(G,2), 1], @(x){x});
 	edges2 = accumarray(edges(:,2), edges(:,1), [size(G,2), 1], @(x){x});

	for i = 1:N
		lnk = [edges1{i}; edges2{i}]; % direct neighbors

		% 'recursively' expand neighborhood
	 	for it = 1:(l-1)
			lnk = unique([cell2mat(arrayfun(@(x) find(G(x, :)), lnk, 'un', 0)), lnk]);
		end
		if l > 1, lnk = setdiff(lnk, i); end

		% check for local-max
		if p(i) > max(p(lnk)), islmx(i) = 1; end
	end

	lmx = find(islmx);
	lmxp = p(lmx);