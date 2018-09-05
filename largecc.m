% find nodes belonging to the largest connected components
function f = largecc(A, t, md)

	if ~exist('md'), md = 'thres'; end
	if ~exist('t'), t = inf; end

	[nc, c] = graphconncomp(A);
	cc = accumarray(c', 1);

	if strcmp(md, 'thres')
		if isinf(t), t = max(cc); end
		f = find(cc>=t);
	elseif strcmp(md, 'rank')
		[~, f] = sort(cc, 'descend');
		f = f(1:t);
	else
		error('unknown mode\n')		
	end

	f = find(ismember(c, f));
