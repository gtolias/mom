function r = powiter(S, iter)

	if ~exist('iter'), iter = 1000; end;
	N = size(S, 1);

	r = ones(1,N)/N;
	for i = 1:iter
		r = r*S; 
		r = r ./ norm(r); 
	end
