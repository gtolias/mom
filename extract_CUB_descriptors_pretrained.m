% extract descriptors (Y in the paper) from CUB dataset using googlenet pre-trained on imagenet
% this set is the input to the mining algorithm
ifolder = 'CUB_200_2011/images/';  % assumes CUB dataset is already downloaded
system('wget -nc http://cmp.felk.cvut.cz/~toliageo/ext/mom/train_imlist.txt')
imlist = textread('train_imlist.txt', '%s\n');
imsize = 512; 

system('wget -nc http://www.vlfeat.org/matconvnet/models/imagenet-googlenet-dag.mat')
net = load('imagenet-googlenet-dag.mat');
net = dagnn.DagNN.loadobj(net);
net = remove_blocks_with_name(net, 'cls');
net = remove_blocks_with_name(net, 'softmax');
mpx = mean(net.meta.normalization.averageImage(:));
net.vars(net.getVarIndex('icp9_out')).precious = 1;
net.move('gpu')

fprintf('Extracting...\n')
for i = 1:numel(imlist);
	I = imread(fullfile(ifolder, imlist{i}));
	I = imresizemaxd(I, imsize);
	if size(I, 3) == 1,	I = repmat(I, [1 1 3]); end
	x = [];
	for s = imsize .* [1, 1./sqrt(2), 1/2]
		I2 = imresizemaxd(I, s);
		net.eval({'data', gpuArray((reshape(single(I2) - mpx, [size(I2,1), size(I2,2), 3, 1]))) });
		A = extractongrid(gather((net.vars(end).value)), 3);
		x = [x, vecpostproc(sum(vecpostproc(A.des), 2))]; % l2norm each descriptor and append
	end	
	V(:, i) = vecpostproc(sum(x, 2));  % sum pool across image scales and l2 norm
end
fprintf('Done with extraction...\n')

fprintf('PCA - learn and apply whitening...\n')
[~, eigvec, eigval, Xm] = yael_pca(V, 512); 
V = vecpostproc(apply_whiten(V, Xm, eigvec, eigval, 512));
fprintf('Done\n');


% ---------------------------------------------------------------------------
% ---------------------------------------------------------------------------
% ---------------------------------------------------------------------------

function net = remove_blocks_with_name(net, name)
% REMOVE_BLOCKS_WITH_NAME function removes layers that contain specific name.
%
%   NET = remove_blocks_with_name(NET, NAME)
%   
%   NAME is a string. All layers from NET containing this string will be removed from it.
%
% Authors: F. Radenovic, G. Tolias, O. Chum. 2017. 
	
	% find layers containing string name
	layers = {};
	for l = 1:numel(net.layers)
		if numel(strfind(net.layers(l).name, name))
			layers{1,end+1} = net.layers(l).name;
		end
	end

	% if no layers return
	if isempty(layers), return; end

	% remove found layers
	fprintf('>> Removing layers that contain word ''%s'' in their name\n', name);
	for i = 1:numel(layers)
		layer = net.layers(net.getLayerIndex(layers{i}));
		net.removeLayer(layers{i});
		net.renameVar(layer.outputs{1}, layer.inputs{1}, 'quiet', true) ;
	end

end

function im = imresizemaxd(im, sz, increase_allowed, method)
% IMRESIZEMAXD resizes image so that longer edge is maximum to the given size.
%   
%   IM = IMRESIZEMAXD(IM, SIZE)
%     Resize IM so that longer edge is maximum SIZE.
%
%   IM = IMRESIZEMAXD(IM, SIZE, INCREASE_ALLOWED, METHOD)
%     Resize IM so that longer edge is maximum SIZE.
%     INCREASE_ALLOWED defines if smaller images will be upscaled. Default is TRUE.
%     METHOD defined the MATLAB supported imresize method. Default is 'BICUBIC'.

	if ~exist('increase_allowed'), increase_allowed = 1; end
	if ~exist('method'), method = 'bicubic'; end	

	if size(im,1) <= sz && size(im,2) <= sz && ~increase_allowed
		return;
	end
	if size(im,1) > size(im,2)
		im = imresize(im, [sz NaN], method);
	elseif size(im,1) < size(im,2)
		im = imresize(im, [NaN sz], method);
	else
		im = imresize(im, [sz sz], method);
	end

end

function v = extractongrid(x, L)
	a = maxp(x);
	opts.mode = 'uniform'; opts.L = L;
	[v.reg, v.l] = rsel(x, opts);
	v.des = cropr(x, v.reg, @maxp);
end


function [r, s] = rsel(x, opts)
%
% Authored by G. Tolias, 2015. 
%

	[H,W,~] = size(x);
	r = [];

	switch opts.mode
		case 'random'
			for i = 1:opts.n
				c = ceil(rand(2,1) .* [H-1;W-1]);
				c = [c; c + ceil(rand(2,1) .* [H-c(1);W-c(2)])];
				r = [r, c];			
			end

		case 'uniform'
			ovr = 0.4; % desired overlap of neighboring regions		
			if isfield(opts, 'ovr'), ovr = opts.ovr; end
			steps = [2:10]; % possible regions for the long dimension

			w = min([W H]);
			w2 = floor(w/2 -1);

			b = (max(H, W)-w)./(steps-1);
			[~, idx] = min(abs(((w.^2 - w.*b)./w.^2)-ovr)); % steps(idx) regions for long dimension

			% region overplus per dimension
			Wd = 0;
			Hd = 0;
			if H < W, Wd = idx; elseif H > W, Hd = idx; end

			r = [1;1;H;W];
			s(1) = 0;
			for l = 1:opts.L
			  wl = floor(2*w./(l+1));
			  wl2 = floor(wl/2 - 1);

			  b = (W-wl)./(l+Wd-1);  
			  if isnan(b), b = 0; end % for the first level
			  cenW = floor(wl2 + [0:l-1+Wd]*b) -wl2; % center coordinates
			  b = (H-wl)./(l+Hd-1);
			  if isnan(b), b = 0; end % for the first level
			  cenH = floor(wl2 + [0:l-1+Hd]*b) - wl2; % center coordinates

			  for i_ = cenH
			    for j_ = cenW
			      if ~wl, continue; end
			      r = [r, [i_+1;j_+1;i_+wl;j_+wl]];
			      s = [s, l];
			    end
			  end

			end
	end

end

function xr = cropr(x, r, fun)
	if nargin < 3, fun = @max_act; end

	xr = [];
	for i = 1:size(r, 2)
		xr(:, i) = fun(x(r(1,i):r(3,i), r(2,i):r(4,i), :));
	end
end

function x = maxp(x)
  if ~max(size(x, 1), size(x, 2))
		x = zeros(size(x, 3), 1, class(x));
		return;
  end
  x = reshape(max(max(x, [], 1), [], 2), [size(x,3) 1]);
end

 function x = vecpostproc(x, a) 
	if ~exist('a'), a = 1; end
	x = replacenan (vecs_normalize  (x));
end

% replace all nan values in a matrix (with zero)
function y = replacenan (x, v)
	if ~exist ('v')
	  v = 0;
	end
	y = x;
	y(isnan(x)) = v;	
end

function X = vecs_normalize(X)
	l = sqrt(sum(X.^2));
	X = bsxfun(@rdivide,X,l);
	X = replacenan(X);
end


% PCA with automatic selection of the method: covariance or gram matrix
% Usage: [X, eigvec, eigval, Xm] = pca (X, dout, center, verbose)
%   X       input vector set (1 vector per column)
%   dout    number of principal components to be computed
%   center  need to center data?
% 
% Note: the eigenvalues are given in decreasing order of magnitude
%
% Author: Herve Jegou, 2011. 
% Last revision: 08/10/2013
function [X, eigvec, eigval, Xm] = yael_pca (X, dout, center, verbose)

	if nargin < 3,         center = true; end
	if ~exist ('verbose'), verbose = false; end

	X = double (X);
	d = size (X, 1);
	n = size (X, 2);

	if nargin < 2
	  dout = d;
	end

	if center
	  Xm = mean (X, 2);
	  X = bsxfun (@minus, X, Xm);
	else
	  Xm = zeros (d, 1);
	end


	opts.issym = true;
	opts.isreal = true;
	opts.tol = eps;
	opts.disp = 0;

	% PCA with covariance matrix
	if n > d 
	  if verbose, fprintf ('PCA with covariance matrix: %d -> %d\n', d, dout); end
	  Xcov = X * X';
	  Xcov = (Xcov + Xcov') / (2 * n);
	  
	  if dout < d
	    [eigvec, eigval] = eigs (Xcov, dout, 'LM', opts);
	  else
	    [eigvec, eigval] = eig (Xcov);
	  end
	else
	  % PCA with gram matrix
	  if verbose, fprintf ('PCA with gram matrix: %d -> %d\n', d, dout); end
	  Xgram = X' * X;
	  Xgram = (Xgram + Xgram') / 2;
	  if dout < d
	    [eigvec, eigval] = eigs (Xgram, dout, 'LM', opts);
	  else
	    [eigvec, eigval] = eig (Xgram);
	  end
	  eigvec = single (X * eigvec);
	  eigvec = vecs_normalize (eigvec);
	end
	           

	X = eigvec' * X;
	X = single (X);
	eigval = diag(eigval);

	% We prefer a consistent order
	[~, eigord] = sort (eigval, 'descend');
	eigval = eigval (eigord);
	eigvec = eigvec (:, eigord);
	X = X(eigord, :);

end


% apply PCA-whitening, with or without dimensionality reduction
function x_ = apply_whiten (x, xm, eigvec, eigval, dout)
	if ~exist ('dout')
	  dout = size (x, 1);
	end
	x_ = bsxfun (@minus, x, xm);  % Subtract the mean
	x_ = diag(eigval(1:dout).^-0.5)*eigvec(:,1:dout)' * x_;
	x_ = replacenan (x_);
end