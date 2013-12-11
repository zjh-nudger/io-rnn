function split_string( str , pattern )
	local pattern = pattern or "[^\t ]+"
	local toks = {}
	for k,v in string.gmatch(str, pattern) do
		toks[#toks+1] = k
	end
	return toks
end

function get_all_filenames( dir )
	local f = io.popen("ls " .. dir)
	local files = f:read("*a")
	f:close()

	return split_string(files, "[^\n]+")
end

function safe_compute_softmax(A)
	maxA,_ = A:max(1)
	maxA = torch.repeatTensor(maxA, A:size(1), 1)
	local B = torch.repeatTensor((A - maxA):exp():sum(1):log(), A:size(1), 1)
	return (A - maxA - B):exp()
end

function spearman_rho(X, Y)
	local _,x = X:sort(1); x = x:double()
	local _,y = Y:sort(1); y = y:double()
	local n = x:numel()
	local x1 = x - torch.Tensor(n):fill(x:mean())
	local y1 = y - torch.Tensor(n):fill(y:mean())
	local a = torch.cmul(x1,y1):sum()
	local b = math.sqrt(x1:pow(2):sum() * y1:pow(2):sum())
	return a/b
end
