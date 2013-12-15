require 'cutils'

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

function kendall_tau_b(x, y)
	local nc = 0
	local nd = 0
	local nx = 0
	local ny = 0
	local n = x:size(1)

	for i = 1,n-1 do
	for j = i+1,n do
		if x[i] > x[j] then 
			if y[i] > y[j] then nc = nc + 1 
			elseif y[i] < y[j] then nd = nd + 1
			else ny = ny + 1 end
		elseif x[i] < x[j] then
			if y[i] > y[j] then nd = nd + 1
			elseif y[i] < y[j] then nc = nc + 1
			else ny = ny + 1 end
		else 
			if y[i] ~= y[j] then nx = nx + 1 end
		end
	end
	end

	return (nc - nd) / math.sqrt((nc+nd+nx)*(nc+nd+ny))
--[[
	x1,_ = x:sort()
	y1,_ = y:sort()
	local nx = 0
	local ny = 0
	local tx = 1
	local ty = 1 
	for i = 1,n-1 do
		if x1[i] == x1[i+1] then tx = tx + 1
		else 
			nx = nx + tx * (tx-1) / 2
			tx = 1
		end

		if y1[i] = y1[i+1] then ty = ty + 1
		else 
			ny = ny + ty * (ty-1) / 2
			ty = 1
		end
	end
	nx = nx + tx * (tx-1) / 2
	ny = ny + ty * (ty-1) / 2
	local n0 = n*(n-1) / 2

	return (nc-nd) / math.sqrt((n0-nx)*(n0-ny))
]]
end

