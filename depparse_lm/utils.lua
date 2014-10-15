--require 'cutils'

function trim_string(s)
  return (s:gsub("^%s*(.-)%s*$", "%1"))
end

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

function is_dir( path )
	local f = io.open(path)
	local ok, err, code = f:read("*a")
	f:close()
	if code == 21 then
		return true
	else 
		return false
	end
end

function safe_compute_softmax(A)
	maxA,_ = A:max(1)
	maxA = torch.repeatTensor(maxA, A:size(1), 1)
	local B = torch.repeatTensor((A - maxA):exp():sum(1):log(), A:size(1), 1)
	return (A - maxA - B):exp()
end

function log_sum_of_exp(xs) 
	max,_ = xs:max(1)
	max = max[1]
	local sum = xs:add(-max):exp():sum()
	return max + math.log(sum)
end

function get_current_time()
	time = os.date("*t")
	return time.day.."/"..time.month.."/"..time.year.." "..time.hour .. ":" .. time.min .. ":" .. time.sec
end

local TH_STORAGE_REFCOUNTED = 1
local TH_STORAGE_RESIZABLE  = 2
local TH_STORAGE_FREEMEM    = 4

function sharefloatstorage(storage, data_p, sz)
	local ffi = require 'ffi'
	local storage_p = ffi.cast('THFloatStorage*', torch.pointer(storage))
	assert(bit.band(storage_p.flag, TH_STORAGE_REFCOUNTED) ~= 0)

	if storage_p.data ~= nil then
		storage_p.allocator.free(storage_p.allocatorContext, storage_p.data)
	end

	storage_p.data = ffi.cast('float*', data_p)
	if sz then
		storage_p.size = sz
	end

	storage_p.flag = TH_STORAGE_REFCOUNTED
end
