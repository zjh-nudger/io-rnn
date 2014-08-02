require 'utils'

if #arg ~= 3 then
	error('input output max_len')
end

local input = arg[1]
local output = arg[2]
local max_len = tonumber(arg[3])

ret = {}
cur_s = {}
for line in io.lines(input) do
	line = trim_string(line)
	if line == '' then
		if #cur_s <= max_len then
			ret[#ret+1] = cur_s
		end
		cur_s = {}
	else
		cur_s[#cur_s+1] = line
	end	
end

local f = io.open(output, 'w')
for _,s in ipairs(ret) do
	for _,line in ipairs(s) do
		f:write(line..'\n')
	end
	f:write('\n')
end
f:close()
