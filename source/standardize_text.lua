--[[ this code is for standardize raw text given a dictionary ]]

require 'dict'
require 'utils'

if #arg == 3 then
	local dic_path = arg[1]
	local input_path = arg[2]
	local output_path = arg[3]

-- load word emb and grammar rules
	print('load (lua)dic...')
	local f = torch.DiskFile(dic_path, 'r')
	local vocaDic = f:readObject(); setmetatable(vocaDic, Dict_mt)
	local wembs = f:readObject()
	f:close()

-- process
	local fout = io.open(output_path, 'w')
	local i = 0

	for line in io.lines(input_path) do
		local toks = split_string(line)
		local str = ''
		for _,tok in ipairs(toks) do
			str = str ..  vocaDic.id2word[vocaDic:get_id(tok)] .. ' '
		end 
		str = str .. '\n'
		fout:write(str)

		--[[print('--------')
		print(line)
		print(str)]]

		i = i + 1
		if math.mod(i,10000) == 0 then print(i) end
	end
	fout:close()

else 
	print('[dic-lua] [input] [output]')
end
