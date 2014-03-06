--[[ this code is for standardize raw text given a dictionary ]]

require 'dict'
require 'utils'

if #arg == 3 then
	local dic_path = arg[1]
	local input_path = arg[2]
	local output_path = arg[3]

-- load word emb and grammar rules
	print('load dic...')
	local dic = Dict:new() --mssc_template)
	dic:load(dic_path)	

-- process
	local fout = io.open(output_path, 'w')
	local i = 0

	for line in io.lines(input_path) do
		local toks = split_string(line)
		local str = ''
		for _,tok in ipairs(toks) do
			str = str ..  dic.id2word[dic:get_id(tok)] .. ' '
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
	print('[dic] [input] [output]')
end
