require 'dict'
require 'utils'

if #arg == 5 then
	local dic_dir_path = arg[1]
	local init_wemb_type = arg[2]
	local treebank_path = arg[3]
	local output_dir = arg[4]
	local threshold = tonumber(arg[5])

	-- load voca and embeddings
	print('load vocabulary and word embeddings')
	local L = nil

	local dic_func = nil
	local subdir = nil
	if init_wemb_type == 'collobert' then
		dic_func = collobert_template
		subdir = '/origin_collobert/' 
	elseif init_wemb_type == 'turian_200' then 
		dic_func = turian_template
		subdir = '/origin_turian_200/'
	elseif init_wemb_type == 'turian_25' then 
		dic_func = turian_template
		subdir = '/origin_turian_25/'
	end
		
	-- load dics
	voca_dic = Dict:new(dic_func)
	voca_dic:load(dic_dir_path..subdir..'/words.lst')
	f = torch.DiskFile(dic_dir_path..subdir..'/embeddings.txt', 'r')

	local info = f:readInt(2)
	local nword = info[1]	
	local embdim = info[2]	
	L = torch.Tensor(f:readDouble(nword*embdim))
					:resize(nword, embdim):t()
	dim = embdim
	f:close()
	if nword ~= voca_dic.size then
		error("not match embs")
	end

	-- load data
	print('load treebank')
	local count = {}
	for line in io.lines(treebank_path) do
		local comps = split_string(line)
		if #comps > 3 then
			local word_id = voca_dic:get_id(comps[2])
			if count[word_id] == nil then count[word_id] = 1 
			else count[word_id] = count[word_id] + 1 end
		end
	end

	print('remove unused words')
	local total = 0
	for w,c in pairs(count) do
		if c >= threshold then
			total = total + 1
		end
	end

	local fw = io.open(output_dir .. '/words.lst', 'w')
	local fe = io.open(output_dir .. '/embeddings.txt', 'w')
	fe:write(total .. ' ' .. L:size(1) .. '\n')
	for w,c in pairs(count) do
		if c >= threshold then
			fw:write(voca_dic.id2word[w] .. '\n')
			local str = ''
			for i = 1,L:size(1) do
				str = str .. L[{i,w}] .. ' '
			end
			fe:write(str .. '\n')
		end
	end
	fw:close()
	fe:close()

else
	print('[dic dir] [emb type] [treebank path] [output dir] [threshold]')
end
