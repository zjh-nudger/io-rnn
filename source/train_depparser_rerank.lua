require 'depparser_rerank'
require 'utils'
require 'dict'
require 'xlua'
require 'dpiornn_gen'
require 'dp_spec'

torch.setnumthreads(NUM_THREADS)

function load_huff_code(voca_dic, filename)
	voca_dic.code_len = torch.IntTensor(voca_dic.size)
	voca_dic.code = torch.Tensor(voca_dic.size, 50)
	voca_dic.path = torch.IntTensor(voca_dic.size, 50)

	for line in io.lines(filename) do
		local comps = split_string(line)
		local id = voca_dic:get_id(comps[1])
		local code_str = split_string(comps[2],'[01]')
		local path_str = split_string(comps[3], '[^-]+')
		
		voca_dic.code_len[id] = #code_str
		for i = 1,#code_str do
			voca_dic.code[{id,i}] = tonumber(code_str[i]) * 2 - 1 -- -1 or +1
			voca_dic.path[{id,i}] = tonumber(path_str[i])
		end
	end

	--[[ test 
	for i = 1,10 do
		local len = voca_dic.code_len[i]
		print('----------------')
		print(voca_dic.id2word[i])
		print(len)
		print(voca_dic.code[{{1,len},i}])
		print(voca_dic.path[{{1,len},i}])
	end
]]
	return voca_dic
end

if #arg == 5 then
	dic_dir_path = arg[1]..'/'
	data_path = arg[2]..'/'

------------------ load dics and wemb ----------------
	init_wemb_type = nil
	dim = tonumber(arg[3])
	if dim == nil then
		init_wemb_type = arg[3]
	end
 
	-- load voca and embeddings
	print('load vocabulary and word embeddings')
	local L = nil

	if init_wemb_type == nil then
		local subdir = 'collobert/'
		voca_dic = Dict:new(collobert_template)
		voca_dic:load(dic_dir_path .. subdir .. WORD_FILENAME)
		L = uniform(dim, voca_dic.size, -0.1, 0.1)
		load_huff_code(voca_dic, dic_dir_path..subdir..WCODE_FILENAME)


	else
		local dic_func = nil
		local subdir = nil
		if init_wemb_type == 'collobert' then
			dic_func = collobert_template
			subdir = 'collobert/' 
		elseif init_wemb_type == 'mikolov' then
			dic_func = collobert_template
			subdir = 'mikolov/' 
		end
	
		-- load dics
		voca_dic = Dict:new(dic_func)
		voca_dic:load(dic_dir_path..subdir..WORD_FILENAME)
		f = torch.DiskFile(dic_dir_path..subdir..WEMB_FILENAME, 'r')

		local info = f:readInt(2)
		local nword = info[1]	
		local embdim = info[2]	
		if torch.getdefaulttensortype() == 'torch.DoubleTensor' then
			L = torch.Tensor(f:readDouble(nword*embdim))
							:resize(nword, embdim):t()
		elseif torch.getdefaulttensortype() == 'torch.FloatTensor' then
			L = torch.Tensor(f:readFloat(nword*embdim))
							:resize(nword, embdim):t()
		end
		dim = embdim
		f:close()
		if nword ~= voca_dic.size then
			error("not match embs")
		end

		load_huff_code(voca_dic, dic_dir_path..subdir..WCODE_FILENAME)
		
	end

	local pos_dic = Dict:new()
	pos_dic:load(dic_dir_path..POS_FILENAME)

	local deprel_dic = Dict:new()
	deprel_dic:load(dic_dir_path..DEPREL_FILENAME)



-------------------------- train depparser ------------------

	print('training...')
	traindsbank_path = data_path .. TRAIN_FILENAME
	devdsbank_path = data_path .. DEV_FILENAME
	kbestdevdsbank_path = data_path .. KBEST_DEV_FILENAME

	model_dir = arg[4]
	dim = tonumber(arg[5])
	sdim = 50

	for line in io.lines('dp_spec.lua') do
		print(line)
	end
	
	local net = IORNN:new({ dim = dim, voca_dic = voca_dic, pos_dic = pos_dic, deprel_dic = deprel_dic, sdim = sdim,
							n_prevtrees = N_PREV_TREES, 
							lookup = L, func = softsign, funcPrime = softsignPrime, 
							complete_inside = CMPL_INSIDE  }) 

	local parser = Depparser:new(voca_dic, pos_dic, deprel_dic)
	parser.mail_subject = model_dir
	parser:train(net, traindsbank_path, devdsbank_path, kbestdevdsbank_path, model_dir)

--	local dsbank = parser:load_dsbank(traindsbank_path, traindsbank_path..'.grouping')
--	parser:output_raw(dsbank, '/tmp/raw')

-- for checking gradient
--	config = {lambda = 1e-4, lambda_L = 1e-7}
--	net.update_L = true
--	local traintreebank = parser:dsbank_to_treebank(parser:load_dsbank(traindsbank_path, traindsbank_path..'.grouping'))
--	net:checkGradient(traintreebank, config)

else
	print("[dictionary-dir] [treebank-dir] [emb-model] [model-dir] [dim]")
end
