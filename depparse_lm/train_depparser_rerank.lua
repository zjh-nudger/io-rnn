require 'depparser_rerank'
require 'utils'
require 'dict'
require 'xlua'
require 'dpiornn_lm'
require 'dp_spec'

torch.setnumthreads(N_THREADS)

function load_huff_code(voca_dic, filename)
	voca_dic.code_len = torch.IntTensor(voca_dic.size)
	voca_dic.code = torch.Tensor(voca_dic.size, 50)
	voca_dic.path = torch.LongTensor(voca_dic.size, 50)

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

if #arg == 4 then
	dic_dir_path = arg[1]..'/'
	data_path = arg[2]..'/'

------------------ load dics and wemb ----------------
	init_wemb_type = nil
	dim = tonumber(arg[3])
 
	-- load voca and embeddings
	print('load vocabulary and word embeddings')
	local L = nil

		voca_dic = Dict:new(collobert_template)
		voca_dic:load(dic_dir_path .. WORD_FILENAME)
		L = uniform(dim, voca_dic.size, -0.1, 0.1)
		load_huff_code(voca_dic, dic_dir_path..WCODE_FILENAME)



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

	local net = IORNN:new({ dim = dim, voca_dic = voca_dic, pos_dic = pos_dic, deprel_dic = deprel_dic,
							lookup = L, func = tanh, funcPrime = tanhPrime }) 
	print(net.pos_dic.size)
	print(net.deprel_dic.size)
	print(net.voca_dic.size)

	local parser = Depparser:new(voca_dic, pos_dic, deprel_dic)
	parser.mail_subject = model_dir
	parser:train(net, traindsbank_path, devdsbank_path, kbestdevdsbank_path, model_dir)

--[[ for checking gradient
	config = {lambda = 1e-4, lambda_L = 1e-7}
	net.update_L = true
	local traindsbank,_ = parser:load_dsbank(traindsbank_path)
	net:checkGradient(traindsbank, config)
]]
else
	print("[dictionary-dir] [treebank-dir] [dim] [model-dir]")
end
