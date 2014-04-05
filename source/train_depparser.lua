require 'depparser'
require 'utils'
require 'dict'
require 'xlua'
require 'dpiornn'

torch.setnumthreads(1)

if #arg == 4 then
	dic_dir_path = arg[1]
	data_path = arg[2]

------------------ load dics and wemb ----------------
	init_wemb_type = nil
	dim = tonumber(arg[3])
	if dim == nil then
		init_wemb_type = arg[3]
	end

	rule_type = arg[4]
	weight_learn_rate = tonumber(arg[5])
	voca_learn_rate = tonumber(arg[6])
	local model_dir = arg[7]
 
	-- load voca and embeddings
	print('load vocabulary and word embeddings')
	local L = nil

	if init_wemb_type == nil then
		voca_dic = Dict:new(collobert_template)
		voca_dic:load(dic_dir_path .. '/words.lst')
		L = uniform(dim, voca_dic:size(), -0.1, 0.1)

	else
		local dic_func = nil
		local subdir = nil
		if init_wemb_type == 'collobert' then
			dic_func = collobert_template
			subdir = '/collobert/' 
		elseif init_wemb_type == 'turian_25' then 
			dic_func = turian_template
			subdir = '/turian_25/'
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
	end

	local pos_dic = Dict:new()
	pos_dic:load("../data/wsj-dep/universal/dic/pos.lst")

	local deprel_dic = Dict:new()
	deprel_dic:load('../data/wsj-dep/universal/dic/deprel.lst')



-------------------------- train depparser ------------------

	print('training...')
	traintreebank_path = data_path .. 'train.conll'
	devtreebank_path = data_path .. 'dev.conll'
	model_dir = arg[4]

	local parser = Depparser:new(L, voca_dic, pos_dic, deprel_dic)
	parser:train(traintreebank_path, devtreebank_path, model_dir)

else
	print("[dic dir path] [treebank] [dim/emb_model] [model dir]")
end
