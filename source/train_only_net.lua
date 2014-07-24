require 'unsup_depparser'
require 'utils'
require 'dict'
require 'xlua'
require 'dpiornn_gen'
require 'dp_spec'

torch.setnumthreads(NUM_THREADS)

if #arg == 5 then
	dic_dir_path = arg[1]
	data_path = arg[2]

------------------ load dics and wemb ----------------
	init_wemb_type = nil
	wdim = tonumber(arg[3])
	if wdim == nil then
		init_wemb_type = arg[3]
	end
 
	-- load voca and embeddings
	print('load vocabulary and word embeddings')
	local L = nil

	if init_wemb_type == nil then
		print('randomly create wordembeddings')
		voca_dic = Dict:new(collobert_template)
		voca_dic:load(dic_dir_path .. '/words.lst')
		L = uniform(wdim, voca_dic.size, -0.001, 0.001)

	else
		local dic_func = nil
		local subdir = nil
		if init_wemb_type == 'collobert' then
			dic_func = collobert_template
			subdir = '/collobert/' 
		elseif init_wemb_type == 'turian_200' then 
			dic_func = turian_template
			subdir = '/turian_200/'
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
		wdim = embdim
		f:close()
		if nword ~= voca_dic.size then
			error("not match embs")
		end
	end

	local pos_dic = Dict:new()
	pos_dic:load(dic_dir_path.."/pos.lst")

	local deprel_dic = Dict:new()
	deprel_dic:load(dic_dir_path..'/deprel.lst')


-------------------------- train depparser ------------------
	-- print dp_spec file
	for line in io.lines('dp_spec.lua') do
		print(line)
	end

	print('training...')
	traindsbank_path = data_path .. '/train.conll'
	devdsbank_path = data_path .. '/test10.gold.conll'
	kbestdevdsbank_path = data_path .. '/test10-10-best.mst.conll'
	

--	traindsbank_path = data_path .. '/dev10.gold.conll'
--	devdsbank_path = data_path .. '/dev10.gold.conll'

	model_dir = arg[4]
	dim = tonumber(arg[5])

	local net_struct = { dim = dim, voca_dic = voca_dic, pos_dic = pos_dic, deprel_dic = deprel_dic,
							lookup = L, func = tanh, funcPrime = tanhPrime }

	local parser = UDepparser:new(voca_dic, pos_dic, deprel_dic)
	parser.mail_subject = model_dir

	parser:train_only_net(net_struct, traindsbank_path, devdsbank_path, kbestdevdsbank_path, model_dir)

else
	print("[dic path] [data path] [emb_model] [model dir] [dim]")
end
