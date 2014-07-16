require 'unsup_depparser'
require 'utils'
require 'dict'
require 'xlua'
require 'dpiornn_gen'
require 'dp_spec'
require 'posix'

torch.setnumthreads(NUM_THREADS)

if #arg == 8 then

	dic_dir_path = arg[1]
	train_file = arg[2]
	dev_file = arg[3]

	model_dir = arg[5]
	dim = tonumber(arg[6])
	kbestparser1 = arg[7]
	kbestparser2 = arg[8]
	
------------------ load dics and wemb ----------------
	init_wemb_type = nil
	wdim = tonumber(arg[4])
	if wdim == nil then
		init_wemb_type = arg[4]
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

	print('================ training... ================')

	execute('mkdir ' .. model_dir)
	local subdir = model_dir..'/1'
	execute('mkdir '..subdir)
	execute('cp '..train_file..' '..subdir..'/train1.conll')
	execute('cp '..train_file..' '..subdir..'/train2.conll')
	execute('cp '..dev_file..' '..subdir..'/dev.conll')

	local parser = UDepparser:new(voca_dic, pos_dic, deprel_dic)

	for it = 1,TRAIN_N_LEAPS do
		train1_file = subdir..'/train1.conll'
		train2_file = subdir..'/train2.conll'
		dev_file = subdir..'/dev.conll'

		-- train kbestparser1
		search1_dir = subdir..'/search1/'
		log_1 = subdir..'/log-search1'
		execute({'nohup th train_unsup_dp_single.lua', dic_dir_path, train1_file, dev_file, init_wemb_type, search1_dir, dim , kbestparser1, '>&', log_1, '&'})

		-- train kbestparser2
		search2_dir = subdir..'/search2/'
		log_2 = subdir..'/log-search2'
		execute({'nohup th train_unsup_dp_single.lua', dic_dir_path, train2_file, dev_file, init_wemb_type, search2_dir, dim , kbestparser2, '>&', log_2, '&'})

		-- merge train files for next iter
		while true do
			local ok = true

			local f = io.open(search1_dir..'/done', 'r')
			if f == nil then ok = false
			else f:close() end

			f = io.open(search2_dir..'/done', 'r')
			if f == nil then ok = false
			else f:close() end

			if ok == true then break
			else posix.sleep(10) end
		end

		bank1 = parser:load_dsbank(search1_dir..'/'..kbestparser1..'-'..(TRAIN_N_ITER_IN_1_LEAP+1)..'/train.conll')
		bank2 = parser:load_dsbank(search2_dir..'/'..kbestparser2..'-'..(TRAIN_N_ITER_IN_1_LEAP+1)..'/train.conll')
		local new_bank1 = {}
		local new_bank2 = {}

		for i = 1,#bank1 do
			if math.random() > 0.5 then 
				new_bank1[i] = bank1[i]
				new_bank2[i] = bank2[i]
			else 
				new_bank1[i] = bank2[i]
				new_bank2[i] = bank1[i]
			end
		end

		subdir = model_dir..'/'..(it+1)
		execute('mkdir '..subdir)
		parser:print_parses(new_bank1, subdir..'/train1.conll')
		parser:print_parses(new_bank2, subdir..'/train2.conll')
		execute({'cp', dev_file, subdir..'/dev.conll'})

	end

else
	print("[dic-dir] [train-file] [dev-file] [wemb] [model-dir] [dim] [kbestparser1] [kbestparse2]")
end
