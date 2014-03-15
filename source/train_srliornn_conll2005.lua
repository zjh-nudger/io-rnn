require 'srliornn'
require 'tree'
require 'utils'
require 'dict'

function load_treebank(path, vocaDic, ruleDic, classDic)
	local treebank = {}

	print('load ' .. path..'.parse.head')
	local head_treebank = {}
	for line in io.lines(path .. '.parse.head') do
		local tree = Tree:create_from_string(line)
		if pcall(function()
					tree = tree:to_torch_matrices(vocaDic, ruleDic, true)
				end)
		then
			head_treebank[#head_treebank+1] = tree
		else
			head_treebank[#head_treebank+1] = {n_nodes = 0}
		end
	end

	print('load ' .. path)
	-- load trees
	local tokens = {}
	local i = 0
	for line in io.lines(path) do
		line = trim_string(line)
		if line ~= '' then  -- continue read tokens
			tokens[#tokens+1] = split_string(line, '[^ ]+')

		-- line == '' means end of sentence
		else -- process the whole sentence
			i = i + 1
			local tree = nil
			local tree_torch = nil
			local srls = nil
			if pcall(function() 
					tree, srls = Tree:create_CoNLL2005_SRL(tokens)
					tree_torch = head_treebank[i]
					if tree_torch.n_nodes == 0 then
						error('empty tree')
					end
					--print(tree:to_string())
					--tree_torch = tree:to_torch_matrices(vocaDic, ruleDic, true)
				end) 
			then
				if #srls == 0 then
					srls = {{}}
				end
				for _,srl in ipairs(srls) do
					local t = Tree:copy_torch_matrix_tree(tree_torch)
					t = Tree:add_srl_torch_matrix_tree(t, srl, classDic)
					treebank[#treebank+1] = t
				end
			else 
				print('error: ')
				print(tokens)
			end
			tokens = {}
		end
	end
	return treebank
end

if #arg == 7 then
	torch.setnumthreads(1)

	dic_dir_path = arg[1]
	data_path = arg[2]

	-- check if randomly initial word embeddings
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
	local vocaDic = nil

	if init_wemb_type == nil then
		vocaDic = Dict:new(collobert_template)
		vocaDic:load(dic_dir_path .. '/words.lst')
		L = uniform(dim, vocaDic:size(), -0.1, 0.1)

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
		vocaDic = Dict:new(dic_func)
		vocaDic:load(dic_dir_path..subdir..'/words.lst')
		f = torch.DiskFile(dic_dir_path..subdir..'/embeddings.txt', 'r')

		local info = f:readInt(2)
		local nword = info[1]	
		local embdim = info[2]	
		L = torch.Tensor(f:readDouble(nword*embdim))
					:resize(nword, embdim):t()
		dim = embdim
		f:close()
		if nword ~= vocaDic:size() then
			error("not match embs")
		end
	end

	print('load rule and class lists')
	local ruleDic = Dict:new()
	ruleDic:load(dic_dir_path .. "/rules_"..rule_type..".lst")
	ruleDic.grammar = 'CFG'

	local classDic = Dict:new()
	classDic:load(dic_dir_path .. '/classes.lst')

	local rules = {}
	for _,str in ipairs(ruleDic.id2word) do
		local comps = split_string(str, "[^ \t]+")
		local rule = {lhs = comps[1], rhs = {}}
		for i = 2,#comps do
			rule.rhs[i-1] = comps[i]
		end
		rules[#rules+1] = rule
	end

-- create net
	print('create iornn...')

	local input = {	lookup = L, voca = vocaDic, class = classDic, 
					func = tanh, funcPrime = tanhPrime,
					rules = rules }
	local net = IORNN:new(input)

	net.update_L = true
	lambda = 1e-4
	lambda_L = 1e-10
	batchsize = 100

	maxnepoch = 100

-- load data	
	print('load treebanks')
	local devtreebank	= load_treebank(data_path .. '/dev-set', vocaDic, ruleDic, classDic)
	local traintreebank	= load_treebank(data_path .. '/train-set', vocaDic, ruleDic, classDic)
	print(#traintreebank .. ' training trees')

	-- shuf the traintreebank
	new_i = torch.randperm(#traintreebank)
	temp = {}
	for i = 1,#traintreebank do
		temp[i] = traintreebank[new_i[i]]
	end
	traintreebank = temp

-- train
	local adagrad_config = {weight_learningRate = weight_learn_rate,
							voca_learningRate = voca_learn_rate}
	local adagrad_state = {}

	net:save(model_dir .. '/model_0')
	local prefix = model_dir..'/model'
	adagrad_config, adagrad_state = net:train_with_adagrad(traintreebank, devtreebank, batchsize,
															maxnepoch, {lambda = lambda, lambda_L = lambda_L}, 
															prefix, adagrad_config, adagrad_state)

else
	print("[dic dir path] [treebank] [dim/emb_model] [weight & voca learning rate] [model dir]")
end
