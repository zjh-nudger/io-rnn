require 'srliornn'
require 'tree'
require 'utils'
require 'dict'

function load_treebank(path, vocaDic, ruleDic, classDic)
	local treebank = {}

	-- load trees
	local tokens = {}
	for line in io.lines(path) do
		line = trim_string(line)
		if line ~= '' then  -- continue read tokens
			tokens[#tokens+1] = split_string(line, '[^ ]+')

		-- line == '' means end of sentence
		else -- process the whole sentence
			local tree = nil
			local tree_torch = nil
			local srls = nil
			if pcall(function() 
					tree, srls = Tree:create_CoNLL2005_SRL(tokens)
					--print(tree:to_string())
					tree_torch = tree:to_torch_matrices(vocaDic, ruleDic)
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

if #arg == 5 then
	torch.setnumthreads(1)

	dic_dir_path = arg[1]
	data_path = arg[2]
	dim = tonumber(arg[3])
	learn_rate = tonumber(arg[4])
	local model_dir = arg[5]

	-- load dics
	local vocaDic = Dict:new(collobert_template)
	vocaDic:load(dic_dir_path .. '/words.lst')
 
	local ruleDic = Dict:new()
	ruleDic:load(dic_dir_path .. "/rules.lst")
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
	local L = uniform(dim, vocaDic:size(), -0.1, 0.1)
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
	local adagrad_config = {learningRate = learn_rate}
	local adagrad_state = {}

	net:save(model_dir .. '/model_0')
	local prefix = model_dir..'/model_'
	adagrad_config, adagrad_state = net:train_with_adagrad(traintreebank, devtreebank, batchsize,
															maxnepoch, {lambda = lambda, lambda_L = lambda_L}, 
															prefix, adagrad_config, adagrad_state)

else
	print("[dic dir path] [treebank] [dim] [learning rate] [model dir]")
end
