require 'sfiornn'
require 'tree'
require 'utils'
require 'dict'
require 'lexicon'
require 'optim'

grammar = 'CFG'

if #arg == 6 then
	torch.setnumthreads(1)

	lex_dir_path = arg[1]
	grrule_path = arg[2]
	treebank_dir = arg[3]
	dim = tonumber(arg[4])
	learn_rate = tonumber(arg[5])
	local model_dir = arg[6]

-- load lexicon
	print('load lexicon...')
	local lex = Lex:new(collobert_template)
	lex:load(lex_dir_path..'/words.lst', lex_dir_path..'/clusters.lst', lex_dir_path..'/wc.txt')
	local vocaDic = lex.voca

	print('load grammar rules...')
	ruleDic = Dict:new(grammar_template)
	ruleDic:load(grrule_path)

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
	local L = uniform(dim, lex.voca:size(), -0.1, 0.1)
	local input = {	Lookup = L, nCategory = n_categories, 
					func = tanh, funcPrime = tanhPrime,
					rules = rules, lexicon = lex }
	local net = sfIORNN:new(input)

	net.update_L = true
	lambda = 1e-4
	lambda_L = 1e-8
	batchsize = 100
	maxnepoch = 100

-- train
	local filenames = get_all_filenames(treebank_dir)
	local devtreebank = {}
	local adagrad_config = {learningRate = learn_rate}
	local adagrad_state = {}

	net:save(model_dir .. '/model_0')

	for nepoch = 1,maxnepoch do
		for i,fn in ipairs(filenames) do
			local prefix = model_dir..'/model_'..tostring(nepoch)
			local traintreebank = {}
			print(prefix .. '_' .. i)
				
			-- reset bag of subtrees
			print('load trees in file ' .. fn)
			for line in io.lines(treebank_dir .. '/' .. fn) do
				if line ~= '(())' then
					local tree = nil
					local tree_torch = nil
					if pcall(function() 
							tree = Tree:create_from_string(line)
							tree_torch = tree:to_torch_matrices(vocaDic, ruleDic, grammar)
						end) 
					then		
						if tree_torch.n_nodes > 1 then
							traintreebank[#traintreebank + 1] = tree_torch
						end
					else 
						print('error: ' .. line)
					end
				end
			end

			-- train the net
			adagrad_config, adagrad_state = 
				net:train_with_adagrad(traintreebank, devtreebank, batchsize,
										1, {lambda = lambda, lambda_L = lambda_L}, 
										prefix, adagrad_config, adagrad_state)
		end
	end

else
	print("[lex dir path] [rule path] [treebank dir] [learning rate] [model dir]")
end
