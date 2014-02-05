require 'iornn'
require 'tree'
require 'utils'
require 'dict'
require 'optim'

if #arg == 5 then
	torch.setnumthreads(1)

	we_path = arg[1]
	grrule_path = arg[2]
	treebank_dir = arg[3]
	learn_rate = tonumber(arg[4])
	n_categories = 1
	local model_dir = arg[5]

-- load word emb and grammar rules
	print('load wordembeddngs...')
	local f = torch.DiskFile(we_path, 'r')
	local vocaDic = f:readObject(); setmetatable(vocaDic, Dict_mt)
	local wembs = f:readObject()
	f:close()

	print('load grammar rules...')
	ruleDic = Dict:new(cfg_template)
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
	--print(rules)

-- create net
	print('create iornn...')
	local struct = {	Lookup = wembs, nCategory = n_categories, 
						func = tanh, funcPrime = tanhPrime }
	local net = IORNN:new(struct, rules)

	net.update_L = false

	--local net = IORNN:load('model_gr_bnc_shuf_1/model_1_1')

	lambda = 1e-4
	batchsize = 100
	alpha = 0
	beta = 1
	maxnepoch = 100
	grammar = 'CCG'

-- train
	local filenames = get_all_filenames(treebank_dir)
	local devtreebank = {}
	local adagrad_config = {learningRate = learn_rate}
	local adagrad_state = {}

	-- create bag of subtrees
	local bag_of_subtrees = {}
	bag_of_subtrees.max_phrase_len = 1
	bag_of_subtrees.only_lexicon = true

	for i = 1,vocaDic:size() do
		local word = vocaDic.id2word[i]
		if word == '(' then word = '-LRB-'
		elseif word == ')' then word = '-RRB-'
		elseif word == '[' then word = '-LSB-'
		elseif word == ']' then word = '-RSB-'
		elseif word == '{' then word = '-LCB-' 
		elseif word == '}' then word = '-RCB-' end

		local t = Tree:create_from_string(word)
		bag_of_subtrees[i] = t:to_torch_matrices(vocaDic, ruleDic, grammar) --n_categories)
	end

	net:save(model_dir .. '/model_0')

	for nepoch = 1,maxnepoch do
		for i,fn in ipairs(filenames) do
			local prefix = model_dir..'/model_'..tostring(nepoch)
			local traintreebank = {}
			print(prefix .. '_' .. i)
				
			-- reset bag of subtrees
			local next_id_bos = vocaDic:size() + 1

			print('load trees in file ' .. fn)
			for line in io.lines(treebank_dir .. '/' .. fn) do
				if line ~= '(TOP())' then
					local tree = nil
					local tree_torch = nil
					if pcall(function() 
							tree = Tree:create_from_string(line)
							tree_torch = tree:to_torch_matrices(vocaDic, ruleDic, grammar)
						end) 
					then
				-- extract subtrees 
						for _,subtree in ipairs(tree:all_nodes()) do
							local len = subtree.cover[2]-subtree.cover[1]+1
							if len > 1 and len <= bag_of_subtrees.max_phrase_len and math.random() > 0.5 and bag_of_subtrees.only_lexicon == false then
								bag_of_subtrees[next_id_bos] = subtree:to_torch_matrices(vocaDic, ruleDic, grammar) --n_categories)
								next_id_bos = next_id_bos + 1
							end
						end
		
						if tree_torch.n_nodes > 1 then
							traintreebank[#traintreebank + 1] = tree_torch
						end
					else 
						print('error: ' .. line)
					end
				end
			end

			bag_of_subtrees.size = next_id_bos - 1
			bag_of_subtrees[next_id_bos] = nil
			print(bag_of_subtrees.size)

			adagrad_config, adagrad_state = 
				net:train_with_adagrad(traintreebank, devtreebank, batchsize,
										1, lambda, alpha, beta, prefix,
										adagrad_config, adagrad_state, bag_of_subtrees)
		end
	end

else
	print("invalid arugments: [wordemb path] [rule path] [treebank dir] [learning rate] [model dir]")
end
