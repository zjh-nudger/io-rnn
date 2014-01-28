require 'iornn'
require 'tree'
require 'utils'
require 'dict'
require 'optim'

if #arg == 4 then
	torch.setnumthreads(1)

	we_path = arg[1]
	treebank_dir = arg[2]
	learn_rate = tonumber(arg[3])
	n_categories = 1

-- load word emb
	print('load wordembeddngs...')
	local f = torch.DiskFile(we_path, 'r')
	local dic = f:readObject(); setmetatable(dic, Dict_mt)
	local wembs = f:readObject()
	f:close()

-- create net
	local struct = {	Lookup = wembs, nCategory = n_categories, 
						func = tanh, funcPrime = tanhPrime }
	local net = IORNN:new(struct)

	net.update_L = false

	--local net = IORNN:load('model_bnc_shuf_2/model_32_1')

	lambda = 1e-4
	batchsize = 100
	alpha = 0
	beta = 1
	maxnepoch = 100

-- train
	local filenames = get_all_filenames(treebank_dir)
	local devtreebank = {}
	local adagrad_config = {learningRate = learn_rate}
	local adagrad_state = {}
	local model_dir = arg[4]

	-- create bag of subtrees
	local bag_of_subtrees = {}
	bag_of_subtrees.max_phrase_len = 3
	local n_subtrees = 2*dic:size()

	for i = 1,dic:size() do
		local word = dic.id2word[i]
		if word == '(' then word = '-LRB-'
		elseif word == ')' then word = '-RRB-'
		elseif word == '[' then word = '-LSB-'
		elseif word == ']' then word = '-RSB-'
		elseif word == '{' then word = '-LCB-' 
		elseif word == '}' then word = '-RCB-' end

		local str = '(X ' .. word .. ')'
		local t = Tree:create_from_string(str)
		bag_of_subtrees[i] = t:to_torch_matrices(dic, n_categories)
	end

	net:save(model_dir .. '/model_0')

	for nepoch = 1,maxnepoch do
		for i,fn in ipairs(filenames) do
			local prefix = model_dir..'/model_'..tostring(nepoch)
			local traintreebank = {}
			print(prefix .. '_' .. i)
				
			-- reset bag of subtrees
			local next_id_bos = dic:size() + 1

			print('load trees in file ' .. fn)
			for line in io.lines(treebank_dir .. '/' .. fn) do
				if line ~= '(TOP())' then
					local tree = nil
					if pcall(function() tree = Tree:create_from_string(line) end) then
						-- extract subtrees 
						for _,subtree in ipairs(tree:all_nodes()) do
							local len = subtree.cover[2]-subtree.cover[1]+1
							if len > 1 and len <= net.max_phrase_len and math.random() > 0.5 then
								bag_of_subtrees[next_id_bos] = subtree:to_torch_matrices(dic, n_categories)
								next_id_bos = next_id_bos + 1
								bag_of_subtrees[next_id_bos] = nil
							end
						end
		
						tree = tree:to_torch_matrices(dic, n_categories)
						if tree.n_nodes > 1 then
							traintreebank[#traintreebank + 1] = tree
						end
					else 
						print('error: ' .. line)
					end
				end
			end

			print(#bag_of_subtrees)	
			adagrad_config, adagrad_state = 
				net:train_with_adagrad(traintreebank, devtreebank, batchsize,
										1, lambda, alpha, beta, prefix,
										adagrad_config, adagrad_state, bag_of_subtrees)
		end
	end

else
	print("invalid arugments: [wordemb path] [treebank dir] [learning rate] [model dir]")
end
