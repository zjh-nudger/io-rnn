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
--[[	
	local struct = {	Lookup = wembs, nCategory = n_categories, 
						func = tanh, funcPrime = tanhPrime }
	local net = IORNN:new(struct)
]]
	local net = IORNN:load('model_bnc_full_cw_50_next/model_init')
	lambda = 1e-4
	batchsize = 100
	alpha = 0
	beta = 1
	maxnepoch = 100

	net.update_L = false

-- train
	local filenames = get_all_filenames(treebank_dir)
	local devtreebank = {}
	local adagrad_config = {learningRate = learn_rate}
	local adagrad_state = {}
	local model_dir = arg[4]

	for nepoch = 1,maxnepoch do
		for i,fn in ipairs(filenames) do
			local prefix = model_dir..'model_'..tostring(nepoch)
			local traintreebank = {}
			print(prefix .. '_' .. i)

			print('load trees in file ' .. fn)
			for line in io.lines(treebank_dir .. '/' .. fn) do
				if line ~= '(TOP())' then
					local tree = nil
					if pcall(function() tree = Tree:create_from_string(line) end) then
						tree = tree:to_torch_matrices(dic, n_categories)
						if tree.n_nodes > 1 then
							traintreebank[#traintreebank + 1] = tree
						end
					else 
						print('error: ' .. line)
					end
				end
			end
		
			adagrad_config, adagrad_state = 
				net:train_with_adagrad(traintreebank, devtreebank, batchsize,
										1, lambda, alpha, beta, prefix,
										adagrad_config, adagrad_state)
		end
	end

else
	print("invalid arugments: [wordemb path] [treebank dir] [learning rate] [model dir]")
end
