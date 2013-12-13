require 'iornn'
require 'tree'
require 'utils'
require 'dict'
require 'optim'

if #arg == 3 then
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

	maxepoch = 2
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
	local model_dir = 'model_bnc/'
	
	for nepoch = 1,maxnepoch do
		for i,fn in ipairs(filenames) do
			local prefix = model_dir..'model_'..tostring(nepoch)
									..'_'..tostring(i)
			local traintreebank = {}

			print('load trees in file ' .. fn)
			for line in io.lines(treebank_dir .. '/' .. fn) do
				if line ~= '(TOP())' then
					tree = Tree:create_from_string(line)
					tree = tree:to_torch_matrices(dic, n_categories)
					if tree.n_nodes > 1 then
						traintreebank[#traintreebank + 1] = tree
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
	print("invalid arugments: [wordemb path] [treebank dir] [learning rate]")
end
