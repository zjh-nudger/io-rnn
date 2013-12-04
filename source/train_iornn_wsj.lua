require 'iornn'
require 'tree'
require 'utils'
require 'dict'
require 'optim'

if #arg == 3 then
	torch.setnumthreads(2)

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

-- load treebank
	print('load treebank...')
	local traintreebank = {}
	for line in io.lines(treebank_dir .. '/wsj-02-21.mrg') do
		tree = Tree:create_from_string(line)
		tree = tree:to_torch_matrices(dic, n_categories)
		if tree.n_nodes > 1 then
			traintreebank[#traintreebank + 1] = tree
		end
	end

	local devtreebank = {}
	
-- create rnn
	print('train rnn...')
	local struct = {	Lookup = wembs, nCategory = n_categories, 
						func = tanh, funcPrime = tanhPrime }
	local net = IORNN:new(struct)

	maxit = 100000
	lambda = 1e-4
	batchsize = 100
	alpha = 0
	beta = 1
	
	net:train_with_adagrad(traintreebank, devtreebank, batchsize,
			                maxit, learn_rate, lambda, alpha, beta)


else
	print("invalid arugments: [wordemb path] [treebank dir] [learning rate]")
end
