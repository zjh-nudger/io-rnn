require 'rnn'
require 'tree'
require 'utils'
require 'dict'
require 'optim'

if #arg == 2 then

	we_path = arg[1]
	treebank_dir = arg[2]
	n_categories = 5

-- load word emb
	print('load wordembeddngs...')
	local f = torch.DiskFile(we_path, 'r')
	local dic = f:readObject(); setmetatable(dic, Dict_mt)
	local wembs = f:readObject()
	f:close()

-- load treebank
	print('load treebank...')
	local traintreebank = {}
	for line in io.lines(treebank_dir .. '/train.txt') do
		tree = Tree:create_from_string(line)
		tree = tree:to_torch_matrices(dic.word2id, n_categories)
		traintreebank[#traintreebank + 1] = tree
	end

	local devtreebank = {}
	for line in io.lines(treebank_dir .. '/dev.txt') do
		tree = Tree:create_from_string(line)
		tree = tree:to_torch_matrices(dic.word2id, n_categories)
		devtreebank[#devtreebank + 1] = tree
	end

	
-- create rnn
	print('train rnn...')
	local struct = {	Lookup = wembs, nCategory = n_categories, 
						func = tanh, funcPrime = tanhPrime }
	local rnn = RNN:new(struct)

	maxit = 100000
	learn_rate = 0.1
	lambda = 1e-8
	batchsize = 50
	
	rnn:train_with_adagrad(traintreebank, devtreebank, batchsize,
			                maxit, learn_rate, lambda)


else
	print("invalid arugments: [wordemb path] [treebank dir]")

end
