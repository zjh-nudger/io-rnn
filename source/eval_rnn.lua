require 'rnn'
require 'dict'

if #arg == 3 then
	torch.setnumthreads(1)	

	local wordemb_path = arg[1]
	local model_path = arg[2]
	local treebank_dir = arg[3]

	-- load word embeddings
	local f = torch.DiskFile(wordemb_path, 'r')
	local dic = f:readObject()
	setmetatable(dic, Dict_mt)

	-- load model --
	print("load model at " .. model_path)
	local net = RNN:load(model_path)

	-- loadl 
	print('load treebank ' .. treebank_dir)
	local testtreebank = {}
	for line in io.lines(treebank_dir .. '/test.txt') do
		tree = Tree:create_from_string(line)
		tree = tree:to_torch_matrices(dic, net.nCat)
		testtreebank[#testtreebank + 1] = tree
	end

	-- eval
	print('eval...')
	local acc_all, acc_root = net:eval(testtreebank)
	print('all ' .. acc_all)
	print('root ' .. acc_root)


else
	print("[word emb path] [model path] [treebank dir]")
end
