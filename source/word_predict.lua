require 'iornn'
require 'dict'
require 'tree'

torch.setnumthreads(1)

if #arg == 3 then
	local net_path = arg[1]
	local treebank_path = arg[2]
	local dic_path = arg[3]

	-- load net
	print('loading net...')
	local net = IORNN:load(net_path)

	-- load dic
	local f = torch.DiskFile(dic_path)
	local dic = f:readObject()
	setmetatable(dic, Dict_mt)
	f:close()
	
	-- load treebank
	print('load treebank...')
	local treebank = {}
	for line in io.lines(treebank_path) do
		tree = Tree:create_from_string(line)
		n_words = tree.cover[2] - tree.cover[1] + 1
		tree = tree:to_torch_matrices(dic, 1)
		treebank[#treebank+1] = tree
	end

	-- parse treebank
	print('parsing treebank...')
	treebank = net:parse(treebank)

	-- predict words
	print('evaluating...')
	local total = 0
	local correct = 0
	local func = net.func
	local Wwo = net.Wwo
	local Wwi = net.Wwi
	local bw = net.bw
	local Ws = net.Ws
	local L = net.L
	local vocabsize = L:size(2)

	local WwiL = Wwi * L
	local big_bw = torch.repeatTensor(bw, 1, vocabsize)

	torch.setnumthreads(5)
	for j,tree in ipairs(treebank) do
		print(j)
		for i = 1,tree.n_nodes do
			local outer = tree.outer[{{},{i}}]
			local word_io = func(
						torch.repeatTensor(Wwo*outer, 1, vocabsize)
						:add(WwiL):add(big_bw))
			local word_score = Ws * word_io
			_,pw_id = torch.max(word_score,2)
			
			total = total + 1
			if pw_id[{1,1}] == tree.word_id[i] then 
				correct = correct + 1
			end
			print(correct / total)
		end
		collectgarbage()
	end

	print(correct/ total)
	
else 
	print("[iornn] [treebank] [dic]")
end
