require 'iornn'
require 'dict'
require 'tree'

torch.setnumthreads(1)

if #arg == 4 then
	local net_path = arg[1]
	local testtreebank_path = arg[2]
	local candtreebank_path = arg[3]
	local dic_path = arg[4]

	-- load net
	print('loading net...')
	local net = IORNN:load(net_path)

-- load dic
	local f = torch.DiskFile(dic_path)
	local dic = f:readObject()
	setmetatable(dic, Dict_mt)
	f:close()
	
	-- load treebank
	local context_raw = {}
	local correct_raw = {}
	local cand_raw = {}

	print('load testtreebank...')
	local context_treebank = {}
	local correct_treebank = {}
	local i = 0
	for line in io.lines(testtreebank_path) do
		tree = Tree:create_from_string(line)
		label = tree.label
		tree = tree:to_torch_matrices(dic, 1)

		-- contexts are indexed odd
		-- correct fillers are indexed even
		if math.mod(i,2) == 0 then
			context_treebank[#context_treebank+1] = tree
			context_raw[#context_raw+1] = line
		else
			if label == 'NP' then
				correct_treebank[#correct_treebank+1] = tree
				correct_raw[#correct_raw+1] = line
			else
				context_treebank[#context_treebank] = nil
				context_raw[#context_raw] = nil
			end
		end
		i = i + 1
	end

	print('load candtreebank...')
	local cand_treebank = {}
	for line in io.lines(candtreebank_path) do
		cand_raw[#cand_raw+1] = line
		tree = Tree:create_from_string(line)
		tree = tree:to_torch_matrices(dic, 1)
		cand_treebank[#cand_treebank+1] = tree
	end

	-- parse treebank
	print('parsing treebank...')
	context_treebank = net:parse(context_treebank)
	correct_treebank = net:parse(correct_treebank)
	cand_treebank = net:parse(cand_treebank)

	-- the first tree is for computing outer meaning (the slot is at "PADDING")
	-- the next is the correct phrase


	-- outer
	local outers = {}
	for k,tree in ipairs(context_treebank) do
		for i = 1,tree.n_nodes do
			if tree.word_id[i] == dic.word2id["PADDING"] then
				outers[k] = tree.outer[{{},{i}}]:clone()
				break
			end
		end
	end

	local correct_scores = {}
	for k,tree in ipairs(correct_treebank) do
		local outer = outers[k]
		local inner = tree.inner[{{},{1}}]
		local io = net.func(net.Wwo * outer + net.Wwi * inner + net.bw)
		local score = net.Ws * io
		correct_scores[k] = score[{1,1}]
	end

	local cand_inners = torch.Tensor(net.dim, #cand_treebank)
	for k,tree in ipairs(cand_treebank) do
		cand_inners[{{},{k}}]:copy(tree.inner[{{},{1}}])
	end

	local total_rank = 0
	for k,outer in ipairs(outers) do
		local io = net.func(
				torch.repeatTensor(net.Wwo * outer, 1, cand_inners:size(2)) +
				net.Wwi * cand_inners + 
				torch.repeatTensor(net.bw, 1, cand_inners:size(2)))
		local scores = net.Ws * io
		rank = torch.gt(scores, correct_scores[k]):double():sum() / 
				cand_inners:size(2)
		print('--------------------')
		print(k)
		print(context_raw[k])
		print(correct_raw[k])
		print(rank)
		total_rank = total_rank + rank
		print(total_rank / k)
		
		_,id = scores:sort(true)
		for i=1,10 do
			print(cand_raw[id[{1,i}]])
		end
	end	
else 
	print("[iornn] [test-treebank] [candidate-treebank] [dic]")
end
