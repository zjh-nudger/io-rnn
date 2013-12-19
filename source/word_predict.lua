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

--[[
	local dim = net.dim
	local mul = 0.1	
	-- unary branch inner
	net.Wui = uniform(dim, dim, -1, 1):mul(mul)
	net.bui = uniform(dim, 1, -1, 1):mul(0)

	-- binary branch inner
	net.Wbil = uniform(dim, dim, -1, 1):mul(mul)	--left
	net.Wbir = uniform(dim, dim, -1, 1):mul(mul)	--right
	net.bbi = uniform(dim, 1, -1, 1):mul(0)

	-- binary brach outer
	net.Wbol = uniform(dim, dim, -1, 1):mul(mul)	--left sister
	net.Wbor = uniform(dim, dim, -1, 1):mul(mul)	--right sister
	net.Wbop = uniform(dim, dim, -1, 1):mul(mul)	--parent
	net.bbol = uniform(dim, 1, -1, 1):mul(0)
	net.bbor = uniform(dim, 1, -1, 1):mul(0)

	-- word ranking
	net.Wwi = uniform(2*dim, dim, -1, 1):mul(mul)	-- for combining inner, outer meanings
	net.Wwo = uniform(2*dim, dim, -1, 1):mul(mul)
	net.bw = uniform(2*dim, 1, -1, 1):mul(0)
	net.Ws = uniform(1, 2*dim, -1, 1):mul(mul)	-- for scoring

	net.L = torch.randn(net.L:size()):mul(0.0001)
]]
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

		--if #treebank > 2 then break end
	end

	-- parse treebank
	print('parsing treebank...')
	treebank = net:parse(treebank)

	-- predict words
	print('evaluating...')
	local total = 0
	local rank = 0
	local func = net.func
	local Wwo = net.Wwo
	local Wwi = net.Wwi
	local bw = net.bw
	local Ws = net.Ws
	local L = net.L
	local vocabsize = L:size(2)

	local WwiL = Wwi * L
	local n = 20000
	local big_bw = torch.repeatTensor(bw, 1, n)

	--torch.setnumthreads(5)
	for j,tree in ipairs(treebank) do
		print(j)
		for i = 1,tree.n_nodes do
			if tree.n_children[i] == 0 then
				local outer = tree.outer[{{},{i}}]
				local word_id = tree.word_id[i]

				local index = torch.rand(n-1):mul(vocabsize):floor():add(1):long()
				local small_WwiL = torch.Tensor(WwiL:size(1), n)
				small_WwiL[{{},{1,n-1}}]:copy(WwiL:index(2,index))
				small_WwiL[{{},{n}}]:copy(WwiL[{{},{word_id}}])

				local word_io = func(
						torch.repeatTensor(Wwo*outer, 1, n)
						:add(small_WwiL):add(big_bw))
				local word_score = (Ws * word_io):reshape(n)
				_,id = word_score:sort(true)
			
				_,targ_id = id:max(1)
				rank = rank + targ_id[1]
				total = total + 1
				print(rank / total)

				print('------------')
				--print(word_score)
				--print(word_score:sort(true))
				print('target word :' .. dic.id2word[word_id])
				for k = 1,10 do
					if id[k] <= n-1 then
						print(dic.id2word[index[id[k]]])
					else
						print(dic.id2word[word_id])
					end
				end
			end
		end
		collectgarbage()
	end

	print(rank/ total)
	
else 
	print("[iornn] [treebank] [dic]")
end
