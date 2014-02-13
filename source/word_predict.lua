require 'iornn'
require 'dict'
require 'tree'
require 'xlua' 
p = xlua.Profiler()

torch.setnumthreads(1)

if #arg == 4 then
	local net_path = arg[1]
	local treebank_path = arg[2]
	local dic_path = arg[3]
	local rule_path = arg[4]

	-- load grammar rules
	print('load grammar rules...')
	ruleDic = Dict:new(cfg_template)
	ruleDic:load(rule_path)
	local grammar = 'CFG'

	local rules = {}
	for _,str in ipairs(ruleDic.id2word) do
		local comps = split_string(str, "[^ \t]+")
		local rule = {lhs = comps[1], rhs = {}}
		for i = 2,#comps do
			rule.rhs[i-1] = comps[i]
		end
		rules[#rules+1] = rule
	end

	-- load net
	print('loading net...')
	local net = IORNN:load(net_path)
	--print(net)

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
	local vocaDic = f:readObject()
	setmetatable(vocaDic, Dict_mt)
	f:close()
	
	-- load treebank
	print('load treebank...')
	local treebank = {}
	local rawtreebank = {}
	for line in io.lines(treebank_path) do
		print(line)
		if line ~= '(())' then
			tree = Tree:create_from_string(line)
			--tree:binarize(true,true)
			--print(tree:to_string())

			n_words = tree.cover[2] - tree.cover[1] + 1
			tree = tree:to_torch_matrices(vocaDic, ruleDic, grammar)
			treebank[#treebank+1] = tree
			rawtreebank[#rawtreebank+1] = line

			--if #treebank > 2 then break end
		end
	end

	-- parse treebank
	print('parsing treebank...')
	treebank = net:parse(treebank)

	-- predict words
	print('evaluating...')
	local total = 0
	local rank = 0
	local Wwo = net.Wwo
	local Wwi = net.Wwi
	local bw = net.bw
	local Ws = net.Ws
	local L = net.L
	local vocabsize = L:size(2)

	local WwiL = Wwi * L
	local WwiL_bw = torch.repeatTensor(bw, 1, vocabsize):add(WwiL)

	--torch.setnumthreads(5)
	for j,tree in ipairs(treebank) do
		print(j)
		for i = 1,tree.n_nodes do
			if tree.n_children[i] == 0 then --and vocaDic.id2word[tree.word_id[i]] == 'table' then
				local outer = tree.outer[{{},{i}}]
				
				local word_id = tree.word_id[i]

				p:start('word io')				
				local word_io = tanh(
						torch.repeatTensor(Wwo*outer, 1, vocabsize)
						:add(WwiL_bw))
				p:lap('word io')
				p:start('word score')
				local word_score = (Ws * word_io):reshape(vocabsize)
				p:lap('word score')
			
				p:start('rank')
				local tg_score = word_score[word_id] --print(tg_score)
				rank = rank + torch.gt(word_score, tg_score):double():sum()
				total = total + 1
				p:lap('rank')
				p:printAll() 

				print('------------')
				print(rank / total)
				print(rawtreebank[j])
				_, sortedid = word_score:sort(true)
				print('target word :' .. vocaDic.id2word[word_id] .. ', ' .. tg_score)
				--print('foe' .. word_score[vocaDic.word2id['dog']])
				--print(outer)
				for k = 1,10 do
					print(vocaDic.id2word[sortedid[k]] .. ' ' .. word_score[sortedid[k]])
				end
			end
		end
		collectgarbage()
	end

	print(rank/ total)
	
else 
	print("[iornn] [treebank] [dic] [rule path]")
end
