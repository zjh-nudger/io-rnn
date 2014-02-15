require 'iornnlm'
require 'dict'
require 'tree'
require 'xlua' 
p = xlua.Profiler()

torch.setnumthreads(1)

if #arg == 3 then
	local net_path = arg[1]
	local senbank_path = arg[2]
	local dic_path = arg[3]

	-- load net
	print('loading net...')
	local net = IORNNLM:load(net_path)
	--print(net)

	-- load dic
	local f = torch.DiskFile(dic_path)
	local vocaDic = f:readObject()
	setmetatable(vocaDic, Dict_mt)
	f:close()
	
	-- load senbank
	print('load senbank...')
	local senbank = {}
	for line in io.lines(senbank_path) do
		local words = split_string(line)
		local sen = {vocaDic:get_id('PADDING')}
		for i,w in ipairs(words) do
			sen[i+1] = vocaDic:get_id(w)
		end
		sen[#sen+1] = vocaDic:get_id('PADDING')
		senbank[#senbank+1] = sen
	end

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
	for i,sen in ipairs(senbank) do
		print(i)
		local storage, tree = net:create_storage_and_tree(#sen)
		tree.word_id[1] = sen[1] -- should be PADDNG
		tree.inner[{{},{1}}]:copy(net.L[{{},{tree.word_id[1]}}])

		for j = 2,#sen do
			net:extend_tree(storage, tree, sen[j])
			net:forward_inside_root(tree)
			net:forward_outside_rml(tree)

			local outer = tree.rml_outer
			local word_id = tree.word_id[tree.n_nodes]
			local word_io = tanh(
						torch.repeatTensor(Wwo*outer, 1, vocabsize)
						:add(WwiL_bw))
			local word_score = (Ws * word_io):reshape(vocabsize)
			local tg_score = word_score[word_id] --print(tg_score)
			local tg_rank = torch.gt(word_score, tg_score):double():sum()
			rank = rank + tg_rank
			total = total + 1 

			print('------------')
			print(rank / total)
			_, sortedid = word_score:sort(true)
			print('target word : ' .. vocaDic.id2word[word_id] .. ' , ' .. tg_score .. ' , ' .. tg_rank)

			for k = 1,10 do
				print(vocaDic.id2word[sortedid[k]] .. ' ' .. word_score[sortedid[k]])
			end
		end
		collectgarbage()
	end

	print(rank/ total)
	
else 
	print("[iornn] [senbank] [dic]")
end
