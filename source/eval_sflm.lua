require 'sfiornnlm'
require 'dict'
require 'tree'
require 'xlua' 
p = xlua.Profiler()

torch.setnumthreads(1)

if #arg == 3 then
	local net_path = arg[1]
	local senbank_path = arg[2]
	local word_lst_path = arg[3]

	-- load word list
	print('load words...')
	local vocaDic = Dict:new()
	vocaDic:load(word_lst_path)	
	vocaDic:addword('<s>')
	vocaDic:addword('</s>')

	-- load net
	print('loading net...')
	local net = SFIORNNLM:load(net_path)
	--print(net)
	
	-- load senbank
	print('load senbank...')
	local senbank = {}
	for line in io.lines(senbank_path) do
		local words = split_string(line)
		local sen = {vocaDic:get_id('<s>')}
		for i,w in ipairs(words) do
			sen[i+1] = vocaDic:get_id(w)
		end
		sen[#sen+1] = vocaDic:get_id('</s>')
		senbank[#senbank+1] = sen
		--print(line)
		--print(sen)
	end

	-- predict words
	print('evaluating...')
	local total = 0
	local rank = 0
	local log_perplexity = 0
	local count = 0

	--torch.setnumthreads(5)
	for i,sen in ipairs(senbank) do
		print(i)
		local storage, tree = net:create_storage_and_tree(#sen)
		tree.word_id[1] = sen[1] -- should be <s>
		tree.inner[{{},{1}}]:copy(net.L[{{},{tree.word_id[1]}}])

		for j = 2,#sen do
			word_id = sen[j]
			net:extend_tree(storage, tree, sen[j])
			net:forward_inside_root(tree)
			net:forward_outside_rml(tree)
			net:forward_compute_prediction_prob(tree)

			local word_score = tree.prob[{{},1}]
			local tg_score = word_score[word_id] --print(tg_score)
			local tg_rank = torch.gt(word_score, tg_score):double():sum()
			rank = rank + tg_rank
			total = total + 1 

			log_perplexity = log_perplexity + math.log(tg_score)
			count = count + 1

			--print('------------')
			--print(rank / total)
			--_, sortedid = word_score:sort(true)
			--print(word_score)
			--print('target word : ' .. vocaDic.id2word[word_id] .. ' , ' .. tg_score .. ' , ' .. tg_rank)

			--for k = 1,10 do
			--	print(vocaDic.id2word[sortedid[k]] .. ' ' .. word_score[sortedid[k]])
			--end
		end
		print(math.exp(-log_perplexity / count))
		collectgarbage()
	end

	print(rank/ total)
	
else 
	print("[iornn] [senbank] [word list]")
end
