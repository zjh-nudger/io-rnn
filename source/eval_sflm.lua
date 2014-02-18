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
		local sen = {}
		for i = 1,net.n_leaves-1 do
			sen[i] = vocaDic:get_id('<s>')
		end
		for i,w in ipairs(words) do
			sen[#sen+1] = vocaDic:get_id(w)
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
		local tree = net:create_tree(sen)
		net:forward_inside(tree)
		local treeletbank = net:build_treeletbank({tree})
		net:forward_outside_rml(treeletbank)
		net:forward_compute_prediction_prob(treeletbank)

		log_perplexity = log_perplexity - treeletbank.error:sum()	
		count = count + treeletbank.n_treelets
		print(math.exp(-log_perplexity / count))
		collectgarbage()
	end
else 
	print("[iornn] [senbank] [word list]")
end
