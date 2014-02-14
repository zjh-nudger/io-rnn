require 'iornn'
require 'dict'
require 'tree'

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
	local grammar = 'CCG'

	-- load net
	print('loading net...')
	local net = IORNN:load(net_path)
	--print(net)

	-- load dic
	local f = torch.DiskFile(dic_path)
	local vocaDic = f:readObject()
	setmetatable(vocaDic, Dict_mt)
	f:close()
	
	-- load treebank
	print('load treebank...')
	local treebank = {}
	for line in io.lines(treebank_path) do
		tree = Tree:create_from_string(line)
		--tree:binarize(true,true)
		--print(tree:to_string())

		n_words = tree.cover[2] - tree.cover[1] + 1
		tree = tree:to_torch_matrices(vocaDic, ruleDic, grammar)
		treebank[#treebank+1] = tree

		--if #treebank > 2 then break end
	end

	-- parse treebank
	print('parsing treebank...')
	treebank = net:parse(treebank)
	
	--torch.setnumthreads(5)
	local target = {['table'] = 1, ['dog'] = 1}
	for j,tree in ipairs(treebank) do
		--print(j)
		for i = 1,tree.n_nodes do
			local word = vocaDic.id2word[tree.word_id[i]]
			if tree.n_children[i] == 0 and target[word] == 1  then
				local str = word..tostring(j)
				local mat = --[[net.Wwo * ]]tree.outer[{{},{i}}]
				for j=1,mat:size(1) do
					str = str .. '\t' .. tostring(mat[{j,1}])
				end
				print(str)
			end
		end
		collectgarbage()
	end
	
else 
	print("[iornn] [treebank] [dic] [rule path]")
end
