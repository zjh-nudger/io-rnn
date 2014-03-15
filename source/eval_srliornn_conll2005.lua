require 'srliornn'
require 'tree'
require 'utils'
require 'dict'

function load_treebank(path, vocaDic, ruleDic, classDic)
	local treebank = {}

	print('load ' .. path..'.parse.head')
	local head_treebank = {}
	for line in io.lines(path .. '.parse.head') do
		local tree = Tree:create_from_string(line)
		if pcall(function()
					tree = tree:to_torch_matrices(vocaDic, ruleDic, true)
				end)
		then
			head_treebank[#head_treebank+1] = tree
		else
			head_treebank[#head_treebank+1] = {n_nodes = 0}
		end
	end

	print('load ' .. path)
	-- load trees
	local tokens = {}
	local i = 0
	for line in io.lines(path) do
		line = trim_string(line)
		if line ~= '' then  -- continue read tokens
			tokens[#tokens+1] = split_string(line, '[^ ]+')

		-- line == '' means end of sentence
		else -- process the whole sentence
			i = i + 1
			local tree = nil
			local tree_torch = nil
			local srls = nil
			if pcall(function() 
					tree, srls = Tree:create_CoNLL2005_SRL(tokens)
					tree_torch = head_treebank[i]
					if tree_torch.n_nodes == 0 then
						error('empty tree')
					end
					--print(tree:to_string())
					--tree_torch = tree:to_torch_matrices(vocaDic, ruleDic, true)
				end) 
			then
				if #srls == 0 then
					srls = {{}}
				end
				for _,srl in ipairs(srls) do
					local t = Tree:copy_torch_matrix_tree(tree_torch)
					t = Tree:add_srl_torch_matrix_tree(t, srl, classDic)
					treebank[#treebank+1] = t
				end
			else 
				print('error: ')
				print(tokens)
			end
			tokens = {}
		end
	end
	return treebank
end

if #arg == 4 then
	torch.setnumthreads(1)

	dic_dir_path = arg[1]
	data_path = arg[2]
	rule_type = arg[3]
	model_path = arg[4]

	-- load dics
 
	local ruleDic = Dict:new()
	ruleDic:load(dic_dir_path .. '/rules_'..rule_type..'.lst')
	ruleDic.grammar = 'CFG'

	local net = IORNN:load(model_path)
	local vocaDic = net.voca
	local classDic = net.class

-- load data	
	print('load treebanks')
	local devtreebank	= load_treebank(data_path, vocaDic, ruleDic, classDic)
	net:eval(devtreebank, 	data_path .. '.props', 
							'../data/SRL/conll05st-release/srlconll-1.1/bin/srl-eval.pl -C')

else
	print("[dic dir path] [treebank] [rule type (min,200,800)][model path]")
end
