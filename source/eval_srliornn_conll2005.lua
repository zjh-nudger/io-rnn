require 'srliornn'
require 'tree'
require 'utils'
require 'dict'

function load_treebank(path, vocaDic, ruleDic, classDic)
	local treebank = {}

	-- load trees
	local tokens = {}
	for line in io.lines(path) do
		line = trim_string(line)
		if line ~= '' then  -- continue read tokens
			tokens[#tokens+1] = split_string(line, '[^ ]+')

		-- line == '' means end of sentence
		else -- process the whole sentence
			local tree = nil
			local tree_torch = nil
			local srls = nil
			if pcall(function() 
					tree, srls = Tree:create_CoNLL2005_SRL(tokens)
					--print(tree:to_string())
					tree_torch = tree:to_torch_matrices(vocaDic, ruleDic)
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
