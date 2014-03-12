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

if #arg == 3 then
	torch.setnumthreads(1)

	dic_dir_path = arg[1]
	data_path = arg[2]
	model_path = arg[3]

	-- load dics
	local vocaDic = Dict:new(collobert_template)
	vocaDic:load(dic_dir_path .. '/words.lst')
 
	local ruleDic = Dict:new()
	ruleDic:load(dic_dir_path .. "/rules.lst")
	ruleDic.grammar = 'CFG'

	local classDic = Dict:new()
	classDic:load(dic_dir_path .. '/classes.lst')

	local rules = {}
	for _,str in ipairs(ruleDic.id2word) do
		local comps = split_string(str, "[^ \t]+")
		local rule = {lhs = comps[1], rhs = {}}
		for i = 2,#comps do
			rule.rhs[i-1] = comps[i]
		end
		rules[#rules+1] = rule
	end

	local net = IORNN:load(model_path)

-- load data	
	print('load treebanks')
	local devtreebank	= load_treebank(data_path, vocaDic, ruleDic, classDic)
	net:eval(devtreebank, 	data_path .. '.props', 
							'../data/SRL/conll05st-release/srlconll-1.1/bin/srl-eval.pl -C')

else
	print("[dic dir path] [treebank] [model path]")
end
