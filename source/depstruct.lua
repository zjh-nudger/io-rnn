require 'utils'

Depstruct = {}
Depstruct_mt = { __index=Depstruct }

N_DEPS = 40

function Depstruct:new( input )
	local len = #input
	local ds = {
		n_words 	= len,
		word_id		= torch.zeros(len),
		pos_id		= torch.zeros(len),
		head_id		= torch.zeros(len),
		deprel_id	= torch.zeros(len),
		n_deps		= torch.zeros(len),
		dep_id		= torch.zeros(N_DEPS, len),

		root_dep_id	= torch.zeros(N_DEPS),
		root_n_deps = 0
	}

	setmetatable(ds, Depstruct_mt)

	-- set data
	for i,row in ipairs(input) do
		ds.word_id[i]	 = row[1]
		ds.pos_id[i]	 = row[2]
		ds.head_id[i]	 = row[3]
		ds.deprel_id[i]  = row[4]
		
		local hid = row[3]
		if hid == 0 then -- root
			ds.root_n_deps = ds.root_n_deps + 1
			ds.root_dep_id[ds.root_n_deps] = i
		else 
			ds.n_deps[hid] = ds.n_deps[hid] + 1
			ds.dep_id[{ds.n_deps[hid],hid}] = i
		end
	end

	return ds
end

function Depstruct:create_from_strings(input, voca_dic, pos_dic, deprel_dic)
	local sent = {}
	for i,row in ipairs(input) do
		local comps = split_string(row)
		row = { voca_dic:get_id(comps[2]),
				pos_dic:get_id(comps[5]),
				tonumber(comps[7]),
				deprel_dic:get_id(comps[8])
			}
		input[i] = row
		sent[i] = comps[2]
	end

	return Depstruct:new(input), sent
end

function Depstruct:create_empty_tree(n_nodes, n_words)
	return {	n_nodes		= n_nodes,
				word_id		= torch.zeros(n_nodes),
				parent_id	= torch.zeros(n_nodes),
				n_children	= torch.zeros(n_nodes),
				children_id	= torch.zeros(N_DEPS, n_nodes),
				wnode_id	= torch.zeros(self.n_words),
				deprel_id	= torch.zeros(n_nodes) }
end

function Depstruct:to_torch_matrix_tree(id, node_id, tree)
	local id = id or 0
	local node_id = node_id or 1
	local n_nodes = self.n_words + self.n_deps:gt(0):double():sum() + 1
	local tree = tree or self:create_empty_tree(n_nodes, self.n_words)

	local dep_id = nil
	local n_deps = 0
	if id == 0 then 
		dep_id = self.root_dep_id
		n_deps = self.root_n_deps
	else
		dep_id = self.dep_id[{{},id}]
		n_deps = self.n_deps[id]
	end

	if n_deps == 0 then
		tree.wnode_id[id] = node_id	
		tree.n_children[node_id] = 0
		tree.word_id[node_id] = self.word_id[id]
		tree.deprel_id[node_id] = self.deprel_id[id]
		node_id = node_id + 1
	else
		tree.n_children[node_id] = n_deps + 1
		if id == 0 then tree.n_children[node_id] = n_deps
		else tree.deprel_id[node_id] = self.deprel_id[id] end

		-- the word is always the left-most child
		if id ~= 0 then
			tree.children_id[{1,node_id}] = node_id + 1
			tree.wnode_id[id] = node_id + 1
			tree.parent_id[node_id+1] = node_id
			tree.word_id[node_id+1] = self.word_id[id]
		end
		
		local cur_node_id = node_id + 2
		if id == 0 then cur_node_id = node_id + 1 end

		for i = 1,n_deps do
			local j = i + 1
			if id == 0 then j = i end
			tree.children_id[{j,node_id}] = cur_node_id
			tree.parent_id[cur_node_id] = node_id
			tree,cur_node_id = self:to_torch_matrix_tree(dep_id[i], cur_node_id, tree)
		end
		node_id = cur_node_id
	end

	return tree, node_id
end

--[[ test
require 'dict'

torch.setnumthreads(1)

local voca_dic = Dict:new(collobert_template)
voca_dic:load('../data/wsj-dep/universal/dic/collobert/words.lst')
 
local pos_dic = Dict:new(cfg_template)
pos_dic:load("../data/wsj-dep/universal/dic/pos.lst")

local deprel_dic = Dict:new()
deprel_dic:load('../data/wsj-dep/universal/dic/deprel.lst')

tokens = {}
for line in io.lines('../data/wsj-dep/universal/data/test.conll') do
	line = trim_string(line)
	if line == '' then
		print(tokens)
		local ds = Depstruct:create_from_strings(tokens, voca_dic, pos_dic, deprel_dic)
		tree,_ = ds:to_torch_matrix_tree()
		for k,v in pairs(tree) do
			print(k)
			print(v)
		end
		tokens = {}
		break
	else 
		tokens[#tokens+1] = line
	end
end
print(deprel_dic)
]]
