require 'utils'
require 'dp_spec'

Depstruct = {}
Depstruct_mt = { __index=Depstruct }

DEPSTRUCT_N_DEPS = 200

-- ROOT is indexed 1, with word_id = 0, pos_id = 0, deprel_id = 0

function Depstruct:new( input )
	local len = #input
	local ds = {
		n_words 	= len,
		word_id		= torch.zeros(len),
		pos_id		= torch.zeros(len),
		cap_id		= torch.zeros(len),
		head_id		= torch.zeros(len),
		deprel_id	= torch.zeros(len),
		n_deps		= torch.zeros(len),
		dep_id		= torch.zeros(DEPSTRUCT_N_DEPS, len),
	}

	setmetatable(ds, Depstruct_mt)

	-- set data
	for i,row in ipairs(input) do
		ds.word_id[i]	 = row[1]
		ds.pos_id[i]	 = row[2]
		ds.cap_id[i]	 = row[5]

		ds.head_id[i]	 = row[3]
		ds.deprel_id[i]  = row[4]
		
		local hid = row[3]
		if hid > 0 then -- not ROOT 
			ds.n_deps[hid] = ds.n_deps[hid] + 1
			ds.dep_id[{ds.n_deps[hid],hid}] = i
		end
	end

	return ds
end

function Depstruct:create_from_strings(rows, voca_dic, pos_dic, deprel_dic)
	local sent = { 'ROOT' }
	local input = { { 0, 0, 0, 0, 0 } }
	for i,row in ipairs(rows) do
		local comps = split_string(row)
		row = { voca_dic:get_id(comps[2]),
				pos_dic:get_id(comps[5]),
				tonumber(comps[7]) + 1,
				deprel_dic:get_id(comps[8]),
				Dict:get_cap_feature(comps[2])
			}
		input[i+1] = row
		sent[i+1] = comps[2]
	end

	return Depstruct:new(input), sent
end

function Depstruct:create_empty_tree(n_nodes, n_words)
	return {	n_nodes		= n_nodes,
				word_id		= torch.zeros(n_nodes),
				pos_id		= torch.zeros(n_nodes),
				cap_id		= torch.zeros(n_nodes),
				parent_id	= torch.zeros(n_nodes),
				n_children	= torch.zeros(n_nodes),
				children_id	= torch.zeros(DEPSTRUCT_N_DEPS, n_nodes),
				wnode_id	= torch.zeros(n_words),
				deprel_id	= torch.zeros(n_nodes) }
end

function Depstruct:to_torch_matrix_tree(id, node_id, tree)
	local id = id or 1
	local node_id = node_id or 1
	local n_nodes = self.n_words + self.n_deps:gt(0):double():sum()
	local tree = tree or self:create_empty_tree(n_nodes, self.n_words)

	local dep_id = self.dep_id[{{},id}]
	local n_deps = self.n_deps[id]


	if n_deps == 0 then
		tree.wnode_id[id] = node_id	
		tree.n_children[node_id] = 0
		tree.word_id[node_id] = self.word_id[id]
		tree.pos_id[node_id] = self.pos_id[id]
		tree.cap_id[node_id] = self.cap_id[id]
		tree.deprel_id[node_id] = self.deprel_id[id]
		node_id = node_id + 1
	else
		tree.n_children[node_id] = n_deps + 1
		if id == 0 then tree.n_children[node_id] = n_deps
		else tree.deprel_id[node_id] = self.deprel_id[id] end

		-- the head word is always the left-most child
		tree.children_id[{1,node_id}] = node_id + 1
		tree.wnode_id[id] = node_id + 1
		tree.parent_id[node_id+1] = node_id
		tree.word_id[node_id+1] = self.word_id[id]
		tree.pos_id[node_id+1] = self.pos_id[id]
		tree.cap_id[node_id+1] = self.cap_id[id]
		
		local cur_node_id = node_id + 2
		for i = 1,n_deps do
			local j = i + 1
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
print(pos_dic)
]]
