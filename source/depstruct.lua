require 'utils'
require 'dp_spec'

Depstruct = {}
Depstruct_mt = { __index=Depstruct }

-- ROOT is indexed 1, with word = 0, pos = 0, deprel = 0

function Depstruct:new( input )
	local len = #input
	local ds = {
		n_words	= len,
		word	= torch.IntTensor(len):fill(0),
		pos		= torch.ByteTensor(len):fill(0),
		cap		= torch.ByteTensor(len):fill(0),
		head	= torch.ByteTensor(len):fill(0),
		deprel	= torch.ByteTensor(len):fill(0),
		n_deps	= torch.ByteTensor(len):fill(0),
		dep		= torch.ByteTensor(DEPSTRUCT_N_DEPS, len):fill(0)
	}

	setmetatable(ds, Depstruct_mt)

	-- set data
	for i,row in ipairs(input) do
		ds.word[i]	 = row[1]
		ds.pos[i]	 = row[2]
		ds.cap[i]	 = row[5]

		ds.head[i]	 = row[3]
		ds.deprel[i] = row[4]
		
		local hid = row[3]
		if hid > 0 then -- not ROOT 
			ds.n_deps[hid] = ds.n_deps[hid] + 1
			ds.dep[{ds.n_deps[hid],hid}] = i
		end
	end
	ds:get_cover()

	return ds
end

function Depstruct:create_from_strings(rows, voca_dic, pos_dic, deprel_dic)
	local sent = { 'ROOT' }
	local input = { { 1, 1, 0, 1, 1 } } -- set mocking value for ROOT
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

function Depstruct:get_cover(id)
	if id == nil then
		id = 1
		self.cover = torch.zeros(2, self.n_words)
	end
	local n_deps = self.n_deps[id]
	
	if n_deps == 0 then 
		self.cover[{1,id}] = id
		self.cover[{2,id}] = id
	else
		for i = 1,n_deps do
			self:get_cover(self.dep[{i,id}])
		end
		self.cover[{1,id}] = math.min(id, self.cover[{1,self.dep[{1,id}]}])
		self.cover[{2,id}] = 
			math.max(id, self.cover[{2,self.dep[{n_deps,id}]}])
	end
end

-- note that number of words == number of nodes 
-- because the first word is ROOT
function Depstruct:create_empty_tree(n_nodes)
	return {	n_nodes		= n_nodes,
				word		= torch.IntTensor(n_nodes):fill(0),
				pos			= torch.ByteTensor(n_nodes):fill(0),
				cap			= torch.ByteTensor(n_nodes):fill(0),
				parent		= torch.ByteTensor(n_nodes):fill(0),
				dist		= torch.ByteTensor(n_nodes):fill(0),	-- distance to parent
				dir			= torch.ByteTensor(n_nodes):fill(0),	-- which parent side the node is on
				[DIR_L]	= {	n_children	= torch.ByteTensor(n_nodes):fill(0),	-- 1: left, 2: right
							children	= torch.ByteTensor(DEPSTRUCT_N_DEPS, n_nodes):fill(0) },
				[DIR_R]	= {	n_children	= torch.ByteTensor(n_nodes):fill(0),
							children	= torch.ByteTensor(DEPSTRUCT_N_DEPS, n_nodes):fill(0) },
				wnode		= torch.ByteTensor(n_nodes):fill(0),
				deprel		= torch.ByteTensor(n_nodes):fill(0) }
end

function Depstruct:delete_tree(tree)
	for k,_ in pairs(tree) do
		tree[k] = nil
	end
	tree = nil
	return nil
end

function Depstruct:to_torch_matrix_tree(id, node, tree)
	local id = id or 1
	local node = node or 1
	local n_nodes = self.n_words 
	local tree = tree or self:create_empty_tree(n_nodes)

	if id == 1 then
		tree.flat_emb = self.flat_emb
	end

	local dep = self.dep[{{},id}]
	local n_deps = self.n_deps[id]

	tree.wnode[id]		= node
	tree.word[node]		= self.word[id]
	tree.pos[node]		= self.pos[id]
	tree.cap[node]		= self.cap[id]
	tree.deprel[node]	= self.deprel[id]

	if n_deps == 0 then
		node = node + 1

	else
		local cur_node = node + 1
		for i = 1,n_deps do
			if dep[i] < id then
				tree[DIR_L].n_children[node] = tree[DIR_L].n_children[node] + 1
				tree[DIR_L].children[{i,node}] = cur_node
			else
				tree[DIR_R].n_children[node] = tree[DIR_R].n_children[node] + 1
				tree[DIR_R].children[{i-tree[DIR_L].n_children[node],node}] = cur_node
			end
			tree.parent[cur_node] = node

			if dep[i] < id then tree.dir[cur_node] = DIR_L
			else				tree.dir[cur_node] = DIR_R
			end

			local d = math.abs(id - dep[i])
			if 		d == 1 then tree.dist[cur_node] = DIST_1
			elseif	d == 2 then tree.dist[cur_node] = DIST_2
			elseif	d <= 6 then tree.dist[cur_node] = DIST_3_6
			else				tree.dist[cur_node] = DIST_7_INF
			end

			tree,cur_node = self:to_torch_matrix_tree(dep[i], cur_node, tree)
		end
		node = cur_node
	end

	return tree, node
end

--[[ test
require 'dict'

torch.setnumthreads(1)

local voca_dic = Dict:new(collobert_template)
voca_dic:load('../data/wsj-dep/universal/dic/collobert/words.lst')
 
local pos_dic = Dict:new(cfg_template)
pos_dic:load("../data/wsj-dep/universal/dic/cpos.lst")

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
			if type(v) == 'table' then 
				for k1,v1 in pairs(v) do
					print(k1); print(v1)
				end
			end
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
