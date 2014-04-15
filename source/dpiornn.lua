require 'depstruct'
require 'utils'
require 'dict'

NPROCESS = 1
N_CONTEXT = 4

require 'xlua'
p = xlua.Profiler()

--**************** rerursive neural network class ******************--
IORNN = {}
IORNN_mt = {__index = IORNN}

--****************** functions ******************--
-- generate a n x m matrix by uniform distibuition within range [min,max]
function uniform(n, m, min, max)
	local M = torch.rand(n, m)
	M:mul(max-min):add(min)
	return M
end

-- logistic function
function logistic(X)
	return torch.cdiv(torch.ones(X.size), (-X):exp():add(1))
end

-- derivative of logistic function
-- 	logiX : logisitic(X)
function logisticPrime(logiX)
	return torch.cmul(-logiX + 1, logiX)
end

-- tanh function 
function tanh( X )
	return torch.tanh(X)
end

function tanhPrime(tanhX)
	return -torch.pow(tanhX,2)+1
end

-- identity function
function identity(X) 
	return X:clone()
end

function identityPrime(X)
	return torch.ones(X.size)
end

--************************* construction ********************--
-- create a new recursive autor encoder with a given structure
function IORNN:new(input)
	local net = {	dim = input.dim, wdim = input.lookup:size(1), 
					voca_dic = input.voca_dic, pos_dic = input.pos_dic, deprel_dic = input.deprel_dic}
	net.func = input.func or tanh
	net.funcPrime = input.funcPrime or tanhPrime
	setmetatable(net, IORNN_mt)

	net:init_params(input)
	return net
end

function IORNN:init_params(input)
	local net	 = self
	local dim	 = net.dim
	local wdim	 = net.wdim
	local voca_dic	 = net.voca_dic
	local deprel_dic = net.deprel_dic
	local pos_dic	 = net.pos_dic

	-- create params
	local n_params = dim*wdim + 4*dim + deprel_dic.size*dim*dim*2 + dim*dim + 2*dim + 
					(2*deprel_dic.size+1)*dim*4*N_CONTEXT + (2*deprel_dic.size+1) + 
					pos_dic.size*dim + N_CAP_FEAT*dim + voca_dic.size*wdim
	net.params = torch.zeros(n_params)

	--%%%%%%%%%%%%% assign ref %%%%%%%%%%%%%
	local r = 0.1
	local index = 1

	-- project word embs on to a higher-dim vector space
	net.Wh = net.params[{{index,index+dim*wdim-1}}]:resize(dim,wdim):copy(uniform(dim, wdim, -1e-3, 1e-3))
	index = index + dim*wdim
	net.bh = net.params[{{index,index+dim-1}}]:resize(dim,1)
	index = index + dim
	
	-- anonymous outer/inner
	net.root_inner = net.params[{{index,index+dim-1}}]:resize(dim,1):copy(uniform(dim, 1, -1e-3, 1e-3)) 
	index = index + dim
	net.anon_outer = net.params[{{index,index+dim-1}}]:resize(dim,1):copy(uniform(dim, 1, -1e-3, 1e-3)) 
	index = index + dim
	net.anon_inner = net.params[{{index,index+dim-1}}]:resize(dim,1):copy(uniform(dim, 1, -1e-3, 1e-3)) 
	index = index + dim

	-- composition weight matrices
	net.Wi = {}
	net.Wo = {}
	for i = 1,deprel_dic.size do
		net.Wi[i] = net.params[{{index,index+dim*dim-1}}]:resize(dim,dim):copy(uniform(dim, dim, -r, r))
		index = index + dim*dim
		net.Wo[i] = net.params[{{index,index+dim*dim-1}}]:resize(dim,dim):copy(uniform(dim, dim, -r, r))
		index = index + dim*dim
	end
	net.Wihead = net.params[{{index,index+dim*dim-1}}]:resize(dim,dim):copy(uniform(dim, dim, -r, r))
	index = index + dim*dim
	net.bi = net.params[{{index,index+dim-1}}]:resize(dim,1)
	index = index + dim
	net.bohead = net.params[{{index,index+dim-1}}]:resize(dim,1)
	index = index + dim

	-- for classification tanh(Wco * o + Wci * i + Wch * h + bc)
	net.Wci_stack = {}
	net.Wco_stack = {}
	net.Wci_buffer = {}
	net.Wco_buffer = {} 
	for i = 1,N_CONTEXT do
		net.Wci_stack[i] = net.params[{{index,index+(2*deprel_dic.size+1)*dim-1}}]	:resize((2*deprel_dic.size+1),dim)
																					:copy(uniform((2*deprel_dic.size+1),dim,-r,r))
		index = index + (2*deprel_dic.size+1)*dim
		net.Wco_stack[i] = net.params[{{index,index+(2*deprel_dic.size+1)*dim-1}}]	:resize((2*deprel_dic.size+1),dim)
																					:copy(uniform((2*deprel_dic.size+1),dim,-r,r))
		index = index + (2*deprel_dic.size+1)*dim
		net.Wci_buffer[i] = net.params[{{index,index+(2*deprel_dic.size+1)*dim-1}}]	:resize((2*deprel_dic.size+1),dim)
																					:copy(uniform((2*deprel_dic.size+1),dim,-r,r))
		index = index + (2*deprel_dic.size+1)*dim
		net.Wco_buffer[i] = net.params[{{index,index+(2*deprel_dic.size+1)*dim-1}}]	:resize((2*deprel_dic.size+1),dim)
																					:copy(uniform((2*deprel_dic.size+1),dim,-r,r))
		index = index + (2*deprel_dic.size+1)*dim
	end
	net.bc = net.params[{{index,index+(2*deprel_dic.size+1)-1}}]:resize((2*deprel_dic.size+1),1)
	index = index + (2*deprel_dic.size+1)

	-- POS tag 
	net.Lpos = net.params[{{index,index+pos_dic.size*dim-1}}]:resize(dim,pos_dic.size):copy(uniform(dim,pos_dic.size,-r,r))
	index = index + pos_dic.size * dim	

	-- capital letter feature
	net.Lcap = net.params[{{index,index+N_CAP_FEAT*dim-1}}]:resize(dim,N_CAP_FEAT):copy(uniform(dim,N_CAP_FEAT,-r,r))
	index = index + N_CAP_FEAT * dim

	--  word embeddings (always always always at the end of the array of params)
	net.L = net.params[{{index,index+voca_dic.size*wdim-1}}]:resize(wdim,voca_dic.size):copy(input.lookup)		-- word embeddings 
	index = index + voca_dic.size*wdim
end

function IORNN:create_grad()
	local grad = {}
	local dim = self.dim
	local wdim = self.wdim
	local voca_dic = self.voca_dic
	local deprel_dic = self.deprel_dic
	local pos_dic = self.pos_dic

	grad.params = torch.zeros(self.params:nElement())

	--%%%%%%%%%%%%% assign ref %%%%%%%%%%%%%
	local index = 1

	-- for projecting wembs onto a higher-dim space
	grad.Wh = grad.params[{{index,index+dim*wdim-1}}]:resize(dim,wdim)
	index = index + dim*wdim
	grad.bh = grad.params[{{index,index+dim-1}}]:resize(dim,1)
	index = index + dim

	-- anonymous outer
	grad.root_inner = grad.params[{{index,index+dim-1}}]:resize(dim,1)
	index = index + dim
	grad.anon_outer = grad.params[{{index,index+dim-1}}]:resize(dim,1)
	index = index + dim
	grad.anon_inner = grad.params[{{index,index+dim-1}}]:resize(dim,1)
	index = index + dim

	-- weights
	grad.Wi = {}
	grad.Wo = {}
	for i = 1,deprel_dic.size do
		grad.Wi[i] = grad.params[{{index,index+dim*dim-1}}]:resize(dim,dim)
		index = index + dim*dim
		grad.Wo[i] = grad.params[{{index,index+dim*dim-1}}]:resize(dim,dim)
		index = index + dim*dim
	end
	grad.Wihead = grad.params[{{index,index+dim*dim-1}}]:resize(dim,dim)
	index = index + dim*dim
	grad.bi = grad.params[{{index,index+dim-1}}]:resize(dim,1)
	index = index + dim
	grad.bohead = grad.params[{{index,index+dim-1}}]:resize(dim,1)
	index = index + dim

	-- for classification tanh(Wco * o + Wci * i + Wch * h + bc)
	grad.Wci_stack = {}
	grad.Wco_stack = {}
	grad.Wci_buffer = {}
	grad.Wco_buffer = {}
	for i = 1,N_CONTEXT do
		grad.Wci_stack[i] = grad.params[{{index,index+(2*deprel_dic.size+1)*dim-1}}]:resize((2*deprel_dic.size+1),dim)
		index = index + (2*deprel_dic.size+1)*dim
		grad.Wco_stack[i] = grad.params[{{index,index+(2*deprel_dic.size+1)*dim-1}}]:resize((2*deprel_dic.size+1),dim)
		index = index + (2*deprel_dic.size+1)*dim
		grad.Wci_buffer[i] = grad.params[{{index,index+(2*deprel_dic.size+1)*dim-1}}]:resize((2*deprel_dic.size+1),dim)
		index = index + (2*deprel_dic.size+1)*dim
		grad.Wco_buffer[i] = grad.params[{{index,index+(2*deprel_dic.size+1)*dim-1}}]:resize((2*deprel_dic.size+1),dim)
		index = index + (2*deprel_dic.size+1)*dim
	end
	grad.bc = grad.params[{{index,index+(2*deprel_dic.size+1)-1}}]:resize((2*deprel_dic.size+1),1)
	index = index + (2*deprel_dic.size+1)

	-- POS tag 
	grad.Lpos = grad.params[{{index,index+pos_dic.size*dim-1}}]:resize(dim,pos_dic.size)
	index = index + pos_dic.size * dim	

	-- capital letter feature
	grad.Lcap = grad.params[{{index,index+N_CAP_FEAT*dim-1}}]:resize(dim,N_CAP_FEAT)
	index = index + N_CAP_FEAT * dim

	--  word embeddings (always always always at the end of params)
	grad.L = grad.params[{{index,index+voca_dic.size*wdim-1}}]:resize(wdim,voca_dic.size)
	index = index + voca_dic.size*wdim

	return grad
end

-- save net into a file
function IORNN:save( filename , binary )
	local file = torch.DiskFile(filename, 'w')
	if binary == nil or binary then file:binary() end
	file:writeObject(self)
	file:close()
end

-- create net from file
function IORNN:load( filename , binary, func, funcPrime )
	local file = torch.DiskFile(filename, 'r')
	if binary == nil or binary then file:binary() end
	local net = file:readObject()
	file:close()

	setmetatable(net, IORNN_mt)
	setmetatable(net.voca_dic, Dict_mt)
	setmetatable(net.pos_dic, Dict_mt)
	setmetatable(net.deprel_dic, Dict_mt)

	return net
end


--************************ forward **********************--
function IORNN:forward_inside(tree)
	if tree.inner == nil then
		tree.inner = torch.zeros(self.dim, tree.n_nodes)
	else
		tree.inner:fill(0)
	end

	for i = tree.n_nodes,1,-1 do 
		local col_i = {{},{i}}

		-- for leaves
		if tree.n_children[i] == 0 then
			if i == 2 then -- ROOT 
				tree.inner[col_i]:copy(self.root_inner)
			else
				local input = (self.Wh * self.L[{{},{tree.word_id[i]}}])
								:add(self.Lpos[{{},{tree.pos_id[i]}}])
								:add(self.Lcap[{{},{tree.cap_id[i]}}])
								:add(self.bh)
				tree.inner[col_i]:copy(self.func(input))
			end

		-- for internal nodes
		else
			-- the left most child is always the head
			local input = torch.zeros(self.dim,1)
			for j = 2,tree.n_children[i] do
				local child = tree.children_id[{j,i}]
				input:addmm(self.Wi[tree.deprel_id[child]], tree.inner[{{},{child}}])
			end
			if tree.n_children[i] > 1 then
				input:div(tree.n_children[i]-1)
			end
			input:addmm(self.Wihead, tree.inner[{{},{tree.children_id[{1,i}]}}])
			input:add(self.bi)
			tree.inner[col_i]:copy(self.func(input))
		end
	end
end

function IORNN:update_with_state(tree, state)
	local sword = state.stack[state.stack_pos]
	local bword = state.buffer[state.buffer_pos]
	local action = state.action

	-- left-arc
	if action <= self.deprel_dic.size then
		local ret = tree.ds[bword]
		ret.outer = torch.zeros(self.dim, 1)

		local depnode = tree.wnode_id[sword]
		if tree.deprel_id[depnode] == 0 then -- this is a head, pick its parent
			depnode = tree.parent_id[depnode]
		end
		ret.depnode_id[#ret.depnode_id+1] = depnode
		ret.deprel_id[#ret.deprel_id+1] = action

		for i,deprel in ipairs(ret.deprel_id) do
			ret.outer:addmm(self.Wo[deprel], tree.inner[{{},{ret.depnode_id[i]}}])
		end
		ret.outer:div(#ret.depnode_id)
		ret.outer:add(self.bohead)
		ret.outer = self.func(ret.outer)

	-- right-arc
	elseif action <= 2*self.deprel_dic.size then
		local ret = tree.ds[sword]
		ret.outer = torch.zeros(self.dim, 1)

		local depnode = tree.wnode_id[bword]
		if tree.deprel_id[depnode] == 0 then -- this is a head, pick its parent
			depnode = tree.parent_id[depnode]
		end
		ret.depnode_id[#ret.depnode_id+1] = depnode
		ret.deprel_id[#ret.deprel_id+1] = action - self.deprel_dic.size

		for i,deprel in ipairs(ret.deprel_id) do
			ret.outer:addmm(self.Wo[deprel], tree.inner[{{},{ret.depnode_id[i]}}])
		end
		ret.outer:div(#ret.depnode_id)
		ret.outer:add(self.bohead)
		ret.outer = self.func(ret.outer)

	-- shift
	else
		-- do nothing
	end
end

function IORNN:with_state(tree, state, mem, pos)
	local word = nil
	if mem == state.stack and pos >= 1 then
		word = state.stack[pos]
	elseif mem == state.buffer and pos <= state.n_words then 
		word = state.buffer[pos]
	end

	local ret = nil

	if word == nil then -- out-of-stack/buffer
		ret = {}
		ret.depnode_id = {}
		ret.deprel_id = {}
		ret.inner = self.anon_inner
		ret.outer = self.anon_outer
	else
		ret = tree.ds[word]
	end

	return ret
end

function IORNN:forward_outside(tree, state)
	tree.classify = {}
	tree.score = torch.zeros(2*self.deprel_dic.size+1,1)
	for i = 1,N_CONTEXT do
		tree.classify[i] = { 	stack = self:with_state(tree, state, state.stack, state.stack_pos-i+1),
								buffer = self:with_state(tree, state, state.buffer, state.buffer_pos+i-1) }
		tree.score	:addmm(self.Wci_stack[i], tree.classify[i].stack.inner)
					:addmm(self.Wco_stack[i], tree.classify[i].stack.outer)
					:addmm(self.Wci_buffer[i], tree.classify[i].buffer.inner)
					:addmm(self.Wco_buffer[i], tree.classify[i].buffer.outer)
	end
	tree.score:add(self.bc)
	tree.prob = safe_compute_softmax(tree.score)
	tree.action = state.action
	tree.cost = -math.log(tree.prob[{tree.action,1}])

	return tree.cost
end

--*********************** backpropagate *********************--
function IORNN:backpropagate_outside(tree, grad)
	local gZc = tree.prob:clone(); gZc[{tree.action,1}] = gZc[{tree.action,1}] - 1
	grad.bc:add(gZc)

	for i = 1,N_CONTEXT do
		for v,typ in pairs(tree.classify[i]) do
			grad['Wco_'..v][i]:addmm(gZc, typ.outer:t())
			grad['Wci_'..v][i]:addmm(gZc, typ.inner:t())

			-- for inner
			if typ.inner == self.anon_inner then 
				grad.anon_inner:addmm(self['Wci_'..v][i]:t(), gZc)
			elseif typ.inner == self.root_inner then --ROOT
				grad.root_inner:addmm(self['Wci_'..v][i]:t(), gZc)
			else
				tree.gradi[{{},{typ.node_id}}]:addmm(self['Wci_'..v][i]:t(), gZc)
			end
	
			-- for outer
			if typ.outer == self.anon_outer then -- 'free' word or out-of-stack/buffer
				grad.anon_outer:addmm(self['Wco_'..v][i]:t(), gZc)
			else
				local gZo = (self['Wco_'..v][i]:t() * gZc):cmul(self.funcPrime(typ.outer))
				grad.bohead:add(gZo)
				local n = #typ.depnode_id
				for i,depnode_id in ipairs(typ.depnode_id) do
					tree.gradi[{{},{depnode_id}}]:addmm(1/n, self.Wo[typ.deprel_id[i]]:t(), gZo)
					grad.Wo[typ.deprel_id[i]]:addmm(1/n, gZo, tree.inner[{{},{depnode_id}}]:t())
				end
			end
		end
	end
end

function IORNN:backpropagate_inside(tree, grad)
	if tree.gradZi == nil then
		tree.gradZi = torch.zeros(self.dim, tree.n_nodes)
	else
		tree.gradZi:fill(0)
	end

	for i = 1,tree.n_nodes do
		local col_i = {{},{i}}
		local gZi = tree.gradi[col_i]
		gZi:cmul(self.funcPrime(tree.inner[col_i]))

		-- for internal node
		if tree.n_children[i] > 0 then
			tree.gradZi[col_i]:add(gZi)

			-- weight matrix for inner
			local n = tree.n_children[i] - 1
			for j = 1,tree.n_children[i] do
				local child_id = tree.children_id[{j,i}]
				local deprel_id = tree.deprel_id[child_id]
				if deprel_id == 0 then -- this is head
					grad.Wihead:addmm(gZi, tree.inner[{{},{child_id}}]:t())
					tree.gradi[{{},{child_id}}]:addmm(self.Wihead:t(), gZi)
				else
					grad.Wi[deprel_id]:addmm(1/n, gZi, tree.inner[{{},{child_id}}]:t())
					tree.gradi[{{},{child_id}}]:addmm(1/n, self.Wi[deprel_id]:t(), gZi)
				end
			end
			grad.bi:add(gZi)

		else -- leaf
			tree.gradZi[col_i]:add(gZi)

			if self.update_L then
				if i > 2 then -- not ROOT
					grad.Wh:addmm(gZi, self.L[{{},{tree.word_id[i]}}]:t())
					grad.bh:add(gZi)
					grad.L[{{},{tree.word_id[i]}}]:addmm(self.Wh:t(), gZi)
					grad.Lpos[{{},{tree.pos_id[i]}}]:add(gZi)
					grad.Lcap[{{},{tree.cap_id[i]}}]:add(gZi)
				end
			end
		end
	end
end

function IORNN:create_treelets(sent)
	local treelets = {}
	for i = 1,sent.n_words do
		local tree = Depstruct:create_empty_tree(1, sent.n_words)
		if i > 1 then
			tree.word_id[1] = sent.word_id[i]
			tree.pos_id[1] = sent.pos_id[i]
			tree.cap_id[1] = sent.cap_id[i]
			tree.wnode_id[i] = 1

			local input = (self.Wh * self.L[{{},{tree.word_id[1]}}])
							:add(self.Lpos[{{},{tree.pos_id[1]}}])
							:add(self.Lcap[{{},{tree.cap_id[1]}}])
							:add(self.bh)
			tree.inner = self.func(input)
			tree.head_inner = tree.inner
		else -- ROOT
			tree.inner = self.root_inner
			tree.head_inner = self.root_inner
		end
		tree.head_outer = self.anon_outer
		treelets[i] = tree
	end	

	return treelets
end

-- note: htree and dtree are not changed during the merging
function IORNN:merge_treelets(htree, dtree, deprel)
	local n_nodes = htree.n_nodes + dtree.n_nodes
	local n_words = htree.wnode_id:nElement()
	local next_id = htree.n_nodes + 1

	if htree.n_nodes == 1 then
		n_nodes = n_nodes + 1
		next_id = next_id + 1
	end

	tree = Depstruct:create_empty_tree(n_nodes, n_words)
	tree.inner = torch.zeros(self.dim, n_nodes)

	-- copy htree into tree
	if htree.n_nodes > 1 then
		tree.word_id[{{1,htree.n_nodes}}]:copy(htree.word_id)
		tree.pos_id[{{1,htree.n_nodes}}]:copy(htree.pos_id)
		tree.cap_id[{{1,htree.n_nodes}}]:copy(htree.cap_id)

		tree.parent_id[{{1,htree.n_nodes}}]:copy(htree.parent_id)
		tree.n_children[{{1,htree.n_nodes}}]:copy(htree.n_children)
		tree.children_id[{{},{1,htree.n_nodes}}]:copy(htree.children_id)
		tree.wnode_id:add(htree.wnode_id)
		tree.deprel_id[{{1,htree.n_nodes}}]:copy(htree.deprel_id)
		tree.inner[{{},{1,htree.n_nodes}}]:copy(htree.inner)
	elseif htree.n_nodes == 1 then
		tree.word_id[2] = htree.word_id[1]
		tree.cap_id[2] = htree.cap_id[1]
		tree.pos_id[2] = htree.pos_id[1]

		tree.parent_id[2] = 1
		tree.n_children[1] = 1
		tree.children_id[{1,1}] = 2
		local temp = htree.wnode_id + 1; temp[temp:eq(1)] = 0
		tree.wnode_id:add(temp)
		tree.inner[{{},{2}}]:copy(htree.inner)
	end

	-- copy dtree into tree
	tree.word_id[{{next_id,-1}}]:copy(dtree.word_id)
	tree.pos_id[{{next_id,-1}}]:copy(dtree.pos_id)
	tree.cap_id[{{next_id,-1}}]:copy(dtree.cap_id)

	tree.parent_id[{{next_id,-1}}]:copy(dtree.parent_id):add(next_id-1)
	tree.parent_id[next_id] = 1

	tree.n_children[1] = tree.n_children[1] + 1
	tree.n_children[{{next_id,-1}}]:copy(dtree.n_children)

	tree.children_id[{tree.n_children[1],1}] = next_id
	local temp = dtree.children_id + (next_id-1); temp[temp:eq(next_id-1)] = 0
	tree.children_id[{{},{next_id,-1}}]:copy(temp)

	local temp = dtree.wnode_id + (next_id-1); temp[temp:eq(next_id-1)] = 0
	tree.wnode_id:add(temp)

	tree.deprel_id[{{next_id,-1}}]:copy(dtree.deprel_id)
	tree.deprel_id[next_id] = deprel

	tree.inner[{{},{next_id,-1}}]:copy(dtree.inner)

	-- dtree is 'complete', compute its inner representation
	if next_id < n_nodes then
		local input = torch.zeros(self.dim,1)
		for j = 2,tree.n_children[next_id] do
			local child = tree.children_id[{j,next_id}]
			input:add(self.Wi[tree.deprel_id[child]] * tree.inner[{{},{child}}])
		end
		input:div(tree.n_children[next_id] - 1)
		input:addmm(self.Wihead, tree.inner[{{},{tree.children_id[{1,next_id}]}}]):add(self.bi)
		tree.inner[{{},{next_id}}]:copy(self.func(input))
	end

	-- only if htree does not contain ROOT compute outer representation at the head word
	local input = torch.zeros(self.dim,1)
	for j = 2,tree.n_children[1] do
		local child = tree.children_id[{j,1}]
		input:addmm(self.Wo[tree.deprel_id[child]], tree.inner[{{},{child}}])
	end
	input:div(tree.n_children[1] - 1)
	input:add(self.bohead)
	tree.head_outer = self.func(input)
	tree.head_inner = tree.inner[{{},{2}}]

--[[
	print('##########################################################')
	for _,t in ipairs({htree,dtree,tree}) do
		print('------------------------------')
		for v,k in pairs(t) do
			print(v)
			print(k)
		end
	end
]]

	return tree
end

function IORNN:predict_action(states)
	local nstates = #states
	local scores = torch.zeros(2*self.deprel_dic.size+1, nstates)

	for j = 1,N_CONTEXT do
		local stack_inner = torch.repeatTensor(self.anon_inner, 1, nstates)
		local stack_outer = torch.repeatTensor(self.anon_outer, 1, nstates)
		local buffer_inner = torch.repeatTensor(self.anon_inner, 1, nstates)
		local buffer_outer = torch.repeatTensor(self.anon_outer, 1, nstates)

		for i,state in ipairs(states) do
			local spos = state.stack_pos - j + 1
			local bpos = state.buffer_pos + j - 1
			local index = {{},{i}}
			if spos >= 1 then
				local stack_tree = state.treelets[state.stack[spos]]
				stack_inner[index]:copy(stack_tree.head_inner)
				stack_outer[index]:copy(stack_tree.head_outer)
			end
			if bpos <= state.n_words then
				local buffer_tree = state.treelets[state.buffer[bpos]]
				buffer_inner[index]:copy(buffer_tree.head_inner)
				buffer_outer[index]:copy(buffer_tree.head_outer)
			end
		end

		scores	:addmm(self.Wci_stack[j], stack_inner)
				:addmm(self.Wco_stack[j], stack_outer)
				:addmm(self.Wci_buffer[j], buffer_inner)
				:addmm(self.Wco_buffer[j], buffer_outer)
	end
	scores:add(torch.repeatTensor(self.bc, 1, nstates))
	return safe_compute_softmax(scores)
end

function IORNN:create_tree_for_training(ds) 
	local tree = ds:to_torch_matrix_tree()
	tree.gradi = torch.zeros(self.dim, tree.n_nodes)

	tree.ds = {}
	tree.ds[1] = { 	inner = self.root_inner, outer = self.anon_outer, 
					depnode_id = {}, deprel_id = {}, node_id = tree.wnode_id[1] }
	for i = 2, ds.n_words do 
		local wnode = tree.wnode_id[i]
		local input = (self.Wh * self.L[{{},{tree.word_id[wnode]}}])
							:add(self.Lpos[{{},{tree.pos_id[wnode]}}])
							:add(self.Lcap[{{},{tree.cap_id[wnode]}}])
							:add(self.bh)

		tree.ds[i] = { 	inner = self.func(input),
						outer = self.anon_outer, 
						depnode_id = {}, deprel_id = {},
						node_id = tree.wnode_id[i] }
	end	

	return tree 
end

function IORNN:computeCostAndGrad(treebank, config, grad, parser)
	local parse = config.parse or false

	p:start('compute cost and grad')	

if NPROCESS > 1 then
else

	grad.params:fill(0)  -- always make sure that this grad is intialized with 0

	local cost = 0
	local nSample = 0
	local tword_id = {}

	p:start('process treebank')
	for i, ds in ipairs(treebank) do
		p:start('process tree')
		local states = parser:extract_training_states(ds)
		local tree = self:create_tree_for_training(ds)
	
		p:start('forward inside')
		self:forward_inside(tree)
		p:lap('forward inside')

		p:start('outside')
		for _,state in ipairs(states) do
			if state.stack_pos > 0 then 
				p:start('forward outside')
				local lcost = self:forward_outside(tree, state)
				p:lap('forward outside')
				cost = cost + lcost
				p:start('backward outside')
				self:backpropagate_outside(tree, grad)
				p:lap('backward outside')
				p:start('update state')
				self:update_with_state(tree, state)
				p:lap('update state')
			end
		end
		p:lap('outside')

		nSample = nSample + #states
		p:start('backward inside')
		self:backpropagate_inside(tree, grad)
		p:lap('backward inside')

		for i=2,tree.wnode_id:nElement() do -- do not take the root into account
			tword_id[tree.word_id[tree.wnode_id[i]]] = 1
		end
		p:lap('process tree')
	end
	p:lap('process treebank') 

	p:start('compute grad')
	local wparams = self.params[{{1,-1-self.wdim*self.voca_dic.size}}]
	local grad_wparams = grad.params[{{1,-1-self.wdim*self.voca_dic.size}}]
	cost = cost / nSample + config.lambda/2 * torch.pow(wparams,2):sum()
	grad_wparams:div(nSample):add(wparams * config.lambda)
	
	for wid,_ in pairs(tword_id) do
		cost = cost + torch.pow(self.L[{{},{wid}}],2):sum() * config.lambda_L/2
		grad.L[{{},{wid}}]:div(nSample):add(config.lambda_L, self.L[{{},{wid}}])
	end 
	p:lap('compute grad')

	p:lap('compute cost and grad') 
	--p:printAll()

	return cost, grad, treebank, tword_id
end

end

-- check gradient
function IORNN:checkGradient(treebank, parser, config)
	local epsilon = 1e-4

	local dim = self.dim
	local wrdDicLen = self.wrdDicLen
	local nCat = self.nCat

	local Theta = self.params
	local grad = self:create_grad()
	local _, gradTheta = self:computeCostAndGrad(treebank, config, grad, parser)
	gradTheta = gradTheta.params
	
	local n = Theta:nElement()
	print(n)
	local numGradTheta = torch.zeros(n)
	for i = 1,n do
		local index = {{i}}
		Theta[index]:add(epsilon)
		local grad = self:create_grad()
		local costPlus,_ = self:computeCostAndGrad(treebank, config, grad, parser)
		
		Theta[index]:add(-2*epsilon)
		local costMinus,_ = self:computeCostAndGrad(treebank, config, grad, parser)
		Theta[index]:add(epsilon)

		numGradTheta[i] = (costPlus - costMinus) / (2*epsilon) 

		local diff = math.abs(numGradTheta[i] - gradTheta[i])
		print('diff ' .. i .. ' ' .. diff)
	end

	local diff = torch.norm(gradTheta - numGradTheta) 
					/ torch.norm(gradTheta + numGradTheta)
	print(diff)
	print("should be < 1e-9")
end

--**************************** training ************************--
--
-- adapted from optim.adagrad
function IORNN:adagrad(func, config, state)
	-- (0) get/update state
	if config == nil and state == nil then
		print('no state table, ADAGRAD initializing')
	end
	local config = config or {}
	local state = state or config

	local weight_lr = config.weight_learningRate or 1e-1
	local voca_dic_lr = config.voca_dic_learningRate or 1e-3

	local lrd = config.learningRateDecay or 0
	state.evalCounter = state.evalCounter or 0
	local nevals = state.evalCounter

	-- (1) evaluate f(x) and df/dx
	local cost, grad, _, tword_id = func()

	-- (3) learning rate decay (annealing)
	local weight_clr	= weight_lr / (1 + nevals*lrd)
	local voca_dic_clr		= voca_dic_lr / (1 + nevals*lrd)

	-- (4) parameter update with single or individual learning rates
	if not state.paramVariance then
		state.paramVariance = self:create_grad()
		state.paramStd = self:create_grad()
	end

	-- for weights
	local wparamindex = {{1,-1-self.wdim*self.voca_dic.size}}
	state.paramVariance.params[wparamindex]:addcmul(1,grad.params[wparamindex],grad.params[wparamindex])
	torch.sqrt(state.paramStd.params[wparamindex],state.paramVariance.params[wparamindex])
	self.params[wparamindex]:addcdiv(-weight_clr, grad.params[wparamindex],state.paramStd.params[wparamindex]:add(1e-10))

	-- for word embeddings
	for wid,_ in pairs(tword_id) do
		local col_i = {{},{wid}}
		state.paramVariance.L[col_i]:addcmul(1,grad.L[col_i],grad.L[col_i])
		torch.sqrt(state.paramStd.L[col_i],state.paramVariance.L[col_i])
		self.L[col_i]:addcdiv(-voca_dic_clr, grad.L[col_i],state.paramStd.L[col_i]:add(1e-10))
	end

	-- (5) update evaluation counter
	state.evalCounter = state.evalCounter + 1
end

function IORNN:train_with_adagrad(traintreebank, batchSize, 
									maxepoch, lambda, prefix,
									adagrad_config, adagrad_state, 
									parser, devtreebank_path)
	local nSample = #traintreebank
	local grad = self:create_grad()
	
	local epoch = 0
	local j = 0
	os.execute('th eval_depparser.lua '..prefix..'_'..epoch..' '..devtreebank_path)


	epoch = epoch + 1
	print('===== epoch ' .. epoch .. '=====')

	while true do
		j = j + 1
		if j > nSample/batchSize then 
			self:save(prefix .. '_' .. epoch)
			os.execute('th eval_depparser.lua '..prefix..'_'..epoch..' '..devtreebank_path)

			j = 1 
			epoch = epoch + 1
			if epoch > maxepoch then break end
			print('===== epoch ' .. epoch .. '=====')
		end

		local subtreebank = {}
		for k = 1,batchSize do
			subtreebank[k] = traintreebank[k+(j-1)*batchSize]
		end
	
		local function func()
			cost, grad, subtreebank, tword_id  = self:computeCostAndGrad(subtreebank, 
							{lambda = lambda.lambda, lambda_L=lambda.lambda_L}, grad, parser)

			print('iter ' .. j .. ': ' .. cost) io.flush()		
			return cost, grad, subtreebank, tword_id
		end

		p:start("optim")
		self:adagrad(func, adagrad_config, adagrad_state)
		
		p:lap("optim")
		p:printAll()

		collectgarbage()
	end

	return adagrad_config, adagrad_state
end


--[[********************************** test ******************************--
	require 'depparser'
	torch.setnumthreads(1)

	local voca_dic = Dict:new()
	voca_dic:load('../data/wsj-dep/toy/dic/words.lst')
 	local pos_dic = Dict:new()
	pos_dic:load('../data/wsj-dep/toy/dic/pos.lst')
	local deprel_dic = Dict:new()
	deprel_dic:load('../data/wsj-dep/toy/dic/deprel.lst')
	local lookup = torch.rand(2, voca_dic.size)

	dim = 3

	print('training...')
	local parser = Depparser:new(lookup, voca_dic, pos_dic, deprel_dic, dim)
	local treebank,_ = parser:load_treebank('../data/wsj-dep/toy/data/train.conll')
	--treebank = {treebank[1]}
	--treebank[1].states = {treebank[1].states[1]}
	--print(treebank)

	net = parser.net

	config = {lambda = 1e-4, lambda_L = 1e-7}
	net.update_L = true
	net:checkGradient(treebank, parser, config)
]]
