require 'depstruct'
require 'utils'
require 'dict'

NPROCESS = 1

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
	local net = {dim = input.lookup:size(1), voca_dic = input.voca_dic, pos_dic = input.pos_dic, deprel_dic = input.deprel_dic}
	net.func = input.func
	net.funcPrime = input.funcPrime
	setmetatable(net, IORNN_mt)

	net:init_params(input)
	return net
end

function IORNN:init_params(input)
	local net	 = self
	local dim	 = net.dim
	local voca_dic	 = net.voca_dic
	local deprel_dic = net.deprel_dic

	-- create params
	local n_params = 2*dim + deprel_dic.size*dim*dim*2 + dim*dim + 2*dim + (2*deprel_dic.size+1)*dim*4 + (2*deprel_dic.size+1) + voca_dic.size*dim
	net.params = torch.zeros(n_params)

	--%%%%%%%%%%%%% assign ref %%%%%%%%%%%%%
	local r = 0.1
	local index = 1
	
	-- anonymous outer/inner
	net.root_inner = net.params[{{index,index+dim-1}}]:resize(dim,1):copy(uniform(dim, 1, -1e-3, 1e-3)) 
	index = index + dim
	net.anon_outer = net.params[{{index,index+dim-1}}]:resize(dim,1):copy(uniform(dim, 1, -1e-3, 1e-3)) 
	index = index + dim

	-- weights
	net.Wi = {}
	net.Wo = {}
	for i = 1,deprel_dic.size do
		net.Wi[i] = net.params[{{index,index+dim*dim-1}}]:resize(dim,dim):copy(torch.eye(dim) + uniform(dim, dim, -1e-3, 1e-3))
		index = index + dim*dim
		net.Wo[i] = net.params[{{index,index+dim*dim-1}}]:resize(dim,dim):copy(torch.eye(dim) + uniform(dim, dim, -1e-3, 1e-3))
		index = index + dim*dim
	end
	net.Wihead = net.params[{{index,index+dim*dim-1}}]:resize(dim,dim):copy(uniform(dim, dim, -r, r))
	index = index + dim*dim
	net.bi = net.params[{{index,index+dim-1}}]:resize(dim,1)
	index = index + dim
	net.bohead = net.params[{{index,index+dim-1}}]:resize(dim,1)
	index = index + dim

	-- for classification tanh(Wco * o + Wci * i + Wch * h + bc)
	net.Wci_stack = net.params[{{index,index+(2*deprel_dic.size+1)*dim-1}}]:resize((2*deprel_dic.size+1),dim):copy(uniform((2*deprel_dic.size+1),dim,-r,r))
	index = index + (2*deprel_dic.size+1)*dim
	net.Wco_stack = net.params[{{index,index+(2*deprel_dic.size+1)*dim-1}}]:resize((2*deprel_dic.size+1),dim):copy(uniform((2*deprel_dic.size+1),dim,-r,r))
	index = index + (2*deprel_dic.size+1)*dim
	net.Wci_buffer = net.params[{{index,index+(2*deprel_dic.size+1)*dim-1}}]:resize((2*deprel_dic.size+1),dim):copy(uniform((2*deprel_dic.size+1),dim,-r,r))
	index = index + (2*deprel_dic.size+1)*dim
	net.Wco_buffer = net.params[{{index,index+(2*deprel_dic.size+1)*dim-1}}]:resize((2*deprel_dic.size+1),dim):copy(uniform((2*deprel_dic.size+1),dim,-r,r))
	index = index + (2*deprel_dic.size+1)*dim
	net.bc = net.params[{{index,index+(2*deprel_dic.size+1)-1}}]:resize((2*deprel_dic.size+1),1)
	index = index + (2*deprel_dic.size+1)

	--  word embeddings (always always always at the end of the array of params)
	net.L = net.params[{{index,index+voca_dic.size*dim-1}}]:resize(dim,voca_dic.size):copy(input.lookup)		-- word embeddings 
	index = index + voca_dic.size*dim
end

function IORNN:create_grad()
	local grad = {}
	local dim = self.dim
	local voca_dic = self.voca_dic
	local deprel_dic = self.deprel_dic

	grad.params = torch.zeros(self.params:nElement())

	--%%%%%%%%%%%%% assign ref %%%%%%%%%%%%%
	local index = 1
	
	-- anonymous outer
	grad.root_inner = grad.params[{{index,index+dim-1}}]:resize(dim,1)
	index = index + dim
	grad.anon_outer = grad.params[{{index,index+dim-1}}]:resize(dim,1)
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
	grad.Wci_stack = grad.params[{{index,index+(2*deprel_dic.size+1)*dim-1}}]:resize((2*deprel_dic.size+1),dim)
	index = index + (2*deprel_dic.size+1)*dim
	grad.Wco_stack = grad.params[{{index,index+(2*deprel_dic.size+1)*dim-1}}]:resize((2*deprel_dic.size+1),dim)
	index = index + (2*deprel_dic.size+1)*dim
	grad.Wci_buffer = grad.params[{{index,index+(2*deprel_dic.size+1)*dim-1}}]:resize((2*deprel_dic.size+1),dim)
	index = index + (2*deprel_dic.size+1)*dim
	grad.Wco_buffer = grad.params[{{index,index+(2*deprel_dic.size+1)*dim-1}}]:resize((2*deprel_dic.size+1),dim)
	index = index + (2*deprel_dic.size+1)*dim
	grad.bc = grad.params[{{index,index+(2*deprel_dic.size+1)-1}}]:resize((2*deprel_dic.size+1),1)
	index = index + (2*deprel_dic.size+1)

	--  word embeddings (always always always at the end of params)
	grad.L = grad.params[{{index,index+voca_dic.size*dim-1}}]:resize(dim,voca_dic.size)
	index = index + voca_dic.size*dim

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
				tree.inner[col_i]:copy(self.L[{{},{tree.word_id[i]}}])
			end

		-- for internal nodes
		else
			-- the left most child is always the head
			local input = torch.zeros(self.dim,1)
			for j = 2,tree.n_children[i] do
				local child = tree.children_id[{j,i}]
				input:add(self.Wi[tree.deprel_id[child]] * tree.inner[{{},{child}}])
			end
			if tree.n_children[i] > 1 then
				input:div(tree.n_children[i]-1)
			end
			input:add(self.Wihead * tree.inner[{{},{tree.children_id[{1,i}]}}])
			input:add(self.bi)
			tree.inner[col_i]:copy(self.func(input))
		end
	end
end

function IORNN:with_state(tree, state, mem)
	local word = state.stack[state.stack_pos]
	if mem == state.buffer then
		word = state.buffer[state.buffer_pos]
	end

	local ret = {}
	ret.node_id = tree.wnode_id[word]

	if word == 1 then --ROOT
		ret.inner = self.root_inner
	else
 		ret.inner = tree.inner[{{},{ret.node_id}}]
	end

	ret.outer = torch.zeros(self.dim,1)

	ret.depnode_id = {}
	ret.deprel_id = {}
	for i = 1,state.head:nElement() do
		if state.head[i] == word then
			local depnode = tree.wnode_id[i]
			if tree.deprel_id[depnode] == 0 then -- this is a head, pick its parent
				depnode = tree.parent_id[depnode]
			end
			ret.outer:add(self.Wo[state.deprel[i]] * tree.inner[{{},{depnode}}])
			ret.depnode_id[#ret.depnode_id+1] = depnode
			ret.deprel_id[#ret.deprel_id+1] = state.deprel[i]
		end
	end

	if #ret.depnode_id == 0 then -- this word has no dependents
		ret.outer = self.anon_outer
	else
		ret.outer:div(#ret.depnode_id)
		ret.outer:add(self.bohead)
		ret.outer = self.func(ret.outer)
	end

	return ret
end

function IORNN:forward_outside(tree, state)
	tree.classify = { 	stack = self:with_state(tree, state, state.stack),
						buffer = self:with_state(tree, state, state.buffer) }

	tree.score = (self.Wci_stack * tree.classify.stack.inner)
				:add(self.Wco_stack * tree.classify.stack.outer)
				:add(self.Wci_buffer * tree.classify.buffer.inner)
				:add(self.Wco_buffer * tree.classify.buffer.outer)
				:add(self.bc)
	tree.prob = safe_compute_softmax(tree.score)
	tree.action = state.action
	tree.cost = -math.log(tree.prob[{tree.action,1}])

	return tree.cost
end

--*********************** backpropagate *********************--
function IORNN:backpropagate_outside(tree, grad)
	local gZc = tree.prob:clone(); gZc[{tree.action,1}] = gZc[{tree.action,1}] - 1
	grad.bc:add(gZc)

	for v,typ in pairs(tree.classify) do
		grad['Wco_'..v]:add(gZc * typ.outer:t())
		grad['Wci_'..v]:add(gZc * typ.inner:t())

		-- for inner
		if typ.inner == self.root_inner then --ROOT
			grad.root_inner:add(self['Wci_'..v]:t() * gZc)
		else
			tree.gradi[{{},{typ.node_id}}]:add(self['Wci_'..v]:t() * gZc)
		end

		-- for outer
		if typ.outer == self.anon_outer then -- 'free' word
			grad.anon_outer:add(self['Wco_'..v]:t() * gZc)
		else
			local gZo = (self['Wco_'..v]:t() * gZc):cmul(self.funcPrime(typ.outer))
			grad.bohead:add(gZo)
			local n = #typ.depnode_id
			for i,depnode_id in ipairs(typ.depnode_id) do
				tree.gradi[{{},{depnode_id}}]:add((self.Wo[typ.deprel_id[i]]:t() * gZo):div(n))
				grad.Wo[typ.deprel_id[i]]:add((gZo * tree.inner[{{},{depnode_id}}]:t()):div(n))
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

		-- for internal node
		if tree.n_children[i] > 0 then
			gZi:cmul(self.funcPrime(tree.inner[col_i]))
			tree.gradZi[col_i]:add(gZi)

			-- weight matrix for inner
			local n = tree.n_children[i] - 1
			for j = 1,tree.n_children[i] do
				local child_id = tree.children_id[{j,i}]
				local deprel_id = tree.deprel_id[child_id]
				if deprel_id == 0 then -- this is head
					grad.Wihead:add(gZi * tree.inner[{{},{child_id}}]:t())
					tree.gradi[{{},{child_id}}]:add(self.Wihead:t() * gZi)
				else
					grad.Wi[deprel_id]:add((gZi * tree.inner[{{},{child_id}}]:t()):div(n))
					tree.gradi[{{},{child_id}}]:add((self.Wi[deprel_id]:t() * gZi):div(n))
				end
			end
			grad.bi:add(gZi)

		else -- leaf
			tree.gradZi[col_i]:add(gZi)

			if self.update_L then
				if i > 2 then -- not ROOT
					grad.L[{{},{tree.word_id[i]}}]:add(gZi)
				end
			end
		end
	end
end

function IORNN:create_treelets(sent)
	local treelets = {}
	for i = 0,sent.n_words do
		local tree = Depstruct:create_empty_tree(1, sent.n_words)
		if i > 1 then
			tree.word_id[1] = sent.word_id[i]
			tree.wnode_id[i] = 1
			tree.inner = self.L[{{},{tree.word_id[1]}}]:clone()
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
		tree.parent_id[{{1,htree.n_nodes}}]:copy(htree.parent_id)
		tree.n_children[{{1,htree.n_nodes}}]:copy(htree.n_children)
		tree.children_id[{{},{1,htree.n_nodes}}]:copy(htree.children_id)
		tree.wnode_id:add(htree.wnode_id)
		tree.deprel_id[{{1,htree.n_nodes}}]:copy(htree.deprel_id)
		tree.inner[{{},{1,htree.n_nodes}}]:copy(htree.inner)
	elseif htree.n_nodes == 1 then
		tree.word_id[2] = htree.word_id[1]
		tree.parent_id[2] = 1
		tree.n_children[1] = 1
		tree.children_id[{1,1}] = 2
		local temp = htree.wnode_id + 1; temp[temp:eq(1)] = 0
		tree.wnode_id:add(temp)
		tree.inner[{{},{2}}]:copy(htree.inner)
	end

	-- copy dtree into tree
	tree.word_id[{{next_id,-1}}]:copy(dtree.word_id)
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
		input:add(self.Wihead * tree.inner[{{},{tree.children_id[{1,next_id}]}}]):add(self.bi)
		tree.inner[{{},{next_id}}]:copy(self.func(input))
	end

	-- only if htree does not contain ROOT compute outer representation at the head word
	local input = torch.zeros(self.dim,1)
	for j = 2,tree.n_children[1] do
		local child = tree.children_id[{j,1}]
		input:add(self.Wo[tree.deprel_id[child]] * tree.inner[{{},{child}}])
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

	local stack_inner = torch.ones(self.dim, nstates)
	local stack_outer = torch.ones(self.dim, nstates)
	local buffer_inner = torch.ones(self.dim, nstates)
	local buffer_outer = torch.ones(self.dim, nstates)

	for i,state in ipairs(states) do
		if state.stack_pos >= 1 and state.buffer_pos <= state.n_words then
			local stack_tree = state.treelets[state.stack[state.stack_pos]]
			local buffer_tree = state.treelets[state.buffer[state.buffer_pos]]
			local index = {{},{i}}
			stack_inner[index]:copy(stack_tree.head_inner)
			stack_outer[index]:copy(stack_tree.head_outer)
			buffer_inner[index]:copy(buffer_tree.head_inner)
			buffer_outer[index]:copy(buffer_tree.head_outer)
		end
	end
	
	scores = (self.Wci_stack * stack_inner)
				:add(self.Wco_stack * stack_outer)
				:add(self.Wci_buffer * buffer_inner)
				:add(self.Wco_buffer * buffer_outer)
				:add(torch.repeatTensor(self.bc, 1, nstates))
	return safe_compute_softmax(scores)
end

function IORNN:computeCostAndGrad(treebank, config, grad, parser)
	local parse = config.parse or false

	p:start('compute cost and grad')	

if NPROCESS > 1 then
else
	local grad = grad or self:create_grad()
	grad.params:fill(0)  -- always make sure that this grad is intialized with 0

	local cost = 0
	local nSample = 0
	local tword_id = {}

	p:start('process treebank')
	for i, ds in ipairs(treebank) do
		local states = parser:extract_training_states(ds)
		local tree = ds:to_torch_matrix_tree()
		self:forward_inside(tree)

		if tree.gradi == nil then 
			tree.gradi = torch.zeros(self.dim, tree.n_nodes)
		else
			tree.gradi:fill(0)
		end	

		for _,state in ipairs(states) do
			if state.stack_pos > 0 then 
				local lcost = self:forward_outside(tree, state)
				cost = cost + lcost
				self:backpropagate_outside(tree, grad)
			end
		end
		nSample = nSample + #states
		self:backpropagate_inside(tree, grad)

		for i=2,tree.wnode_id:nElement() do -- do not take the root into account
			tword_id[tree.word_id[tree.wnode_id[i]]] = 1
		end
	end
	p:lap('process treebank') 

	p:start('compute grad')
	local wparams = self.params[{{1,-1-self.dim*self.voca_dic.size}}]
	local grad_wparams = grad.params[{{1,-1-self.dim*self.voca_dic.size}}]
	cost = cost / nSample + config.lambda/2 * torch.pow(wparams,2):sum()
	grad_wparams:div(nSample):add(wparams * config.lambda)
	
	for wid,_ in pairs(tword_id) do
		cost = cost + torch.pow(self.L[{{},{wid}}],2):sum() * config.lambda_L/2
		grad.L[{{},{wid}}]:div(nSample):add(self.L[{{},{wid}}] * config.lambda_L)
	end 
	p:lap('compute grad')

	p:lap('compute cost and grad') 
	--p:printAll()

	return cost, grad, treebank, tword_id
end

end

-- check gradient
function IORNN:checkGradient(treebank, config)
	local epsilon = 1e-4

	local dim = self.dim
	local wrdDicLen = self.wrdDicLen
	local nCat = self.nCat

	local Theta = self.params
	local _, gradTheta = self:computeCostAndGrad(treebank, config)
	gradTheta = gradTheta.params
	
	local n = Theta:nElement()
	print(n)
	local numGradTheta = torch.zeros(n)
	for i = 1,n do
		local index = {{i}}
		Theta[index]:add(epsilon)
		local costPlus,_ = self:computeCostAndGrad(treebank, config)
		
		Theta[index]:add(-2*epsilon)
		local costMinus,_ = self:computeCostAndGrad(treebank, config)
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
	local wparamindex = {{1,-1-self.dim*self.voca_dic.size}}
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
	
	local epoch = 1
	local j = 0

	print('===== epoch ' .. epoch .. '=====')

	while true do
		j = j + 1
		if j > nSample/batchSize then 
			self:save(prefix .. '_' .. epoch)
			j = 1 
			epoch = epoch + 1

			parser:eval(devtreebank_path, '/tmp/parsed.conll')

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

	print('training...')
	local parser = Depparser:new(lookup, voca_dic, pos_dic, deprel_dic)
	local treebank = parser:load_train_treebank('../data/wsj-dep/toy/data/train.conll')
	--treebank = {treebank[1]}
	--treebank[1].states = {treebank[1].states[1]}
	--print(treebank)

	input = {	lookup = lookup, 
				func = tanh, funcPrime = tanhPrime, 
				voca_dic = voca_dic,
				pos_dic = pos_dic,
				deprel_dic = deprel_dic }
	net = IORNN:new(input)

	config = {lambda = 1e-4, lambda_L = 1e-7}
	net.update_L = true
	net:checkGradient(treebank, config)
]]
