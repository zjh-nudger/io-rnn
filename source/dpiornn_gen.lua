require 'depstruct'
require 'utils'
require 'dict'
require 'dp_spec'

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
					voca_dic = input.voca_dic, pos_dic = input.pos_dic, deprel_dic = input.deprel_dic }
	net.func = input.func or tanh
	net.funcPrime = input.funcPrime or tanhPrime
	setmetatable(net, IORNN_mt)

	net:init_params(input)
	return net
end

function IORNN:create_weight_matrix(params, index, size1, size2, r)
	local W = nil
	if r then 
		W = params[{{index,index+size1*size2-1}}]:resize(size1,size2):copy(uniform(size1,size2,-r,r))
	else 
		W = params[{{index,index+size1*size2-1}}]:resize(size1,size2)
	end
	return W, index+size1*size2
end

function IORNN:init_params(input)
	local dim	 = self.dim
	local wdim	 = self.wdim
	local voca_dic	 = self.voca_dic
	local deprel_dic = self.deprel_dic
	local pos_dic	 = self.pos_dic

	-- create params
	local n_params = 	dim * (wdim + 1) + 
						3 * dim + 
						dim*dim * (2 + 2*deprel_dic.size) + 2*dim + 
						(deprel_dic.size + 1) * (dim + 1) + 
						pos_dic.size * (dim + deprel_dic.size + 1) + 
						voca_dic.size * (dim + deprel_dic.size + pos_dic.size + 1) + 
						pos_dic.size * dim + 
						N_CAP_FEAT * dim + 
						voca_dic.size * wdim
			
	self.params = torch.zeros(n_params)

	--%%%%%%%%%%%%% assign ref %%%%%%%%%%%%%
	local r = 0.1
	local index = 1

	-- project word embs on to a higher-dim vector space
	self.Wh, index = self:create_weight_matrix(self.params, index, dim, wdim, 1e-3)	
	self.bh, index = self:create_weight_matrix(self.params, index, dim, 1)
	
	-- anonymous outer/inner
	self.root_inner, index = self:create_weight_matrix(self.params, index, dim, 1, 1e-3)
	self.anon_outer, index = self:create_weight_matrix(self.params, index, dim, 1, 1e-3)
	self.anon_inner, index = self:create_weight_matrix(self.params, index, dim, 1, 1e-3)

	-- composition weight matrices
	self.Wi = {}
	self.Wo = {}
	for i = 1,deprel_dic.size do
		self.Wi[i], index = self:create_weight_matrix(self.params, index, dim, dim, r)
		self.Wo[i], index = self:create_weight_matrix(self.params, index, dim, dim, r)
	end
	self.Woh, index = self:create_weight_matrix(self.params, index, dim, dim, r)
	self.Wop, index = self:create_weight_matrix(self.params, index, dim, dim, r)
	self.bi, index = self:create_weight_matrix(self.params, index, dim, 1)
	self.bo, index = self:create_weight_matrix(self.params, index, dim, 1)

	-- Pr(deprel | outer)
	self.Wdr, index = self:create_weight_matrix(self.params, index, deprel_dic.size+1, dim, r) -- +1 for EOC
	self.bdr, index = self:create_weight_matrix(self.params, index, deprel_dic.size+1, 1)

	-- Pr(POS | deprel, outer)
	self.Wpos, index	= self:create_weight_matrix(self.params, index, pos_dic.size, dim, r)
	self.Ldrpos, index	= self:create_weight_matrix(self.params, index, pos_dic.size, deprel_dic.size, r)
	self.bpos, index	= self:create_weight_matrix(self.params, index, pos_dic.size, 1)

	-- Pr(word | POS, deprel, outer)
	self.Wword, index		= self:create_weight_matrix(self.params, index, voca_dic.size, dim, r)
	self.Ldrword, index		= self:create_weight_matrix(self.params, index, voca_dic.size, deprel_dic.size, r)
	self.Lposword, index	= self:create_weight_matrix(self.params, index, voca_dic.size, pos_dic.size, r)
	self.bword, index		= self:create_weight_matrix(self.params, index, voca_dic.size, 1)

	-- POS tag 
	self.Lpos, index = self:create_weight_matrix(self.params, index, dim, pos_dic.size, r)

	-- capital letter feature
	self.Lcap, index = self:create_weight_matrix(self.params, index, dim, N_CAP_FEAT, r)

	--  word embeddings (always always always at the end of the array of params)
	self.L = self.params[{{index,index+voca_dic.size*wdim-1}}]:resize(wdim,voca_dic.size):copy(input.lookup)	-- word embeddings 
	index = index + voca_dic.size*wdim

	if index -1 ~= n_params then error('size not match') end
end

function IORNN:create_grad()
	local grad = {}
	local dim = self.dim
	local wdim = self.wdim
	local voca_dic = self.voca_dic
	local deprel_dic = self.deprel_dic
	local pos_dic = self.pos_dic

	grad.params = torch.zeros(self.params:numel())

	--%%%%%%%%%%%%% assign ref %%%%%%%%%%%%%
	local index = 1

	-- project word embs on to a higher-dim vector space
	grad.Wh, index = self:create_weight_matrix(grad.params, index, dim, wdim)	
	grad.bh, index = self:create_weight_matrix(grad.params, index, dim, 1)
	
	-- anonymous outer/inner
	grad.root_inner, index = self:create_weight_matrix(grad.params, index, dim, 1)
	grad.anon_outer, index = self:create_weight_matrix(grad.params, index, dim, 1)
	grad.anon_inner, index = self:create_weight_matrix(grad.params, index, dim, 1)

	-- composition weight matrices
	grad.Wi = {}
	grad.Wo = {}
	for i = 1,deprel_dic.size do
		grad.Wi[i], index = self:create_weight_matrix(grad.params, index, dim, dim)
		grad.Wo[i], index = self:create_weight_matrix(grad.params, index, dim, dim)
	end
	grad.Woh, index = self:create_weight_matrix(grad.params, index, dim, dim)
	grad.Wop, index = self:create_weight_matrix(grad.params, index, dim, dim)
	grad.bi, index = self:create_weight_matrix(grad.params, index, dim, 1)
	grad.bo, index = self:create_weight_matrix(grad.params, index, dim, 1)

	-- Pr(deprel | outer)
	grad.Wdr, index = self:create_weight_matrix(grad.params, index, deprel_dic.size+1, dim)
	grad.bdr, index = self:create_weight_matrix(grad.params, index, deprel_dic.size+1, 1)

	-- Pr(POS | deprel, outer)
	grad.Wpos, index	= self:create_weight_matrix(grad.params, index, pos_dic.size, dim)
	grad.Ldrpos, index	= self:create_weight_matrix(grad.params, index, pos_dic.size, deprel_dic.size)
	grad.bpos, index	= self:create_weight_matrix(grad.params, index, pos_dic.size, 1)

	-- Pr(word | POS, deprel, outer)
	grad.Wword, index		= self:create_weight_matrix(grad.params, index, voca_dic.size, dim)
	grad.Ldrword, index		= self:create_weight_matrix(grad.params, index, voca_dic.size, deprel_dic.size)
	grad.Lposword, index	= self:create_weight_matrix(grad.params, index, voca_dic.size, pos_dic.size)
	grad.bword, index		= self:create_weight_matrix(grad.params, index, voca_dic.size, 1)

	-- POS tag 
	grad.Lpos, index = self:create_weight_matrix(grad.params, index, dim, pos_dic.size)

	-- capital letter feature
	grad.Lcap, index = self:create_weight_matrix(grad.params, index, dim, N_CAP_FEAT)

	--  word embeddings (always always always at the end of the array of params)
	grad.L = grad.params[{{index,index+voca_dic.size*wdim-1}}]:resize(wdim,voca_dic.size)	-- word embeddings 
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

	local input = (self.Wh * self.L:index(2, tree.word_id))
					:add(self.Lpos:index(2, tree.pos_id))
					:add(self.Lcap:index(2, tree.cap_id))
					:add(torch.repeatTensor(self.bh, 1, tree.n_nodes))
	tree.inner:copy(self.func(input))
	tree.inner[{{},{1}}]:copy(self.root_inner)
end

function IORNN:forward_outside(tree)
	if tree.outer == nil then 
		tree.outer		= torch.zeros(self.dim, tree.n_nodes)
		tree.cstr_outer	= torch.zeros(self.dim, tree.n_nodes) -- outer rep. during construction
		tree.EOC_outer	= torch.zeros(self.dim, tree.n_nodes)
	else
		tree.outer		:fill(0)
		tree.cstr_outer	:fill(0)
		tree.EOC_outer	:fill(0)
	end

	for i = 1,tree.n_nodes do
		local col_i = {{},{i}}

		-- compute full outer
		if i == 1 then -- ROOT
			tree.outer[col_i] = self.anon_outer

		else
			local parent = tree.parent_id[i]
			local input_parent = 	(self.Woh * tree.inner[{{},{parent}}])
									:addmm(self.Wop, tree.outer[{{},{parent}}])
									:add(self.bo)
			if tree.n_children[parent] == 1 then
				tree.outer[col_i] = self.func(input_parent:add(self.anon_inner))
			else
				local input = torch.zeros(self.dim, 1)
				for j = 1, tree.n_children[parent] do
					local sister = tree.children_id[{j,parent}]
					if sister ~= i then
						input:addmm(self.Wo[tree.deprel_id[sister]], tree.inner[{{},{sister}}])
					end
				end
				tree.outer[col_i] = self.func(input_parent:add(input:div(tree.n_children[parent]-1)))
			end
		end

		-- compute children's constr. outers and EOC outer
		local input_head = (self.Woh * tree.inner[col_i]):addmm(self.Wop, tree.outer[col_i]):add(self.bo)

		if tree.n_children[i] == 0 then 
			tree.EOC_outer[col_i] = self.func(input_head+self.anon_inner)

		else 
			local input			= torch.zeros(self.dim, 1)
			local left_sister	= nil
			
			-- compute outer rep. for its children
			for j = 1, tree.n_children[i] do
				local child = tree.children_id[{j,i}]
				local col_c = {{},{child}}

				-- compute constructed outer
				if left_sister then 
					input:addmm(self.Wo[tree.deprel_id[left_sister]], tree.inner[{{},{left_sister}}])
					tree.cstr_outer[col_c] = self.func(torch.div(input, j-1):add(input_head))
				else 
					tree.cstr_outer[col_c] = self.func(input_head + self.anon_inner)
				end
				left_sister = child			
			end

			-- compute outer rep. for EOC
			input:addmm(self.Wo[tree.deprel_id[left_sister]], tree.inner[{{},{left_sister}}])
			tree.EOC_outer[col_i] = self.func(input:div(tree.n_children[i]):add(input_head))
		end
	end

	-- compute probabilities
	-- Pr(deprel | outer)
	tree.deprel_score	= (self.Wdr * tree.cstr_outer):add(torch.repeatTensor(self.bdr, 1, tree.n_nodes))
	tree.deprel_prob	= safe_compute_softmax(tree.deprel_score)
	tree.EOC_score	= (self.Wdr * tree.EOC_outer):add(torch.repeatTensor(self.bdr, 1, tree.n_nodes))
	tree.EOC_prob	= safe_compute_softmax(tree.EOC_score)

	-- Pr(pos | deprel, outer)
	tree.pos_score	= 	(self.Wpos * tree.cstr_outer)
						:add(self.Ldrpos:index(2, tree.deprel_id))
						:add(torch.repeatTensor(self.bpos, 1, tree.n_nodes))
	tree.pos_prob	= safe_compute_softmax(tree.pos_score)

	-- Pr(word | pos, deprel, outer)
	tree.word_score	= 	(self.Wword * tree.cstr_outer)
						:add(self.Ldrword:index(2, tree.deprel_id))
						:add(self.Lposword:index(2, tree.pos_id))
						:add(torch.repeatTensor(self.bword, 1, tree.n_nodes))
	tree.word_prob	= safe_compute_softmax(tree.word_score)

	-- compute error
	tree.total_err = 0
	for i = 2, tree.n_nodes do
		tree.total_err = tree.total_err - math.log(tree.deprel_prob[{tree.deprel_id[i],i}])
										- math.log(tree.pos_prob[{tree.pos_id[i],i}])
										- math.log(tree.word_prob[{tree.word_id[i],i}])
	end
	tree.total_err = tree.total_err - torch.log(tree.EOC_prob[{self.deprel_dic.size+1,{}}]):sum()

	return tree.total_err
end

--*********************** backpropagate *********************--
function IORNN:backpropagate_outside(tree, grad)
	if tree.gradi == nil then
		tree.gradi		= torch.zeros(self.dim, tree.n_nodes)
		tree.grado		= torch.zeros(self.dim, tree.n_nodes)
		tree.gradcstro	= torch.zeros(self.dim, tree.n_nodes)
		tree.gradEOCo	= torch.zeros(self.dim, tree.n_nodes)
	else
		tree.gradi		:fill(0)
		tree.grado		:fill(0)
		tree.gradcstro	:fill(0)
		tree.gradEOCo	:fill(0)
	end

	local gZdr		= tree.deprel_prob	:clone()
	local gZpos		= tree.pos_prob		:clone()
	local gZword	= tree.word_prob	:clone()
	local gZEOC		= tree.EOC_prob		:clone()

	for i = 2, tree.n_nodes do
		gZdr[{tree.deprel_id[i],i}]	= gZdr[{tree.deprel_id[i],i}]	- 1
		gZpos[{tree.pos_id[i],i}]	= gZpos[{tree.pos_id[i],i}]		- 1
		gZword[{tree.word_id[i],i}]	= gZword[{tree.word_id[i],i}]	- 1
	end
	gZdr[{{},{1}}]	:fill(0) -- don't take ROOT into account
	gZpos[{{},{1}}]	:fill(0)
	gZword[{{},{1}}]:fill(0)

	gZEOC[{self.deprel_dic.size+1,{}}]:add(-1)

	-- for Pr( . | context)
	grad.Wdr		:addmm(gZdr, tree.cstr_outer:t())
					:addmm(gZEOC, tree.EOC_outer:t())
	grad.bdr		:add(gZdr:sum(2))
					:add(gZEOC:sum(2))
	tree.gradcstro	:addmm(self.Wdr:t(), gZdr)
	tree.gradEOCo	:addmm(self.Wdr:t(), gZEOC)

	grad.Wpos		:addmm(gZpos, tree.cstr_outer:t())
	grad.bpos		:add(gZpos:sum(2))
	tree.gradcstro	:addmm(self.Wpos:t(), gZpos)

	grad.Wword		:addmm(gZword, tree.cstr_outer:t())
	grad.bword		:add(gZword:sum(2))
	tree.gradcstro	:addmm(self.Wword:t(), gZword)

	for i = 2,tree.n_nodes do
		grad.Ldrpos[{{},{tree.deprel_id[i]}}]	:add(gZpos[{{},{i}}])
		grad.Ldrword[{{},{tree.deprel_id[i]}}]	:add(gZword[{{},{i}}])
		grad.Lposword[{{},{tree.pos_id[i]}}]	:add(gZword[{{},{i}}])
	end

	-- backward 
	tree.gradZEOCo  = tree.gradEOCo:cmul(self.funcPrime(tree.EOC_outer))
	tree.gradZcstro = tree.gradcstro:cmul(self.funcPrime(tree.cstr_outer))

	grad.Woh	:addmm(tree.gradZEOCo, tree.inner:t())
	grad.Wop	:addmm(tree.gradZEOCo, tree.outer:t())
	grad.bo		:add(tree.gradZEOCo:sum(2))

	tree.gradi	:addmm(self.Woh:t(), tree.gradZEOCo)
	tree.grado	:addmm(self.Wop:t(), tree.gradZEOCo)

	for i = tree.n_nodes, 1, -1 do
		local col_i = {{},{i}}

		-- for EOC outer
		local gz = tree.gradZEOCo[col_i]

		if tree.n_children[i] == 0 then
			grad.anon_inner:add(gz)
		else 
			local t = 1/tree.n_children[i]
			for j = 1,tree.n_children[i] do
				local child = tree.children_id[{j,i}]
				local col_c = {{},{child}}
				grad.Wo[tree.deprel_id[child]]:addmm(t, gz, tree.inner[col_c]:t())
				tree.gradi[col_c]			  :addmm(t, self.Wo[tree.deprel_id[child]]:t(), gz)
			end
		end

		-- for children's constr outers
		for j = 1,tree.n_children[i] do
			local child = tree.children_id[{j,i}]
			local col_c = {{},{child}}
			local gz = tree.gradZcstro[col_c]

			grad.Woh:addmm(gz, tree.inner[col_i]:t())
			grad.Wop:addmm(gz, tree.outer[col_i]:t())
			grad.bo	:add(gz)

			tree.gradi[col_i]:addmm(self.Woh:t(), gz)
			tree.grado[col_i]:addmm(self.Wop:t(), gz)
	
			if j == 1 then 
				grad.anon_inner:add(gz)
			else
				local t = 1 / (j-1)
				for k = 1,j-1 do
					local sister = tree.children_id[{k,i}]
					local col_s = {{},{sister}}
					grad.Wo[tree.deprel_id[sister]]	:addmm(t, gz, tree.inner[col_s]:t())
					tree.gradi[col_s]				:addmm(t, self.Wo[tree.deprel_id[sister]]:t(), gz)
				end
			end
		end

		-- for full outer
		if i == 1 then 
			grad.anon_outer:add(tree.grado[{{},{1}}])

		else 
			local parent = tree.parent_id[i]
			local col_p = {{},{parent}}
			local gz = tree.grado[col_i]:cmul(self.funcPrime(tree.outer[col_i]))

			grad.Woh:addmm(gz, tree.inner[col_p]:t())	
			grad.Wop:addmm(gz, tree.outer[col_p]:t())
			grad.bo	:add(gz)

			tree.gradi[col_p]:addmm(self.Woh:t(), gz)
			tree.grado[col_p]:addmm(self.Wop:t(), gz)

			if tree.n_children[parent] == 1 then
				grad.anon_inner:add(gz)
			else
				local t = 1 / (tree.n_children[parent] - 1)
				for j = 1,tree.n_children[parent] do
					local sister = tree.children_id[{j,parent}]
					if sister ~= i then
						local col_s = {{},{sister}}
						grad.Wo[tree.deprel_id[sister]]	:addmm(t, gz, tree.inner[col_s]:t())
						tree.gradi[col_s]				:addmm(t, self.Wo[tree.deprel_id[sister]]:t(), gz)
					end
				end
			end	
		end
	end
end

function IORNN:backpropagate_inside(tree, grad)
	-- root
	grad.root_inner:add(tree.gradi[{{},{1}}])

	tree.gradZi = tree.gradi:cmul(self.funcPrime(tree.inner))
	grad.Wh		:addmm(tree.gradZi[{{},{2,-1}}], self.L:index(2, tree.word_id[{{2,-1}}]):t())
	grad.bh		:add(tree.gradZi[{{},{2,-1}}]:sum(2))

	for i = 2, tree.n_nodes do
		local col = {{},{i}}
		local gz = tree.gradZi[col]
		grad.L[{{},{tree.word_id[i]}}]:addmm(self.Wh:t(), gz)
		grad.Lpos[{{},{tree.pos_id[i]}}]:add(gz)
		grad.Lcap[{{},{tree.cap_id[i]}}]:add(gz)
	end
end

function IORNN:computeCostAndGrad(treebank, config, grad, parser)
	local parse = config.parse or false

	p:start('compute cost and grad')	

	grad.params:fill(0)  -- always make sure that this grad is intialized with 0

	local cost = 0
	local nSample = 0
	local tword_id = {}

	p:start('process treebank')
	for i, ds in ipairs(treebank) do
		local tree = ds:to_torch_matrix_tree()
		self:forward_inside(tree)
		cost = cost + self:forward_outside(tree)
		self:backpropagate_outside(tree, grad)
		self:backpropagate_inside(tree, grad)

		nSample = nSample + tree.n_nodes-1
		for i=2,tree.wnode_id:numel() do -- do not take the root into account
			tword_id[tree.word_id[tree.wnode_id[i]]] = 1
		end
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
	--os.execute('th eval_depparser.lua '..prefix..'_'..epoch..' '..devtreebank_path)


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


--********************************** test ******************************--
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
L = torch.rand(2, voca_dic.size)

print('training...')
local net = IORNN:new({ dim = dim, voca_dic = voca_dic, pos_dic = pos_dic, deprel_dic = deprel_dic,
						lookup = L, func = tanh, funcPrime = tanhPrime })

local parser = Depparser:new(voca_dic, pos_dic, deprel_dic)
local treebank,_ = parser:load_treebank('../data/wsj-dep/toy/data/train.conll')

config = {lambda = 1e-4, lambda_L = 1e-7}
net.update_L = true
net:checkGradient(treebank, parser, config)

