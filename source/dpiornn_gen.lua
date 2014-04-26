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
function IORNN:forward_inside(forest)
	if forest.inner == nil then
		forest.inner = torch.zeros(self.dim, forest.n_nodes)
	else
		forest.inner:fill(0)
	end

	local input = (self.Wh * self.L:index(2, forest.word))
					:add(self.Lpos:index(2, forest.pos))
					:add(self.Lcap:index(2, forest.cap))
					:add(torch.repeatTensor(self.bh, 1, forest.n_nodes))
	forest.inner:copy(self.func(input))

	for i = 1, forest.n_trees do
		forest.inner[{{},{forest.tree_index[{1,i}]}}]:copy(self.root_inner)
	end
end

function IORNN:forward_outside(forest)

	-- create matrixes
	local attrs = { 'outer', 'cstr_outer', 'EOC_outer', 
					'deprel_score', 'deprel_prob', 'EOC_score', 'EOC_prob',
					'pos_score', 'pos_prob', 'word_score', 'word_prob' }

	if forest.outer == nil then 
		forest.outer 		= torch.zeros(self.dim, forest.n_nodes)
		forest.cstr_outer 	= torch.zeros(self.dim, forest.n_nodes)
		forest.EOC_outer 	= torch.zeros(self.dim, forest.n_nodes)

		forest.deprel_score	= torch.zeros(self.deprel_dic.size+1, forest.n_nodes)
		forest.deprel_prob	= torch.zeros(self.deprel_dic.size+1, forest.n_nodes)
		forest.EOC_score	= torch.zeros(self.deprel_dic.size+1, forest.n_nodes)
		forest.EOC_prob		= torch.zeros(self.deprel_dic.size+1, forest.n_nodes)

		forest.pos_score	= torch.zeros(self.pos_dic.size, forest.n_nodes)
		forest.pos_prob		= torch.zeros(self.pos_dic.size, forest.n_nodes)

		forest.word_score	= torch.zeros(self.voca_dic.size, forest.n_nodes)
		forest.word_prob	= torch.zeros(self.voca_dic.size, forest.n_nodes)
	else
		for _,attr in ipairs(attrs) do
			forest[attr]:fill(0)
		end
	end

	-- compute outers
	for t = 1, forest.n_trees do 
	local root_id = forest.tree_index[{1,t}]
	local end_id = forest.tree_index[{2,t}]

	for i = root_id, end_id  do
		local col_i = {{},{i}}

		-- compute full outer
		if i == root_id then -- ROOT
			forest.outer[col_i]:copy(self.anon_outer)

		else
			local parent = forest.parent[i]
			local input_parent = 	(self.Woh * forest.inner[{{},{parent}}])
									:addmm(self.Wop, forest.outer[{{},{parent}}])
									:add(self.bo)
			if forest.n_children[parent] == 1 then
				forest.outer[col_i] = self.func(input_parent:add(self.anon_inner))
			else
				local input = torch.zeros(self.dim, 1)
				for j = 1, forest.n_children[parent] do
					local sister = forest.children[{j,parent}]
					if sister ~= i then
						input:addmm(self.Wo[forest.deprel[sister]], forest.inner[{{},{sister}}])
					end
				end
				forest.outer[col_i] = self.func(input_parent:add(input:div(forest.n_children[parent]-1)))
			end
		end

		-- compute children's constr. outers and EOC outer
		local input_head = (self.Woh * forest.inner[col_i]):addmm(self.Wop, forest.outer[col_i]):add(self.bo)

		if forest.n_children[i] == 0 then 
			forest.EOC_outer[col_i] = self.func(input_head+self.anon_inner)

		else 
			local input			= torch.zeros(self.dim, 1)
			local left_sister	= nil
			
			-- compute outer rep. for its children
			for j = 1, forest.n_children[i] do
				local child = forest.children[{j,i}]
				local col_c = {{},{child}}

				-- compute constructed outer
				if left_sister then 
					input:addmm(self.Wo[forest.deprel[left_sister]], forest.inner[{{},{left_sister}}])
					forest.cstr_outer[col_c] = self.func(torch.div(input, j-1):add(input_head))
				else 
					forest.cstr_outer[col_c] = self.func(input_head + self.anon_inner)
				end
				left_sister = child
			end

			-- compute outer rep. for EOC
			input:addmm(self.Wo[forest.deprel[left_sister]], forest.inner[{{},{left_sister}}])
			forest.EOC_outer[col_i] = self.func(input:div(forest.n_children[i]):add(input_head))
		end
	end
	end

	-- compute probabilities
	-- Pr(deprel | outer)
	forest.deprel_score	= (self.Wdr * forest.cstr_outer):add(torch.repeatTensor(self.bdr, 1, forest.n_nodes))
	forest.deprel_prob	= safe_compute_softmax(forest.deprel_score)
	forest.EOC_score	= (self.Wdr * forest.EOC_outer):add(torch.repeatTensor(self.bdr, 1, forest.n_nodes))
	forest.EOC_prob		= safe_compute_softmax(forest.EOC_score)

	-- Pr(pos | deprel, outer)
	forest.pos_score	= 	(self.Wpos * forest.cstr_outer)
									:add(self.Ldrpos:index(2, forest.deprel))
									:add(torch.repeatTensor(self.bpos, 1, forest.n_nodes))
	forest.pos_prob		= safe_compute_softmax(forest.pos_score)

	-- Pr(word | pos, deprel, outer)
	forest.word_score	= 	(self.Wword * forest.cstr_outer)
									:add(self.Ldrword:index(2, forest.deprel))
									:add(self.Lposword:index(2, forest.pos))
									:add(torch.repeatTensor(self.bword, 1, forest.n_nodes))
	forest.word_prob	= safe_compute_softmax(forest.word_score)

	-- compute error
	local tree_err = torch.zeros(forest.n_trees)
	for t = 1, forest.n_trees do
		for i = forest.tree_index[{1,t}]+1, forest.tree_index[{2,t}] do
			tree_err[t] = tree_err[t]	- math.log(forest.deprel_prob[{forest.deprel[i],i}])
										- math.log(forest.pos_prob[{forest.pos[i],i}])
										- math.log(forest.word_prob[{forest.word[i],i}])
		end
		tree_err[t] = tree_err[t] - torch.log(forest.EOC_prob[{self.deprel_dic.size+1,{forest.tree_index[{1,t}], forest.tree_index[{2,t}]}}]):sum()
	end

	return tree_err
end

--*********************** backpropagate *********************--
function IORNN:backpropagate_outside(forest, grad)

	-- create matrices
	local attrs = {'gradi', 'grado', 'gradcstro', 'gradEOCo'}

	if forest.gradi == nil then
		for _,attr in ipairs(attrs) do
			forest[attr] = torch.zeros(self.dim, forest.n_nodes)
		end
	else
		for _,attr in ipairs(attrs) do
			forest[attr]:fill(0)
		end
	end

	-- compute grads for classification
	local gZdr		= forest.deprel_prob
	local gZpos		= forest.pos_prob
	local gZword	= forest.word_prob
	local gZEOC		= forest.EOC_prob

	for i = 1, forest.n_nodes do
		gZdr	[{forest.deprel[i],i}]	= gZdr	[{forest.deprel[i],i}]	- 1
		gZpos	[{forest.pos[i],i}]		= gZpos	[{forest.pos[i],i}]		- 1
		gZword	[{forest.word[i],i}]	= gZword[{forest.word[i],i}]	- 1
	end

	for t = 1,forest.n_trees do
		local root = forest.tree_index[{1,t}]
		gZdr	[{{},{root}}]:fill(0) -- don't take ROOT into account
		gZpos	[{{},{root}}]:fill(0)
		gZword	[{{},{root}}]:fill(0)
	end

	gZEOC[{self.deprel_dic.size+1,{}}]:add(-1)

	-- for Pr( . | context)
	grad.Wdr		:addmm(gZdr, forest.cstr_outer:t())
					:addmm(gZEOC, forest.EOC_outer:t())
	grad.bdr		:add(gZdr:sum(2))
					:add(gZEOC:sum(2))

	forest.gradcstro:addmm(self.Wdr:t(), gZdr)
	forest.gradEOCo	:addmm(self.Wdr:t(), gZEOC)

	grad.Wpos		:addmm(gZpos, forest.cstr_outer:t())
	grad.bpos		:add(gZpos:sum(2))
	forest.gradcstro:addmm(self.Wpos:t(), gZpos)

	grad.Wword		:addmm(gZword, forest.cstr_outer:t())
	grad.bword		:add(gZword:sum(2))
	forest.gradcstro:addmm(self.Wword:t(), gZword)

	for t = 1, forest.n_trees do
		for i = forest.tree_index[{1,t}]+1, forest.tree_index[{2,t}] do
			grad.Ldrpos[{{},{forest.deprel[i]}}]	:add(gZpos[{{},{i}}])
			grad.Ldrword[{{},{forest.deprel[i]}}]	:add(gZword[{{},{i}}])
			grad.Lposword[{{},{forest.pos[i]}}]		:add(gZword[{{},{i}}])
		end
	end

	-- backward 
	forest.gradZEOCo  = forest.gradEOCo	:cmul(self.funcPrime(forest.EOC_outer))
	forest.gradZcstro = forest.gradcstro:cmul(self.funcPrime(forest.cstr_outer))

	grad.Woh	:addmm(forest.gradZEOCo, forest.inner:t())
	grad.Wop	:addmm(forest.gradZEOCo, forest.outer:t())
	grad.bo		:add(forest.gradZEOCo:sum(2))

	forest.gradi	:addmm(self.Woh:t(), forest.gradZEOCo)
	forest.grado	:addmm(self.Wop:t(), forest.gradZEOCo)

	for t = 1, forest.n_trees do
	local root = forest.tree_index[{1,t}]
	local endid	 = forest.tree_index[{2,t}]

	for i = endid, root, -1 do
		local col_i = {{},{i}}

		-- for EOC outer
		local gz = forest.gradZEOCo[col_i]

		if forest.n_children[i] == 0 then
			grad.anon_inner:add(gz)

		else 
			local num = 1/forest.n_children[i]
			for j = 1,forest.n_children[i] do
				local child = forest.children[{j,i}]
				local col_c = {{},{child}}
				grad.Wo[forest.deprel[child]]	:addmm(num, gz, forest.inner[col_c]:t())
				forest.gradi[col_c]				:addmm(num, self.Wo[forest.deprel[child]]:t(), gz)
			end
		end

		-- for children's constr outers
		for j = 1,forest.n_children[i] do
			local child = forest.children[{j,i}]
			local col_c = {{},{child}}
			local gz = forest.gradZcstro[col_c]

			grad.Woh:addmm(gz, forest.inner[col_i]:t())
			grad.Wop:addmm(gz, forest.outer[col_i]:t())
			grad.bo	:add(gz)

			forest.gradi[col_i]:addmm(self.Woh:t(), gz)
			forest.grado[col_i]:addmm(self.Wop:t(), gz)
	
			if j == 1 then 
				grad.anon_inner:add(gz)
			else
				local t = 1 / (j-1)
				for k = 1,j-1 do
					local sister = forest.children[{k,i}]
					local col_s = {{},{sister}}
					grad.Wo[forest.deprel[sister]]	:addmm(t, gz, forest.inner[col_s]:t())
					forest.gradi[col_s]				:addmm(t, self.Wo[forest.deprel[sister]]:t(), gz)
				end
			end
		end

		-- for full outer
		if i == root then 
			grad.anon_outer:add(forest.grado[{{},{root}}])

		else 
			local parent = forest.parent[i]
			local col_p = {{},{parent}}
			local gz = forest.grado[col_i]:cmul(self.funcPrime(forest.outer[col_i]))
	
			grad.Woh:addmm(gz, forest.inner[col_p]:t())	
			grad.Wop:addmm(gz, forest.outer[col_p]:t())
			grad.bo	:add(gz)

			forest.gradi[col_p]:addmm(self.Woh:t(), gz)
			forest.grado[col_p]:addmm(self.Wop:t(), gz)

			if forest.n_children[parent] == 1 then
				grad.anon_inner:add(gz)
			else
				local num = 1 / (forest.n_children[parent] - 1)
				for j = 1,forest.n_children[parent] do
					local sister = forest.children[{j,parent}]
					if sister ~= i then
						local col_s = {{},{sister}}
						grad.Wo[forest.deprel[sister]]	:addmm(num, gz, forest.inner[col_s]:t())
						forest.gradi[col_s]				:addmm(num, self.Wo[forest.deprel[sister]]:t(), gz)
					end
				end
			end	
		end
	end
	end
end

function IORNN:backpropagate_inside(forest, grad)
	-- root
	grad.root_inner:add(forest.gradi:index(2, forest.tree_index[{1,{}}]):sum(2))

	forest.gradZi 	= forest.gradi:cmul(self.funcPrime(forest.inner))
	for t = 1,forest.n_trees do 
		forest.gradZi[{{},forest.tree_index[{1,t}]}]:fill(0)
	end
	grad.Wh	:addmm(forest.gradZi, self.L:index(2, forest.word):t())
	grad.bh	:add(forest.gradZi:sum(2))

	for t = 1, forest.n_trees do
		for i = forest.tree_index[{1,t}]+1, forest.tree_index[{2,t}] do
			local col = {{},{i}}
			local gz = forest.gradZi[col]
			grad.L[{{},{forest.word[i]}}]:addmm(self.Wh:t(), gz)
			grad.Lpos[{{},{forest.pos[i]}}]:add(gz)
			grad.Lcap[{{},{forest.cap[i]}}]:add(gz)
		end
	end
end

function IORNN:compute_log_prob(dsbank)
	local forest = Depstruct:to_torch_matrix_forest(dsbank)
	self:forward_inside(forest)
	return self:forward_outside(forest)
end

function IORNN:computeCostAndGrad(dsbank, config, grad)
	local parse = config.parse or false

	p:start('compute cost and grad')	

	grad.params:fill(0)  -- always make sure that this grad is intialized with 0

	local cost = 0
	local nSample = 0
	local tword = {}

	-- process dsbank
	p:start('process dsbank')
	local forest = Depstruct:to_torch_matrix_forest(dsbank)
	self:forward_inside(forest)
	local tree_err = self:forward_outside(forest)
	self:backpropagate_outside(forest, grad)
	self:backpropagate_inside(forest, grad)

	cost = cost + tree_err:sum()
	nSample = nSample + (forest.n_nodes-1)*3 + 1
	for i=2,forest.wnode:numel() do -- do not take the root into account
		tword[forest.word[forest.wnode[i]]] = 1
	end
	p:lap('process dsbank') 

	-- compute grad
	p:start('compute grad')
	local wparams = self.params[{{1,-1-self.wdim*self.voca_dic.size}}]
	local grad_wparams = grad.params[{{1,-1-self.wdim*self.voca_dic.size}}]
	cost = cost / nSample + config.lambda/2 * torch.pow(wparams,2):sum()
	grad_wparams:div(nSample):add(wparams * config.lambda)
	
	for wid,_ in pairs(tword) do
		cost = cost + torch.pow(self.L[{{},{wid}}],2):sum() * config.lambda_L/2
		grad.L[{{},{wid}}]:div(nSample):add(config.lambda_L, self.L[{{},{wid}}])
	end 
	p:lap('compute grad')

	p:lap('compute cost and grad') 

	return cost, grad, tword
end

-- check gradient
function IORNN:checkGradient(dsbank, config)
	local epsilon = 1e-4

	local dim = self.dim
	local wrdDicLen = self.wrdDicLen
	local nCat = self.nCat

	local Theta = self.params
	local grad = self:create_grad()
	local _, gradTheta = self:computeCostAndGrad(dsbank, config, grad)
	gradTheta = gradTheta.params
	
	local n = Theta:nElement()
	print(n)
	local numGradTheta = torch.zeros(n)
	for i = 1,n do
		local index = {{i}}
		Theta[index]:add(epsilon)
		local grad = self:create_grad()
		local costPlus,_ = self:computeCostAndGrad(dsbank, config, grad, parser)
		
		Theta[index]:add(-2*epsilon)
		local costMinus,_ = self:computeCostAndGrad(dsbank, config, grad, parser)
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
	local cost, grad, tword = func()

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
	for wid,_ in pairs(tword) do
		local col_i = {{},{wid}}
		state.paramVariance.L[col_i]:addcmul(1,grad.L[col_i],grad.L[col_i])
		torch.sqrt(state.paramStd.L[col_i],state.paramVariance.L[col_i])
		self.L[col_i]:addcdiv(-voca_dic_clr, grad.L[col_i],state.paramStd.L[col_i]:add(1e-10))
	end

	-- (5) update evaluation counter
	state.evalCounter = state.evalCounter + 1
end

function IORNN:train_with_adagrad(traindsbank, batchSize, 
									maxepoch, lambda, prefix,
									adagrad_config, adagrad_state, 
									devdsbank_path, kbestdevdsbank_path)
	local nSample = #traindsbank
	local grad = self:create_grad()
	
	local epoch = 0
	local j = 0
	--os.execute('th eval_depparser_rerank.lua '..prefix..'_'..epoch..' '..devdsbank_path..' '..kbestdevdsbank_path)


	epoch = epoch + 1
	print('===== epoch ' .. epoch .. '=====')

	while true do
		j = j + 1
		if j > nSample/batchSize then 
			self:save(prefix .. '_' .. epoch)
			os.execute('th eval_depparser_rerank.lua '..prefix..'_'..epoch..' '..devdsbank_path..' '..kbestdevdsbank_path)

			j = 1 
			epoch = epoch + 1
			if epoch > maxepoch then break end
			print('===== epoch ' .. epoch .. '=====')
		end

		local subdsbank = {}
		for k = 1,batchSize do
			subdsbank[k] = traindsbank[k+(j-1)*batchSize]
		end
	
		local function func()
			cost, grad, tword  = self:computeCostAndGrad(subdsbank, 
							{lambda = lambda.lambda, lambda_L=lambda.lambda_L}, grad)

			print('iter ' .. j .. ': ' .. cost) io.flush()		
			return cost, grad, tword
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
require 'depparser_rerank'
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
local dsbank,_ = parser:load_dsbank('../data/wsj-dep/toy/data/train.conll')

local forest = Depstruct:to_torch_matrix_forest(dsbank)
for k,v in pairs(forest) do
	print(k)
	print(v)
end

config = {lambda = 1e-4, lambda_L = 1e-7}
net.update_L = true
--net:checkGradient(dsbank, config)
]]
