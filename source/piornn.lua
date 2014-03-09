require 'tree'
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
	return torch.cdiv(torch.ones(X:size()), (-X):exp():add(1))
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
	return torch.ones(X:size())
end

--************************* construction ********************--
-- create a new recursive autor encoder with a given structure
function IORNN:new(input)

	local dim		= input.t_lookup:size(1)
	local voca		= input.voca
	local rules		= input.rules
	local subdim	= input.subdim

	local net = {dim = dim, subdim = subdim, voca = voca}
	net.func = input.func
	net.funcPrime = input.funcPrime
	setmetatable(net, IORNN_mt)

	net:init_params(input)
	return net
end

function IORNN:init_params(input)
	local net = self
	local dim = net.dim
	local subdim = net.subdim
	local voca = net.voca

	net.rules = {}

	-- create params
	local n_params = 0
	n_params = n_params + dim	-- outer at root
	for i,rule in ipairs(input.rules) do
		for j = 1,#rule.rhs do
			n_params = n_params + 4*(dim * subdim) + 3*dim
		end		
		n_params = n_params + 2*(dim*subdim) + 2*dim
	end
	n_params = n_params + 2*(dim * voca:size())	-- target & conditional embeddings
	net.params = torch.zeros(n_params)

	--%%%%%%%%%%%%% assign ref %%%%%%%%%%%%%
	local r = 0.1
	local index = 1
	
	-- root outer
	net.root_outer = net.params[{{index,index+dim-1}}]:resize(dim,1):copy(uniform(dim, 1, -1e-3, 1e-3))
	index = index + dim

	-- weights
	for i,rule in ipairs(input.rules) do
		net.rules[i] = {}

		net.rules[i].Wi_U = {}
		net.rules[i].Wi_V = {}
		net.rules[i].Wi_r = {}

		net.rules[i].Wo_U = {}
		net.rules[i].Wo_V = {}
		net.rules[i].Wo_r = {}

		net.rules[i].bo = {}

		for j = 1,#rule.rhs do
			net.rules[i].Wi_U[j] = net.params[{{index,index+dim*subdim-1}}]:resize(dim,subdim):copy(uniform(dim, subdim, -r, r))
			index = index + dim*subdim
			net.rules[i].Wi_V[j] = net.params[{{index,index+dim*subdim-1}}]:resize(subdim,dim):copy(uniform(subdim, dim, -r, r))
			index = index + dim*subdim
			net.rules[i].Wi_r[j] = net.params[{{index,index+dim-1}}]:resize(dim,1):copy(uniform(dim, 1, -r, r))
			index = index + dim

			net.rules[i].Wo_U[j] = net.params[{{index,index+dim*subdim-1}}]:resize(dim,subdim):copy(uniform(dim, subdim, -r, r))
			index = index + dim*subdim
			net.rules[i].Wo_V[j] = net.params[{{index,index+subdim*dim-1}}]:resize(subdim,dim):copy(uniform(subdim, dim, -r, r))
			index = index + subdim*dim
			net.rules[i].Wo_r[j] = net.params[{{index,index+dim-1}}]:resize(dim,1):copy(uniform(dim, 1, -r, r))
			index = index + dim

			net.rules[i].bo[j] = net.params[{{index,index+dim-1}}]:resize(dim,1):copy(torch.zeros(dim, 1))
			index = index + dim
		end		

		net.rules[i].bi = net.params[{{index,index+dim-1}}]:resize(dim,1)
		index = index + dim
		net.rules[i].Wop_U = net.params[{{index,index+dim*subdim-1}}]:resize(dim,subdim):copy(uniform(dim, subdim, -r, r))
		index = index + dim*subdim
		net.rules[i].Wop_V = net.params[{{index,index+subdim*dim-1}}]:resize(subdim,dim):copy(uniform(subdim, dim, -r, r))
		index = index + subdim*dim
		net.rules[i].Wop_r = net.params[{{index,index+dim-1}}]:resize(dim,1):copy(uniform(dim, 1, -r, r))
		index = index + dim
	end

	-- target/conditional embeddings
	net.Lt = net.params[{{index,index+voca:size()*dim-1}}]:resize(dim,voca:size()):copy(input.t_lookup)		-- target lookup
	index = index + voca:size()*dim
	net.Lc = net.params[{{index,index+voca:size()*dim-1}}]:resize(dim,voca:size()):copy(input.c_lookup)		-- conditional lookup
	index = index + voca:size()*dim
end

function IORNN:create_grad()
	local grad = {}
	local dim = self.dim
	local subdim = self.subdim
	local voca = self.voca

	grad.rules = {}
	grad.params = torch.zeros(self.params:nElement())

	--%%%%%%%%%%%%% assign ref %%%%%%%%%%%%%
	local index = 1
	
	-- root outer
	grad.root_outer = grad.params[{{index,index+dim-1}}]:resize(dim,1)
	index = index + dim

	-- weights
	for i,rule in ipairs(self.rules) do
		grad.rules[i] = {}

		grad.rules[i].Wi_U = {}
		grad.rules[i].Wi_V = {}
		grad.rules[i].Wi_r = {}

		grad.rules[i].Wo_U = {}
		grad.rules[i].Wo_V = {}
		grad.rules[i].Wo_r = {}

		grad.rules[i].bo = {}

		for j = 1,#rule.Wi_U do
			grad.rules[i].Wi_U[j] = grad.params[{{index,index+dim*subdim-1}}]:resize(dim,subdim)
			index = index + dim*subdim
			grad.rules[i].Wi_V[j] = grad.params[{{index,index+dim*subdim-1}}]:resize(subdim,dim)
			index = index + dim*subdim
			grad.rules[i].Wi_r[j] = grad.params[{{index,index+dim-1}}]:resize(dim,1)
			index = index + dim

			grad.rules[i].Wo_U[j] = grad.params[{{index,index+dim*subdim-1}}]:resize(dim,subdim)
			index = index + dim*subdim
			grad.rules[i].Wo_V[j] = grad.params[{{index,index+subdim*dim-1}}]:resize(subdim,dim)
			index = index + subdim*dim
			grad.rules[i].Wo_r[j] = grad.params[{{index,index+dim-1}}]:resize(dim,1)
			index = index + dim

			grad.rules[i].bo[j] = grad.params[{{index,index+dim-1}}]:resize(dim,1)
			index = index + dim
		end		

		grad.rules[i].bi = grad.params[{{index,index+dim-1}}]:resize(dim,1)
		index = index + dim
		grad.rules[i].Wop_U = grad.params[{{index,index+dim*subdim-1}}]:resize(dim,subdim)
		index = index + dim*subdim
		grad.rules[i].Wop_V = grad.params[{{index,index+subdim*dim-1}}]:resize(subdim,dim)
		index = index + subdim*dim
		grad.rules[i].Wop_r = grad.params[{{index,index+dim-1}}]:resize(dim,1)
		index = index + dim
	end

	-- target/conditional embeddings
	grad.Lt = grad.params[{{index,index+voca:size()*dim-1}}]:resize(dim,voca:size())		-- target lookup
	index = index + voca:size()*dim
	grad.Lc = grad.params[{{index,index+voca:size()*dim-1}}]:resize(dim,voca:size())		-- conditional lookup
	index = index + voca:size()*dim

	return grad
end

-- save net into a bin file
function IORNN:save( filename , binary )
	local file = torch.DiskFile(filename, 'w')
	if binary == nil or binary then print(binary) file:binary() end
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
	setmetatable(net.voca, Dict_mt)

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
			tree.inner[col_i]:copy(self.Lc[{{},{tree.word_id[i]}}])

		-- for internal nodes
		else
			local rule = self.rules[tree.rule_id[i]]
			if (#rule.Wi_U ~= tree.n_children[i]) then 
				error("rules not match")
			end

			local temp = tree.inner[{{},{tree.children_id[{1,i}]}}] 
			local input = 	(rule.Wi_U[1] * (rule.Wi_V[1] * temp))
							:add(torch.cmul(rule.Wi_r[1], temp))
							
			for j = 2,tree.n_children[i] do
				local temp = tree.inner[{{},{tree.children_id[{j,i}]}}]
				input	:add(rule.Wi_U[j] * (rule.Wi_V[j] * temp))
						:add(torch.cmul(rule.Wi_r[j], temp))
			end
			input:add(rule.bi)
			tree.inner[col_i]:copy(self.func(input))
		end
	end
end

function IORNN:forward_outside(tree)	
	if tree.outer == nil then 
		tree.outer = torch.Tensor(self.dim, tree.n_nodes)
	else
		tree.outer:fill(0)
	end

	-- process
	tree.outer[{{},{1}}]:copy(self.root_outer)

	for i = 2,tree.n_nodes do
		local col_i = {{},{i}}
		local parent_id = tree.parent_id[i]
		local input = nil
		local rule = self.rules[tree.rule_id[parent_id]]

		local temp = tree.outer[{{},{parent_id}}]
		local input =	(rule.Wop_U * (rule.Wop_V * temp))
						:add(torch.cmul(rule.Wop_r, temp))
		for j = 1, tree.n_children[parent_id] do
			local sister_id = tree.children_id[{j,parent_id}]
			if sister_id ~= i then
				local temp = tree.inner[{{},{sister_id}}]
				input	:add(rule.Wo_U[j] * (rule.Wo_V[j] * temp))
						:add(torch.cmul(rule.Wo_r[j], temp))
			else
				input:add(rule.bo[j])
			end
		end
		tree.outer[col_i]:copy(self.func(input))
	end
end

function IORNN:forward_word_prediction(tree, n_noise_cases)
	tree.pred_w = {}
	tree.cross_entropy = torch.zeros(tree.n_nodes)
	
	tree.noise_word_id = torch.LongTensor(n_noise_cases, tree.n_nodes)
	tree.word_score = torch.zeros(1, tree.n_nodes)
	tree.noise_word_score = torch.zeros(n_noise_cases, tree.n_nodes)
	tree.score = torch.zeros(n_noise_cases, tree.n_nodes)
	tree.rank_error = torch.zeros(n_noise_cases, tree.n_nodes)

	local n_leaves = 0
	for i = 2,tree.n_nodes do
		if tree.n_children[i] == 0 then
			n_leaves = n_leaves + 1
			local col_i = {{},{i}}
			local outer = tree.outer[col_i]
			local word_id = tree.word_id[i]
			tree.noise_word_id[{{},i}] = 6 -- torch.rand(n_noise_cases):mul(self.voca:size()):floor():add(1):long()

			-- compute score
			tree.word_score[{1,i}] = torch.cmul(self.Lt[{{},{word_id}}],outer):sum()
			tree.noise_word_score[col_i] = self.Lt:index(2,tree.noise_word_id[{{},i}]):t() * outer
			tree.rank_error[col_i] = tree.noise_word_score[col_i] - tree.word_score[{1,i}] + 1
			tree.rank_error[col_i][tree.rank_error[col_i]:lt(0)] = 0
		end
	end

	return tree.rank_error:sum(), n_leaves
end

--*********************** backpropagate *********************--
function IORNN:backpropagate_outside(tree, grad)
	if tree.gradZo == nil then
		tree.gradZo = torch.zeros(self.dim, tree.n_nodes)
	else
		tree.gradZo:fill(0)
	end

	for i = tree.n_nodes, 2, -1 do
		local col_i = {{},{i}}
		local outer = tree.outer[col_i]
		local gZo = torch.zeros(self.dim, 1)
		local rule = self.rules[tree.rule_id[i]]

		-- word prediction
		for j = 1,tree.rank_error:size(1) do
			local word_id = tree.word_id[i]
			if tree.rank_error[{j,i}] > 0 then
				local noise_word_id = tree.noise_word_id[{j,i}]
				grad.Lt[{{},{word_id}}]:add(-outer)
				grad.Lt[{{},{noise_word_id}}]:add(outer)
				gZo:add(self.Lt[{{},{noise_word_id}}] - self.Lt[{{},{word_id}}])
			end
		end

		local input = nil
		for j = 1, tree.n_children[i] do
			if input == nil then 
				input = tree.gradZo[{{},{tree.children_id[{j,i}]}}]:clone()
			else
				input:add(tree.gradZo[{{},{tree.children_id[{j,i}]}}])
			end
		end
		if input ~= nil then 
			gZo	:add(rule.Wop_V:t() * (rule.Wop_U:t() * input))
				:add(torch.cmul(rule.Wop_r, input))
		end
		gZo:cmul(self.funcPrime(outer))
		tree.gradZo[col_i]:copy(gZo)

		-- Wop, Wo, bo
		local parent_id = tree.parent_id[i]
		local grad_rule = grad.rules[tree.rule_id[parent_id]]
		local prule = self.rules[tree.rule_id[parent_id]]

		local temp = tree.outer[{{},{parent_id}}]
		grad_rule.Wop_U:add(gZo * (temp:t() * prule.Wop_V:t()))
		grad_rule.Wop_V:add(prule.Wop_U:t() * gZo * temp:t())
		grad_rule.Wop_r:add(torch.cmul(gZo,temp))

		for j = 1, tree.n_children[parent_id] do
			local sister_id = tree.children_id[{j,parent_id}]
			local temp = tree.inner[{{},{sister_id}}]

			if sister_id ~= i then
				grad_rule.Wo_U[j]:add(gZo * (temp:t() * prule.Wo_V[j]:t()))
				grad_rule.Wo_V[j]:add(prule.Wo_U[j]:t() * gZo * temp:t())
				grad_rule.Wo_r[j]:add(torch.cmul(gZo,temp))
			else
				grad_rule.bo[j]:add(gZo)
			end
		end
	end

	-- root
	local input = tree.gradZo[{{},{tree.children_id[{1,1}]}}]:clone()
	for j = 2, tree.n_children[1] do
		input:add(tree.gradZo[{{},{tree.children_id[{j,1}]}}])
	end
	local rule = self.rules[tree.rule_id[1]]
	local gZo = (rule.Wop_V:t() * (rule.Wop_U:t() * input))
				:add(torch.cmul(rule.Wop_r, input))
	grad.root_outer:add(gZo)
end

function IORNN:backpropagate_inside(tree, grad)
	if tree.gradZi == nil then
		tree.gradZi = torch.zeros(self.dim, tree.n_nodes)
	else
		tree.gradZi:fill(0)
	end

	for i = 1,tree.n_nodes do
		local col_i = {{},{i}}
		local gZi = torch.zeros(self.dim, 1)

		-- if not the root
		if tree.parent_id[i] > 0 then
			local parent_id = tree.parent_id[i]
			local rule = self.rules[tree.rule_id[parent_id]]
	
			local temp = tree.gradZi[{{},{parent_id}}]	
			for j = 1, tree.n_children[parent_id] do
				local sister_id = tree.children_id[{j,parent_id}]
				if sister_id == i then
					gZi	:add(rule.Wi_V[j]:t() * (rule.Wi_U[j]:t() * temp))
						:add(torch.cmul(rule.Wi_r[j], temp))
				else 
					local temp1 = tree.gradZo[{{},{sister_id}}]
					local k = tree.sibling_order[i]
					gZi	:add(rule.Wo_V[k]:t() * (rule.Wo_U[k]:t() * temp1))
						:add(torch.cmul(rule.Wo_r[k], temp1))
				end
			end
		end

		-- for internal node
		if tree.n_children[i] > 0 then	
			gZi:cmul(self.funcPrime(tree.inner[col_i]))
			tree.gradZi[col_i]:add(gZi)

			-- weight matrix for inner
			local rule = self.rules[tree.rule_id[i]]
			local grad_rule = grad.rules[tree.rule_id[i]]
			for j = 1, tree.n_children[i] do
				local child_id = tree.children_id[{j,i}]
				local temp = tree.inner[{{},{child_id}}]

				grad_rule.Wi_U[j]:add(gZi * (temp:t() * rule.Wi_V[j]:t()))
				grad_rule.Wi_V[j]:add(rule.Wi_U[j]:t() * gZi * temp:t())
				grad_rule.Wi_r[j]:add(torch.cmul(gZi,temp))
			end
			grad_rule.bi:add(gZi)

		else -- leaf
			tree.gradZi[col_i]:add(gZi)

			if self.update_L then
				grad.Lc[{{},{tree.word_id[i]}}]:add(gZi)
			end
		end
	end
end

function IORNN:computeCostAndGrad(treebank, config)
	local parse = config.parse or false

	p:start('compute cost and grad')	

if NPROCESS > 1 then
else
	local grad = self:create_grad()

	local cost = 0
	local nSample = 0

	p:start('process treebank') 
	for i, tree in ipairs(treebank) do
		-- forward
		self:forward_inside(tree)
		self:forward_outside(tree)
		local lcost,lnleaves = self:forward_word_prediction(tree, config.n_noise_words)
		cost = cost + lcost
		nSample = nSample + lnleaves

		-- backward
		self:backpropagate_outside(tree, grad)
		self:backpropagate_inside(tree, grad)
	end
	p:lap('process treebank') 

	p:start('compute grad')
	local wparams = self.params[{{1,-1-2*self.dim*self.voca:size()}}]
	local lparams = self.params[{{-2*self.dim*self.voca:size(),-1}}]
	cost = cost / nSample 	+ config.lambda/2 * torch.pow(wparams,2):sum() 
							+ config.lambda_L/2 * torch.pow(lparams,2):sum()

	local grad_wparams = grad.params[{{1,-1-2*self.dim*self.voca:size()}}]
	local grad_lparams = grad.params[{{-2*self.dim*self.voca:size(),-1}}]
	grad.params:div(nSample)
	grad_wparams:add(wparams * config.lambda)
	grad_lparams:add(lparams * config.lambda_L)
	p:lap('compute grad')

	p:lap('compute cost and grad') 
	p:printAll()

	return cost, grad, treebank
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
-- adagrad, adapt from optim.adagrad
function IORNN:adagrad(config, state)
	local config = config or {}
	local state = state or config
	local lr = config.learningRate or 1e-3
	local lrd = config.learningRateDecay or 0
	state.evalCounter = state.evalCounter or 0
	local nevals = state.evalCounter

	-- (1) evaluate f(x) and df/dx
	local cost,grad = opfunc(x)

	-- (3) learning rate decay (annealing)
	local clr = lr / (1 + nevals*lrd)
      
	-- (4) parameter update with single or individual learning rates
	if not state.paramVariance then
		state.paramVariance = torch.Tensor():typeAs(x):resizeAs(dfdx):zero()
		state.paramStd = torch.Tensor():typeAs(x):resizeAs(dfdx)
	end
	state.paramVariance:addcmul(1,dfdx,dfdx)
	torch.sqrt(state.paramStd,state.paramVariance)
	x:addcdiv(-clr, dfdx,state.paramStd:add(1e-10))

	-- (5) update evaluation counter
	state.evalCounter = state.evalCounter + 1

	-- return x*, f(x) before optimization
	return x,{fx}
end

function IORNN:train_with_adagrad(traintreebank, devtreebank, batchSize, 
									maxepoch, lambda, prefix,
									adagrad_config, adagrad_state, bag_of_subtrees)
	local nSample = #traintreebank
	
	local epoch = 1
	local j = 0

	print('===== epoch ' .. epoch .. '=====')

	while true do
		j = j + 1
		if j > nSample/batchSize then 
			self:save(prefix .. '_' .. epoch)
			j = 1 
			epoch = epoch + 1
			if epoch > maxepoch then break end
			print('===== epoch ' .. epoch .. '=====')
		end

		local subtreebank = {}
		for k = 1,batchSize do
			subtreebank[k] = traintreebank[k+(j-1)*batchSize]
		end
	
		local function func(M)
			-- extract data
			p:start("compute grad")
			cost, Grad = self:computeCostAndGrad(subtreebank, 
							{lambda = lambda.lambda, lambda_L=lambda.lambda_L, alpha = alpha, beta = beta})
			p:lap("compute grad")

			print('iter ' .. j .. ': ' .. cost) --io.flush()		
			return cost, Grad
		end

		p:start("optim")
		M,_ = optim.adagrad(func, self.params, adagrad_config, adagrad_state)
		
		p:lap("optim")
		p:printAll()

		collectgarbage()
	end

	return adagrad_config, adagrad_state
end

function IORNN:parse(treebank)
	for _,tree in ipairs(treebank) do 
		self:forward_inside(tree)
		self:forward_outside(tree)
		self:forward_word_prediction(tree)
	end
	return treebank
end

--********************************** test ******************************--
	torch.setnumthreads(1)

	local vocaDic = Dict:new(huang_template)
	vocaDic:load('../data/toy/words.lst')

	raw = {
		"(TOP (S (CC Yet) (NP (DT the) (NN act)) (VP (VBZ is) (ADVP (RB still)) (ADJP (JJ charming)) (ADVP (RB here))) (. .)))",
		"(TOP (S (NP (DT A) (NN screenplay)) (VP (VBZ is) (ADJP (ADVP (RBR more) (RB ingeniously)) (VBN constructed) (PP (IN than) (NP (`` ``) (NNP Memento) ('' ''))))) (. .)))",
		"(TOP (S (NP (DT The) (NN act) (NN screenplay)) (VP (VBZ is) (ADJP (JJR more) (PP (IN than) (ADVP (RB here))))) (. .)))"
	}

	treebank = {}
	for i = 1,#raw do
		local tree = Tree:create_from_string(raw[i])
		for j = 1,100 do
			treebank[#treebank+1] = tree
		end
	end

	ruleDic = Dict:new(cfg_template)
	ruleDic:load("../data/toy/cfg_simple_rule.txt")
--	ruleDic:load("grammar/cfg_simple_rule.txt")
		
	for i = 1,#treebank do
		treebank[i] = treebank[i]:to_torch_matrices(vocaDic, ruleDic)
	end

	local rules = {}
	for _,str in ipairs(ruleDic.id2word) do
		local comps = split_string(str, "[^ \t]+")
		local rule = {lhs = comps[1], rhs = {}}
		for i = 2,#comps do
			rule.rhs[i-1] = comps[i]
		end
		rules[#rules+1] = rule
	end

	local dim = 100
	local subdim = 5
	input = {	t_lookup = torch.randn(dim,vocaDic:size()), 
				c_lookup = torch.randn(dim,vocaDic:size()),
				subdim = subdim, 
				func = identity, funcPrime = identityPrime, 
				rules = rules,
				voca = vocaDic }
	net = IORNN:new(input)

	--print(net)	

	config = {lambda = 1e-4, lambda_L = 1e-7, n_noise_words = 1}
	net.update_L = true
	net:checkGradient(treebank, config)

