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

	local dim		= input.lookup:size(1)
	local voca		= input.voca
	local class		= input.class
	local rules		= input.rules

	local net = {dim = dim, voca = voca, class = class}
	net.func = input.func
	net.funcPrime = input.funcPrime
	setmetatable(net, IORNN_mt)

	net:init_params(input)
	return net
end

function IORNN:init_params(input)
	local net	= self
	local dim	= net.dim
	local voca	= net.voca
	local class	= net.class

	net.rules = {}

	-- create params
	local n_params = 0
	n_params = n_params + dim	-- outer at root
	for i,rule in ipairs(input.rules) do
		for j = 1,#rule.rhs do
			n_params = n_params + 2*(dim * dim) + dim
		end		
		n_params = n_params + dim*dim + dim
	end
	n_params = n_params + dim * voca:size()	-- target & conditional embeddings
	n_params = n_params + 2*class:size()*dim + class:size() + dim	-- for classification
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

		net.rules[i].Wi = {}
		net.rules[i].Wo = {}
		net.rules[i].bo = {}

		for j = 1,#rule.rhs do
			net.rules[i].Wi[j] = net.params[{{index,index+dim*dim-1}}]:resize(dim,dim):copy(uniform(dim, dim, -r, r))
			index = index + dim*dim
			net.rules[i].Wo[j] = net.params[{{index,index+dim*dim-1}}]:resize(dim,dim):copy(uniform(dim, dim, -r, r))
			index = index + dim*dim
			net.rules[i].bo[j] = net.params[{{index,index+dim-1}}]:resize(dim,1):copy(torch.zeros(dim, 1))
			index = index + dim
		end		

		net.rules[i].bi = net.params[{{index,index+dim-1}}]:resize(dim,1)
		index = index + dim
		net.rules[i].Wop = net.params[{{index,index+dim*dim-1}}]:resize(dim,dim):copy(uniform(dim, dim, -r, r))
		index = index + dim*dim
	end

	-- for classification
	net.Wco = net.params[{{index,index+class:size()*dim-1}}]:resize(class:size(),dim):copy(uniform(class:size(),dim, -r, r))
	index = index + class:size()*dim
	net.Wci = net.params[{{index,index+class:size()*dim-1}}]:resize(class:size(),dim):copy(uniform(class:size(),dim, -r, r))
	index = index + class:size()*dim
	net.bc = net.params[{{index,index+class:size()-1}}]:resize(class:size(),1)
	index = index + class:size()

	-- for marking target verb
	net.bv = net.params[{{index,index+dim-1}}]:resize(dim,1):copy(uniform(dim,1,-r,r))
	index = index + dim

	-- target/conditional embeddings (always always always at the end of params)
	net.L = net.params[{{index,index+voca:size()*dim-1}}]:resize(dim,voca:size()):copy(input.lookup)		-- word embeddings 
	index = index + voca:size()*dim

end

function IORNN:create_grad()
	local grad = {}
	local dim = self.dim
	local voca = self.voca
	local class = self.class

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
		grad.rules[i].Wi = {}
		grad.rules[i].Wo = {}
		grad.rules[i].bo = {}

		for j = 1,#rule.Wi do
			grad.rules[i].Wi[j] = grad.params[{{index,index+dim*dim-1}}]:resize(dim,dim)
			index = index + dim*dim
			grad.rules[i].Wo[j] = grad.params[{{index,index+dim*dim-1}}]:resize(dim,dim)
			index = index + dim*dim
			grad.rules[i].bo[j] = grad.params[{{index,index+dim-1}}]:resize(dim,1)
			index = index + dim
		end		

		grad.rules[i].bi = grad.params[{{index,index+dim-1}}]:resize(dim,1)
		index = index + dim
		grad.rules[i].Wop = grad.params[{{index,index+dim*dim-1}}]:resize(dim,dim)
		index = index + dim*dim
	end

	-- for classification
	grad.Wco = grad.params[{{index,index+class:size()*dim-1}}]:resize(class:size(),dim)
	index = index + class:size()*dim
	grad.Wci = grad.params[{{index,index+class:size()*dim-1}}]:resize(class:size(),dim)
	index = index + class:size()*dim
	grad.bc = grad.params[{{index,index+class:size()-1}}]:resize(class:size(),1)
	index = index + class:size()

	-- for marking target verb
	grad.bv = grad.params[{{index,index+dim-1}}]:resize(dim,1)
	index = index + dim

	-- target/conditional embeddings
	grad.L = grad.params[{{index,index+voca:size()*dim-1}}]:resize(dim,voca:size())		-- word embeddings
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
	setmetatable(net.class, Dict_mt)

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
			tree.inner[col_i]:copy(self.L[{{},{tree.word_id[i]}}])

		-- for internal nodes
		else
			local rule = self.rules[tree.rule_id[i]]
			if (#rule.Wi ~= tree.n_children[i]) then 
				error("rules not match")
			end

			local input = rule.Wi[1] * tree.inner[{{},{tree.children_id[{1,i}]}}]
			-- marking target verb
			if tree.target_verb[i] == 1 then
				input:add(self.bv)
			end
							
			for j = 2,tree.n_children[i] do
				local temp = tree.inner[{{},{tree.children_id[{j,i}]}}]
				input:add(rule.Wi[j] * tree.inner[{{},{tree.children_id[{j,i}]}}])
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

		local input = rule.Wop * tree.outer[{{},{parent_id}}]

		for j = 1, tree.n_children[parent_id] do
			local sister_id = tree.children_id[{j,parent_id}]
			if sister_id ~= i then
				local temp = tree.inner[{{},{sister_id}}]
				input:add(rule.Wo[j] * tree.inner[{{},{sister_id}}])
			else
				input:add(rule.bo[j])
			end
		end
		tree.outer[col_i]:copy(self.func(input))

	end
end

function IORNN:forward_predict_class(tree)
	tree.class_score = (self.Wco * tree.outer):add(self.Wci * tree.inner)
					:add(torch.repeatTensor(self.bc, 1, tree.n_nodes))
	tree.class_predict = safe_compute_softmax(tree.class_score)
	tree.class_error = -tree.class_predict[tree.class_gold]:log():sum()
	return tree.class_error
end

--*********************** backpropagate *********************--
function IORNN:backpropagate_outside(tree, grad)
	if tree.gradZo == nil then
		tree.gradZo = torch.zeros(self.dim, tree.n_nodes)
	else
		tree.gradZo:fill(0)
	end

	-- for classification
	local gZc = tree.class_predict - tree.class_gold:double()
	tree.grado = self.Wco:t() * gZc
	tree.gradi = self.Wci:t() * gZc
	grad.Wco:add(gZc * tree.outer:t())
	grad.Wci:add(gZc * tree.inner:t())
	grad.bc:add(gZc:sum(2))

	-- iterate over all nodes
	for i = tree.n_nodes, 2, -1 do
		local col_i = {{},{i}}
		local outer = tree.outer[col_i]
		local gZo = tree.grado[col_i]
		local rule = self.rules[tree.rule_id[i]]

		local input = nil
		for j = 1, tree.n_children[i] do
			if input == nil then 
				input = tree.gradZo[{{},{tree.children_id[{j,i}]}}]:clone()
			else
				input:add(tree.gradZo[{{},{tree.children_id[{j,i}]}}])
			end
		end
		if input ~= nil then 
			gZo:add(rule.Wop:t() * input)
		end
		gZo:cmul(self.funcPrime(outer))
		tree.gradZo[col_i]:copy(gZo)

		-- Wop, Wo, bo
		local parent_id = tree.parent_id[i]
		local grad_rule = grad.rules[tree.rule_id[parent_id]]

		local temp = tree.outer[{{},{parent_id}}]
		grad_rule.Wop:add(gZo * tree.outer[{{},{parent_id}}]:t())

		for j = 1, tree.n_children[parent_id] do
			local sister_id = tree.children_id[{j,parent_id}]

			if sister_id ~= i then
				grad_rule.Wo[j]:add(gZo * tree.inner[{{},{sister_id}}]:t())
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
	grad.root_outer:add(rule.Wop:t() * input):add(tree.grado[{{},{1}}])
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

		-- if not the root
		if tree.parent_id[i] > 0 then
			local parent_id = tree.parent_id[i]
			local rule = self.rules[tree.rule_id[parent_id]]
	
			local temp = tree.gradZi[{{},{parent_id}}]	
			for j = 1, tree.n_children[parent_id] do
				local sister_id = tree.children_id[{j,parent_id}]
				if sister_id == i then
					gZi	:add(rule.Wi[j]:t() * tree.gradZi[{{},{parent_id}}])
				else 
					local temp1 = tree.gradZo[{{},{sister_id}}]
					local k = tree.sibling_order[i]
					gZi	:add(rule.Wo[k]:t() * tree.gradZo[{{},{sister_id}}])
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

				grad_rule.Wi[j]:add(gZi * tree.inner[{{},{child_id}}]:t())
			end
			grad_rule.bi:add(gZi)

			-- for target verb
			if tree.target_verb[i] == 1 then
				grad.bv:add(gZi)
			end

		else -- leaf
			tree.gradZi[col_i]:add(gZi)

			if self.update_L then
				grad.L[{{},{tree.word_id[i]}}]:add(gZi)
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
		local lcost = self:forward_predict_class(tree)
		cost = cost + lcost
		nSample = nSample + tree.n_nodes

		-- backward
		self:backpropagate_outside(tree, grad)
		self:backpropagate_inside(tree, grad)
	end
	p:lap('process treebank') 

	p:start('compute grad')
	local wparams = self.params[{{1,-1-self.dim*self.voca:size()}}]
	local lparams = self.params[{{-self.dim*self.voca:size(),-1}}]
	cost = cost / nSample 	+ config.lambda/2 * torch.pow(wparams,2):sum() 
							+ config.lambda_L/2 * torch.pow(lparams,2):sum()

	local grad_wparams = grad.params[{{1,-1-self.dim*self.voca:size()}}]
	local grad_lparams = grad.params[{{-self.dim*self.voca:size(),-1}}]
	grad.params:div(nSample)
	grad_wparams:add(wparams * config.lambda)
	grad_lparams:add(lparams * config.lambda_L)
	p:lap('compute grad')

	p:lap('compute cost and grad') 
	--p:printAll()

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

--[[********************************** test ******************************--
	torch.setnumthreads(1)

	local vocaDic = Dict:new()
	vocaDic:load('../data/SRL/toy/words.lst')
 
	local ruleDic = Dict:new(cfg_template)
	ruleDic:load("../data/toy/cfg_simple_rule.txt")
	ruleDic.grammar = 'CFG'

	local classDic = Dict:new()
	classDic:load('../data/SRL/toy/class.lst')

	local treebank = {}
	local tokens = {}
	for line in io.lines('../data/SRL/toy/train.txt') do
		if line ~= '' then
			tokens[#tokens+1] = split_string(line, '[^ ]+')
		else 
			local tree,srls = Tree:create_CoNLL2005_SRL(tokens)
			tree = tree:to_torch_matrices(vocaDic, ruleDic)
			for _,srl in ipairs(srls) do
				local t = Tree:copy_torch_matrix_tree(tree)
				t = Tree:add_srl_torch_matrix_tree(t, srl, classDic)
				treebank[#treebank+1] = t
			end
		end
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

	local dim = 2
	input = {	lookup = torch.randn(dim,vocaDic:size()), 
				func = identity, funcPrime = identityPrime, 
				rules = rules,
				voca = vocaDic,
				class = classDic }
	net = IORNN:new(input)

	--print(net)	

	config = {lambda = 1e-4, lambda_L = 1e-7, n_noise_words = 1}
	net.update_L = true
	net:checkGradient(treebank, config)
]]
