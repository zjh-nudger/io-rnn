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
	local temp = -tree.class_predict[tree.class_gold]:log()

	--local wid = tree.word_id:gt(0)
	--temp[wid] = 0
	tree.class_error = temp:sum()
	return tree.class_error, tree.n_nodes-- -wid:double():sum()
end

--*********************** backpropagate *********************--
function IORNN:backpropagate_outside(tree, grad)
	if tree.gradZo == nil then
		tree.gradZo = torch.zeros(self.dim, tree.n_nodes)
		tree.grado = torch.zeros(self.dim, tree.n_nodes)
		tree.gradi = torch.zeros(self.dim, tree.n_nodes)
	else
		tree.gradZo:fill(0)
	end

	-- for classification
	local gZc = tree.class_predict - tree.class_gold:double()
	--gZc[torch.repeatTensor(tree.word_id:gt(0):resize(1,gZc:size(2)),gZc:size(1),1)] = 0
	torch.mm(tree.grado, self.Wco:t(), gZc)
	torch.mm(tree.gradi, self.Wci:t(), gZc)
	grad.Wco:add(gZc * tree.outer:t())
	grad.Wci:add(gZc * tree.inner:t())
	grad.bc:add(gZc:sum(2))

	-- iterate over all nodes
	for i = tree.n_nodes, 2, -1 do
		local col_i = {{},{i}}
		local outer = tree.outer[col_i]
		local rule = self.rules[tree.rule_id[i]]
		local gZo = tree.grado[col_i]

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

function IORNN:computeCostAndGrad(treebank, config, grad)
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
	for i, tree in ipairs(treebank) do
		-- forward
		self:forward_inside(tree)
		self:forward_outside(tree)
		local lcost,ncases = self:forward_predict_class(tree)
		cost = cost + lcost
		nSample = nSample + ncases

		-- backward
		self:backpropagate_outside(tree, grad)
		self:backpropagate_inside(tree, grad)

		-- extract target word id for updating
		for i = 1,tree.n_nodes do
			if tree.word_id[i] > 0 then
				tword_id[tree.word_id[i]] = 1
			end
		end
	end
	p:lap('process treebank') 

	p:start('compute grad')
	local wparams = self.params[{{1,-1-self.dim*self.voca:size()}}]
	local grad_wparams = grad.params[{{1,-1-self.dim*self.voca:size()}}]
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
	local voca_lr = config.voca_learningRate or 1e-3

	local lrd = config.learningRateDecay or 0
	state.evalCounter = state.evalCounter or 0
	local nevals = state.evalCounter

	-- (1) evaluate f(x) and df/dx
	local cost, grad, _, tword_id = func()

	-- (3) learning rate decay (annealing)
	local weight_clr	= weight_lr / (1 + nevals*lrd)
    local voca_clr		= voca_lr / (1 + nevals*lrd)
  
	-- (4) parameter update with single or individual learning rates
	if not state.paramVariance then
		state.paramVariance = self:create_grad()
		state.paramStd = self:create_grad()
	end

	-- for weights
	local wparamindex = {{1,-1-self.dim*self.voca:size()}}
	state.paramVariance.params[wparamindex]:addcmul(1,grad.params[wparamindex],grad.params[wparamindex])
	torch.sqrt(state.paramStd.params[wparamindex],state.paramVariance.params[wparamindex])
	self.params[wparamindex]:addcdiv(-weight_clr, grad.params[wparamindex],state.paramStd.params[wparamindex]:add(1e-10))

	-- for word embeddings
	for wid,_ in pairs(tword_id) do
		local col_i = {{},{wid}}
		--state.paramVariance.L[col_i]:addcmul(1,grad.L[col_i],grad.L[col_i])
		--torch.sqrt(state.paramStd.L[col_i],state.paramVariance.L[col_i])
		--self.L[col_i]:addcdiv(-voca_clr, grad.L[col_i],state.paramStd.L[col_i]:add(1e-10))
		self.L[col_i]:add(-voca_clr, grad.L[col_i])  -- don't use adagrad for word embeddings
	end

	--print(state.paramStd.params:max())
	--print(state.paramStd.params[state.paramStd.params:gt(1e-8)]:min())

	-- (5) update evaluation counter
	state.evalCounter = state.evalCounter + 1
end

function IORNN:train_with_adagrad(traintreebank, devtreebank, batchSize, 
									maxepoch, lambda, prefix,
									adagrad_config, adagrad_state, bag_of_subtrees)
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

			self:eval(devtreebank,  '../data/SRL/conll05st-release/devel/props/devel.24.props',
									--'../data/SRL/toy/dev-props',
									'../data/SRL/conll05st-release/srlconll-1.1/bin/srl-eval.pl')

			if epoch > maxepoch then break end
			print('===== epoch ' .. epoch .. '=====')
		end

		local subtreebank = {}
		for k = 1,batchSize do
			subtreebank[k] = traintreebank[k+(j-1)*batchSize]
		end
	
		local function func()
			cost, grad, subtreebank, tword_id  = self:computeCostAndGrad(subtreebank, 
							{lambda = lambda.lambda, lambda_L=lambda.lambda_L}, grad)

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

function IORNN:recompute_class_predict(tree, alpha)
	local alpha = alpha or 0.5
	tree.class_predict:log()

	for i = 1, tree.n_nodes do
		local temp = torch.zeros(self.class:size(),1)
		for j = 1,tree.n_children[i] do
			temp:add(tree.class_predict[{{},{tree.children_id[{j,i}]}}])
		end
		tree.class_predict[{{},{i}}]:mul(alpha):add(temp:mul(1-alpha))
	end
end

function IORNN:label(tree, node_id, label)
	--local _,cid = tree.class_predict[{{},node_id}]:max(1)
	--cid = cid[1]
	--[[	
	local scores = torch.log(tree.class_predict[{{},node_id}])
	for j = 1,tree.n_children[node_id] do
		local child_id = tree.children_id[{j,node_id}]
		scores:add(torch.log(tree.class_predict[{{},child_id}]))
	end
	local _,cid = scores:max(1)
	cid = cid[1]
	]]

	local _,cid = tree.class_predict[{{},node_id}]:max(1)
	cid = cid[1]
	
	if cid == self.class:get_id('NULL') then
		for j = 1,tree.n_children[node_id] do
			self:label(tree, tree.children_id[{j,node_id}], label)
		end
	else

		local cover = tree.cover[{{},node_id}]
		if cover[1] == cover[2] then
			label[cover[1]] = '('..self.class.id2word[cid]..'*)'
		else
			label[cover[1]] = '('..self.class.id2word[cid]..'*'
			label[cover[2]] = '*)'
		end
	end
	return label
end

function IORNN:predict_srl(treebank, filename)
	for _,tree in ipairs(treebank) do 
		if tree.target_verb:max() > 0 then
			self:forward_inside(tree)
			self:forward_outside(tree)
			self:forward_predict_class(tree)
			self:recompute_class_predict(tree)
		end
	end

	local ret = {}
	local srls = nil
	local prev_tree = nil
	for _,tree in ipairs(treebank) do
		-- check if this is a new sentence
		if prev_tree == nil or prev_tree.n_nodes ~= tree.n_nodes or torch.ne(prev_tree.word_id,tree.word_id):double():sum() > 0 then
			if srls ~= nil then
				ret[#ret+1] = srls
			end
			srls = {verb = {}, role = {}}
			for i = 1, tree.cover[{2,1}] do
				srls.verb[i] = '-'
			end
		end
		prev_tree = tree
		
		-- marking target verb
		local m,vid = tree.target_verb:max(1)
		if m[1] > 0 then
			vid = vid[1]
			srls.verb[tree.cover[{1,vid}]] = self.voca.id2word[tree.word_id[tree.children_id[{1,vid}]]]

			-- read role
			local label = {}
			for i = 1,tree.cover[{2,1}] do
				label[i] = '*'
			end
			self:label(tree, 1, label)
			
			-- merge label 
			local cur_l = '*'
			for i,l in ipairs(label) do
				local h = l:sub(1,1)
				local e = l:sub(l:len())
				if h == '(' then
					cur_l = split_string(l, '[^(*)]+')[1]
					label[i] = cur_l			
				elseif h == '*' then
					label[i] = cur_l
				end
				if e == ')' then 
					cur_l = '*'
				end
			end

			cur_l = '*'
			for i,l in ipairs(label) do
				if l == cur_l then
					label[i] = '*'
				else
					if l ~= '*' then
						if cur_l ~= '*' then
							label[i-1] = label[i-1] .. ')'
						end
						cur_l = l
						label[i] = '('..label[i]..'*'
					else
						if cur_l ~= '*' then
							label[i-1] = label[i-1] .. ')'
						end
						cur_l = l
					end
				end
				if i == #label and cur_l ~= '*' then
					label[i] = label[i] .. ')'
				end
			end
			
			srls.role[#srls.role+1] = label
		end
	end
	ret[#ret+1] = srls

	-- print to file
	if filename ~= nil then
		local f = io.open(filename, 'w')
		for _,srls in ipairs(ret) do
			for i = 1, #srls.verb do
				local str = '' --srls.verb[i]
				for _,r in ipairs(srls.role) do
					str = str .. '\t' .. r[i]
				end
				f:write(str .. '\n')
			end
			f:write('\n')
		end
		f:close()
	end

	return ret
end

function IORNN:eval(treebank, gold_path, eval_prog_path)
	local predicts = self:predict_srl(treebank, '/tmp/predicts.txt')
	--print(predicts)
	os.execute('cat ' .. gold_path .. " | awk '{print $1}' > /tmp/verbs.txt")
	os.execute('paste /tmp/verbs.txt /tmp/predicts.txt > predicts.txt')
	os.execute(eval_prog_path .. ' ' .. gold_path .. ' ' .. 'predicts.txt')
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
	for line in io.lines('../data/SRL/toy/train-set') do
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

	config = {lambda = 1e-4, lambda_L = 1e-7}
	net.update_L = true
	net:checkGradient(treebank, config)
]]
