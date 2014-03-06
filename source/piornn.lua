require 'tree'
require 'utils'
require 'dict'

NPROCESS = 1

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

	local dim		= input.Lookup:size(1)
	local voca		= input.voca
	local rules		= input.rules
	local subdim	= input.subdim

	local net = {dim = dim, subdim = subdim, voca = voca}
	net.rules = {}

	local r = 0.1
	-- create weight matrices for rules
	-- W = U x V + diag(r)
	for i,rule in ipairs(rules) do
		net.rules[i] = {}

		net.rules[i].Wi_U = {}
		net.rules[i].Wi_V = {}
		net.rules[i].Wi_r = {}

		net.rules[i].Wo_U = {}
		net.rules[i].Wo_V = {}
		net.rules[i].Wo_r = {}

		net.rules[i].bo = {}

		for j = 1,#rule.rhs do
			net.rules[i].Wi_U[j] = uniform(dim, subdim, -r, r)
			net.rules[i].Wi_V[j] = uniform(subdim, dim, -r, r)
			net.rules[i].Wi_r[j] = uniform(dim, 1, -r, r)

			net.rules[i].Wo_U[j] = uniform(dim, subdim, -r, r)
			net.rules[i].Wo_V[j] = uniform(subdim, dim, -r, r)
			net.rules[i].Wo_r[j] = uniform(dim, 1, -r, r)

			net.rules[i].bo[j] = torch.zeros(dim, 1)
		end		

		net.rules[i].bi = torch.zeros(dim, 1)
		net.rules[i].Wop_U = uniform(dim, subdim, -r, r)
		net.rules[i].Wop_V = uniform(subdim, dim, -r, r)
		net.rules[i].Wop_r = uniform(dim, 1, -r, r)
	end

	-- word prediction
	net.Ww = uniform(voca:size(), dim, -r, r)
	net.bw = torch.zeros(voca:size(), 1)

	-- wordembedding
	net.L = input.Lookup
	net.func = input.func
	net.funcPrime = input.funcPrime

	-- root outer
	net.root_outer = uniform(dim, 1, -1e-3, 1e-3)
	
	setmetatable(net, IORNN_mt)
	return net
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

-- fold parameters to a vector
function IORNN:fold( Model )
	local net = Model or self

	local Params = {
		net.root_outer,
		net.Ww, net.bw
	}
	
	for _,rule in ipairs(net.rules) do
		for j = 1,#rule.Wi_U do
			Params[#Params+1] = rule.Wi_U[j] 
			Params[#Params+1] = rule.Wi_V[j]
			Params[#Params+1] = rule.Wi_r[j]

			Params[#Params+1] = rule.Wo_U[j] 
			Params[#Params+1] = rule.Wo_V[j] 
			Params[#Params+1] = rule.Wo_r[j] 
	
			Params[#Params+1] = rule.bo[j]
		end
		Params[#Params+1] = rule.bi
	
		Params[#Params+1] = rule.Wop_U
		Params[#Params+1] = rule.Wop_V
		Params[#Params+1] = rule.Wop_r
	end

	if self.update_L == true then
		Params[#Params+1] = net.L
	end

	local length = 0
	for _,P in ipairs(Params) do
		length = length + P:nElement()
	end

	local Theta = torch.zeros(length)
	local i = 1
	for _,P in ipairs(Params) do
		local nElem = P:nElement()
		Theta[{{i,i+nElem-1}}] = P
		i = i + nElem
	end

	return Theta
end

-- unfold param-vector 
function IORNN:unfold(Theta)
	local net = self
	local Params = {
		net.root_outer,
		net.Ww, net.bw
	}
	
	for _,rule in ipairs(net.rules) do
		for j = 1,#rule.Wi_U do
			Params[#Params+1] = rule.Wi_U[j] 
			Params[#Params+1] = rule.Wi_V[j]
			Params[#Params+1] = rule.Wi_r[j]

			Params[#Params+1] = rule.Wo_U[j] 
			Params[#Params+1] = rule.Wo_V[j] 
			Params[#Params+1] = rule.Wo_r[j] 
	
			Params[#Params+1] = rule.bo[j]
		end
		Params[#Params+1] = rule.bi
	
		Params[#Params+1] = rule.Wop_U
		Params[#Params+1] = rule.Wop_V
		Params[#Params+1] = rule.Wop_r
	end

	if self.update_L == true then
		Params[#Params+1] = net.L
	end

	local i = 1
	for _,P in ipairs(Params) do
		local nElem = P:nElement()
		P:copy(Theta[{{i,i+nElem-1}}])
		i = i + nElem
	end
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

function IORNN:forward_word_prediction(tree)
	tree.pred_w = {}
	tree.cross_entropy = torch.zeros(tree.n_nodes)

	local n_leaves = 0
	for i = 2,tree.n_nodes do
		if tree.n_children[i] == 0 then
			n_leaves = n_leaves + 1
			local col_i = {{},{i}}
			local outer = tree.outer[col_i]
			local word_id = tree.word_id[i]
			tree.pred_w[i] = safe_compute_softmax((self.Ww*outer):add(self.bw))
			tree.cross_entropy[i] = -math.log(tree.pred_w[i][{word_id,1}])
		end
	end

	return tree.cross_entropy:sum(), n_leaves
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
		if tree.cross_entropy[i] > 0 then
			local gZw = tree.pred_w[i]:clone()
			gZw[{tree.word_id[i],{1}}]:add(-1)
			grad.Ww:add(gZw * outer:t())
			grad.bw:add(gZw)
			gZo:add(self.Ww:t()*gZw)
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
		grad_rule.Wop_r:add(temp)

		for j = 1, tree.n_children[parent_id] do
			local sister_id = tree.children_id[{j,parent_id}]
			local temp = tree.inner[{{},{sister_id}}]

			if sister_id ~= i then
				grad_rule.Wo_U[j]:add(gZo * (temp:t() * prule.Wo_V[j]:t()))
				grad_rule.Wo_V[j]:add(prule.Wo_U[j]:t() * gZo * temp:t())
				grad_rule.Wo_r[j]:add(temp:t())
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
				grad_rule.Wi_r[j]:add(temp)
			end
			grad_rule.bi:add(gZi)

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
	
if NPROCESS > 1 then
else
	-- create zero grad
	local grad = {}
	grad.rules = {}

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
			grad.rules[i].Wi_U[j] = torch.zeros(rule.Wi_U[j]:size())
			grad.rules[i].Wi_V[j] = torch.zeros(rule.Wi_V[j]:size())
			grad.rules[i].Wi_r[j] = torch.zeros(rule.Wi_r[j]:size())

			grad.rules[i].Wo_U[j] = torch.zeros(rule.Wo_U[j]:size())
			grad.rules[i].Wo_V[j] = torch.zeros(rule.Wo_V[j]:size())
			grad.rules[i].Wo_r[j] = torch.zeros(rule.Wo_r[j]:size())
			
			grad.rules[i].bo[j] = torch.zeros(rule.bo[j]:size())
		end		

		grad.rules[i].bi = torch.zeros(rule.bi:size())
		grad.rules[i].Wop_U = torch.zeros(rule.Wop_U:size())
		grad.rules[i].Wop_V = torch.zeros(rule.Wop_V:size())
		grad.rules[i].Wop_r = torch.zeros(rule.Wop_r:size())
	end

	grad.Ww = torch.zeros(self.Ww:size())
	grad.bw = torch.zeros(self.bw:size())
	
	if self.update_L then
		grad.L = torch.zeros(self.L:size())
	end
	grad.root_outer = torch.zeros(self.root_outer:size())

	local cost = 0
	local nSample = 0
	for i, tree in ipairs(treebank)  do
		-- forward
		self:forward_inside(tree)
		self:forward_outside(tree)
		local lcost,lnleaves = self:forward_word_prediction(tree)
		cost = cost + lcost
		nSample = nSample + lnleaves

		-- backward
		self:backpropagate_outside(tree, grad)
		self:backpropagate_inside(tree, grad)
	end

	local M = self:fold()
	grad = self:fold(grad)

	cost = cost / nSample + config.lambda/2 * torch.pow(M,2):sum()
	grad:div(nSample):add(M * config.lambda)

	p:start('lambda L')
	if self.update_L then
		cost = cost + (config.lambda_L - config.lambda)/2 * torch.pow(self.L,2):sum()
		grad[{{-self.L:nElement(),-1}}]:add(self.L * (config.lambda_L-config.lambda))
	end
	p:lap('lambda L')

	return cost, grad, treebank
end
end

-- check gradient
function IORNN:checkGradient(treebank, config, bag_of_subtrees)
	local epsilon = 1e-4

	local dim = self.dim
	local wrdDicLen = self.wrdDicLen
	local nCat = self.nCat

	local Theta = self:fold()
	local _, gradTheta = self:computeCostAndGrad(treebank, config, bag_of_subtrees)

	local n = Theta:nElement()
	print(n)
	local numGradTheta = torch.zeros(n)
	for i = 1,n do
		local index = {{i}}
		Theta[index]:add(epsilon)
		self:unfold(Theta)
		local costPlus,_ = self:computeCostAndGrad(treebank, config, bag_of_subtrees)
		
		Theta[index]:add(-2*epsilon)
		self:unfold(Theta)
		local costMinus,_ = self:computeCostAndGrad(treebank, config, bag_of_subtrees)
		Theta[index]:add(epsilon)
		self:unfold(Theta)

		numGradTheta[i] = (costPlus - costMinus) / (2*epsilon) 

		local diff = math.abs(numGradTheta[i] - gradTheta[i])
		print('diff ' .. i .. ' ' .. diff)
	end

	local diff = torch.norm(gradTheta - numGradTheta) 
					/ torch.norm(gradTheta + numGradTheta)
	print(diff)
	print("should be < 1e-9")
end


--[[***************************** eval **************************
function IORNN:eval(treebank)
	_, _, treebank = self:computeCostAndGrad(treebank, {parse=true})
	local total_all = 0
	local correct_all = 0

	local total_root = 0
	local correct_root = 0

	for i,tree in ipairs(treebank) do
		total_all = total_all + tree.n_nodes
		total_root = total_root + 1

		local m,_ = tree.cat_predict:max(1)
		pred = 	torch.eq(
					tree.cat_predict, 
					torch.repeatTensor(torch.reshape(m,1,tree.n_nodes), self.nCat,1))
				:double()
		correct_all = correct_all + torch.cmul(pred, tree.category):sum()
		correct_root = correct_root + 
						torch.cmul(pred[{{},{1}}], tree.category[{{},{1}}]):sum()
	end

	return correct_all / total_all, correct_root / total_root
end
]]

--**************************** training ************************--
require 'optim'
require 'xlua'
p = xlua.Profiler()

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
			self:unfold(M)

			-- extract data
			p:start("compute grad")
			cost, Grad = self:computeCostAndGrad(subtreebank, 
							{lambda = lambda.lambda, lambda_L=lambda.lambda_L, alpha = alpha, beta = beta}, 
							bag_of_subtrees)
			p:lap("compute grad")

			print('iter ' .. j .. ': ' .. cost) --io.flush()		
			return cost, Grad
		end

		p:start("optim")
		M,_ = optim.adagrad(func, self:fold(), adagrad_config, adagrad_state)
		self:unfold(M)
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

--*********************************** test ******************************--
	torch.setnumthreads(1)

	local vocaDic = Dict:new(huang_template)
	vocaDic:load('../data/toy/words.lst')

	treebank = {
		"(TOP (S (CC Yet) (NP (DT the) (NN act)) (VP (VBZ is) (ADVP (RB still)) (ADJP (JJ charming)) (ADVP (RB here))) (. .)))",
		"(TOP (S (NP (DT A) (NN screenplay)) (VP (VBZ is) (ADJP (ADVP (RBR more) (RB ingeniously)) (VBN constructed) (PP (IN than) (NP (`` ``) (NNP Memento) ('' ''))))) (. .)))",
		"(TOP (S (NP (DT The) (NN act) (NN screenplay)) (VP (VBZ is) (ADJP (JJR more) (PP (IN than) (ADVP (RB here))))) (. .)))"
	}

	for i = 1,#treebank do
		treebank[i] = Tree:create_from_string(treebank[i])
	end

	ruleDic = Dict:new(cfg_template)
	ruleDic:load("grammar/cfg_simple_rule.txt")
		
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

	input = {	Lookup = torch.randn(2,vocaDic:size()), 
				subdim = 1, 
				func = identity, funcPrime = identityPrime, 
				rules = rules,
				voca = vocaDic }
	net = IORNN:new(input)

	--print(net)	

	config = {lambda = 0*1e-4, lambda_L = 0*1e-7}
	net.update_L = false
	net:checkGradient(treebank, config, bag_of_subtrees)

