require 'tree'
require 'utils'
require 'dict'
require 'lexicon'

NPROCESS = 1

--**************** rerursive neural network class ******************--
sfIORNN = {}
sfIORNN_mt = {__index = sfIORNN}

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
function sfIORNN:new(input)

	local dim		= input.Lookup:size(1)
	local wrdDicLen	= input.Lookup:size(2)
	local lexicon	= input.lexicon
	local rules		= input.rules

	local net = {dim = dim, lex = lexicon}
	net.rules = {}
	net.wpred = {}

	local r = 0.1 --math.sqrt(6 / (dim + dim))
	local rw = 0.1 --math.sqrt(6 / (dim + 2*dim))

	-- create weight matrices for rules
	for i,rule in ipairs(rules) do
		net.rules[i] = {}

		net.rules[i].Wi = {}
		net.rules[i].Wo = {}
		net.rules[i].bo = {}

		for j = 1,#rule.rhs do
			net.rules[i].Wi[j] = uniform(dim, dim, -r, r)
			net.rules[i].Wo[j] = uniform(dim, dim, -r, r)
			net.rules[i].bo[j] = torch.zeros(dim, 1)
		end		

		net.rules[i].bi = torch.zeros(dim, 1)
		net.rules[i].Wop = uniform(dim, dim, -r, r)
	end

	-- word prediction
	net.Wc = uniform(lexicon.class:size(), dim, -r, r) -- for class
	net.bc = torch.zeros(lexicon.class:size(), 1)

	for i,wc in ipairs(lexicon.word_in_class) do -- for words in class
		net.wpred[i] = {	Ww = uniform(wc:size(), dim, -r, r),
							bw = torch.zeros(wc:size(), 1) }
	end

	-- wordembedding
	net.L = input.Lookup
	net.func = input.func
	net.funcPrime = input.funcPrime

	-- root outer
	net.root_outer = uniform(dim, 1, -1e-3, 1e-3)
	
	setmetatable(net, sfIORNN_mt)
	return net
end

-- save net into a bin file
function sfIORNN:save( filename , binary )
	local file = torch.DiskFile(filename, 'w')
	if binary == nil or binary then print(binary) file:binary() end
	file:writeObject(self)
	file:close()
end

-- create net from file
function sfIORNN:load( filename , binary, func, funcPrime )
	local file = torch.DiskFile(filename, 'r')
	if binary == nil or binary then file:binary() end
	local net = file:readObject()
	file:close()

	setmetatable(net, sfIORNN_mt)
	net.lex = Lex:setmetatable(net.lex)

	return net
end

-- fold parameters to a vector
function sfIORNN:fold( Model )
	local net = Model or self

	local Params = {
		net.root_outer,
		net.Wc, net.bc,
	}
	
	for _,rule in ipairs(net.rules) do
		for j = 1,#rule.Wi do
			Params[#Params+1] = rule.Wi[j]; 
			Params[#Params+1] = rule.Wo[j]; 
			Params[#Params+1] = rule.bo[j]
		end
		Params[#Params+1] = rule.bi
		Params[#Params+1] = rule.Wop
	end

	for _,wp in ipairs(net.wpred) do
		Params[#Params+1] = wp.Ww
		Params[#Params+1] = wp.bw
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
function sfIORNN:unfold(Theta)
	local Params = {
		self.root_outer,
		self.Wc, self.bc,
	}
	
	for _,rule in ipairs(self.rules) do
		for j = 1,#rule.Wi do
			Params[#Params+1] = rule.Wi[j]; 
			Params[#Params+1] = rule.Wo[j]; 
			Params[#Params+1] = rule.bo[j];
		end
		Params[#Params+1] = rule.bi
		Params[#Params+1] = rule.Wop
	end

	for _,wp in ipairs(self.wpred) do
		Params[#Params+1] = wp.Ww
		Params[#Params+1] = wp.bw
	end

	if self.update_L == true then
		Params[#Params+1] = self.L
	end

	local i = 1
	for _,P in ipairs(Params) do
		local nElem = P:nElement()
		P:copy(Theta[{{i,i+nElem-1}}])
		i = i + nElem
	end
end

--************************ forward **********************--
--input:
--	tree : compact tree, a list of nodes
--output:
--	Tree

function sfIORNN:forward_inside(tree)
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
			for j = 2,tree.n_children[i] do
				input:add(rule.Wi[j] * tree.inner[{{},{tree.children_id[{j,i}]}}])
			end
			input:add(rule.bi)
			tree.inner[col_i]:copy(self.func(input))
		end
	end
end

function sfIORNN:forward_outside(tree)	
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
				input:add(rule.Wo[j] * tree.inner[{{},{sister_id}}])
			else
				input:add(rule.bo[j])
			end
		end
		tree.outer[col_i]:copy(self.func(input))
	end
end

function sfIORNN:forward_word_prediction(tree)
	tree.pred_c = {}
	tree.pred_w = {}
	tree.cross_entropy = torch.zeros(tree.n_nodes)
	tree.class_id = torch.zeros(tree.n_nodes)
	tree.word_in_class_id = torch.zeros(tree.n_nodes)

	local n_leaves = 0
	for i = 2,tree.n_nodes do
		if tree.n_children[i] == 0 then
			n_leaves = n_leaves + 1
			local col_i = {{},{i}}
			local outer = tree.outer[col_i]
			local word_id = tree.word_id[i]
			local class_id = self.lex.class_of_word[word_id]
			local word_in_class_id = self.lex.word_in_class[class_id]:get_id(word_id)
			local wp = self.wpred[class_id]

			tree.pred_c[i] = safe_compute_softmax((self.Wc*outer):add(self.bc))
			tree.pred_w[i] = safe_compute_softmax((wp.Ww*outer):add(wp.bw))
			tree.cross_entropy[i] = -math.log(tree.pred_c[i][{class_id,1}] * 
											tree.pred_w[i][{word_in_class_id,1}])

			tree.class_id[i] = class_id
			tree.word_in_class_id[i] = word_in_class_id
		end
	end

	return tree.cross_entropy:sum(), n_leaves
end

--*********************** backpropagate *********************--
function sfIORNN:backpropagate_outside(tree, grad)
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
			-- for class prediction
			local gZc = tree.pred_c[i]:clone()
			gZc[{tree.class_id[i],{1}}]:add(-1)
			grad.Wc:add(gZc * outer:t())
			grad.bc:add(gZc)

			-- for word prediction (given class)
			local gZw = tree.pred_w[i]:clone()
			gZw[{tree.word_in_class_id[i],{1}}]:add(-1)
			local grad_wp = grad.wpred[tree.class_id[i]]
			grad_wp.Ww:add(gZw * outer:t())
			grad_wp.bw:add(gZw)

			gZo:add(self.Wc:t()*gZc)
			gZo:add(self.wpred[tree.class_id[i]].Ww:t()*gZw)
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
			gZo:add(rule.Wop:t() * input)
		end
		gZo:cmul(self.funcPrime(outer))
		tree.gradZo[col_i]:copy(gZo)

		-- Wop, Wo, bo
		local parent_id = tree.parent_id[i]
		local grad_rule = grad.rules[tree.rule_id[parent_id]]

		local gWop = gZo * tree.outer[{{},{parent_id}}]:t()
		grad_rule.Wop:add(gWop)

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
	local gZo = self.rules[tree.rule_id[1]].Wop:t() * input
	grad.root_outer:add(gZo)
end

function sfIORNN:backpropagate_inside(tree, grad)
	-- gradient over Z of softmax nodes
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
			
			for j = 1, tree.n_children[parent_id] do
				local sister_id = tree.children_id[{j,parent_id}]
				if sister_id == i then
					gZi:add(rule.Wi[j]:t() * tree.gradZi[{{},{parent_id}}])
				else 
					gZi:add(rule.Wo[tree.sibling_order[i]]:t() * tree.gradZo[{{},{sister_id}}])
				end
			end
		end

		-- for internal node
		if tree.n_children[i] > 0 then	
			gZi:cmul(self.funcPrime(tree.inner[col_i]))
			tree.gradZi[col_i]:add(gZi)

			-- weight matrices for inner
			local grad_rule = grad.rules[tree.rule_id[i]]
			for j = 1, tree.n_children[i] do
				local child_id = tree.children_id[{j,i}]
				grad_rule.Wi[j]:add(gZi * tree.inner[{{},{child_id}}]:t())
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

function sfIORNN:computeCostAndGrad(treebank, config)
	local parse = config.parse or false
	
if NPROCESS > 1 then
else
	-- create zero grad
	local grad = {}
	grad.rules = {}

	for i,rule in ipairs(self.rules) do
		grad.rules[i] = {}

		grad.rules[i].Wi = {}
		grad.rules[i].Wo = {}
		grad.rules[i].bo = {}

		for j = 1,#rule.Wi do
			grad.rules[i].Wi[j] = torch.zeros(rule.Wi[j]:size())
			grad.rules[i].Wo[j] = torch.zeros(rule.Wo[j]:size())
			grad.rules[i].bo[j] = torch.zeros(rule.bo[j]:size())
		end		

		grad.rules[i].bi = torch.zeros(rule.bi:size())
		grad.rules[i].Wop = torch.zeros(rule.Wop:size())
	end

	grad.Wc = torch.zeros(self.Wc:size())
	grad.bc = torch.zeros(self.bc:size())

	grad.wpred = {}
	for i,wp in ipairs(self.wpred) do
		grad.wpred[i] = { 	Ww = torch.zeros(wp.Ww:size()),
							bw = torch.zeros(wp.bw:size()) }
	end
	
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
function sfIORNN:checkGradient(treebank, config, bag_of_subtrees)
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
function sfIORNN:eval(treebank)
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

function sfIORNN:train_with_adagrad(traintreebank, devtreebank, batchSize, 
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

function sfIORNN:parse(treebank)
	for _,tree in ipairs(treebank) do 
		self:forward_inside(tree)
		self:forward_outside(tree)
		self:forward_word_prediction(tree)
	end
	return treebank
end

--[[*********************************** test ******************************--
	torch.setnumthreads(1)

	require 'lexicon'
	local lex = Lex:new(huang_template)
	lex:load('../data/toy/words.lst', '../data/toy/clusters.lst', '../data/toy/wc.txt')

	treebank = {
		"(TOP (S (CC Yet) (NP (DT the) (NN act)) (VP (VBZ is) (ADVP (RB still)) (ADJP (JJ charming)) (ADVP (RB here))) (. .)))",
		"(TOP (S (NP (DT A) (NN screenplay)) (VP (VBZ is) (ADJP (ADVP (RBR more) (RB ingeniously)) (VBN constructed) (PP (IN than) (NP (`` ``) (NNP Memento) ('' ''))))) (. .)))",
		"(TOP (S (NP (DT The) (NN act) (NN screenplay)) (VP (VBZ is) (ADJP (JJR more) (PP (IN than) (ADVP (RB here))))) (. .)))"
	}

	for i = 1,#treebank do
		treebank[i] = Tree:create_from_string(treebank[i])
	end

	require "dict"
	vocaDic = lex.voca

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
				func = identity, funcPrime = identityPrime, 
				rules = rules,
				lexicon = lex }
	net = sfIORNN:new(input)

	--print(net)	

	config = {lambda = 1e-4, lambda_L = 1e-7}
	net.update_L = true
	net:checkGradient(treebank, config, bag_of_subtrees)
]]
