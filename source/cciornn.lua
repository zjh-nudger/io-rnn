require 'tree'
require 'utils'

NPROCESS = 1

--**************** rerursive neural network class ******************--
CCIORNN = {}
CCIORNN_mt = {__index = CCIORNN}

--****************** needed functions ******************--
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
-- input : 
-- 	logiX : logisitic(X)
function logisticPrime(logiX)
	return torch.cmul(-logiX + 1, logiX)
end

-- tanh function 
-- in range [-1,1]
function tanh( X )
	return torch.tanh(X)
end

function tanhPrime(tanhX)
	return -torch.pow(tanhX,2)+1
end

--************************* construction ********************--
-- create a new recursive autor encoder with a given structure
-- input:
-- 		struct = { dimension, nCategories, Lookup }
function CCIORNN:new(struct, rules)

	local dim = struct.Lookup:size(1)
	local wrdDicLen = struct.Lookup:size(2)
	local nCat = struct.nCategory

	local net = {dim = dim, wrdDicLen = wrdDicLen, nCat = nCat}
	net.rules = {}
	local mul = 0.1

	-- create weight matrices for rules
	for i,rule in ipairs(rules) do
		net.rules[i] = {}

		net.rules[i].Wi = {}
		net.rules[i].Wo = {}

		for j = 1,#rule.rhs do
			net.rules[i].Wi[j] = uniform(dim, dim, -1, 1):mul(mul)
			net.rules[i].Wo[j] = uniform(dim, dim, -1, 1):mul(mul)
			net.rules[i].bo[j] = uniform(dim, 1, -1, 1):mul(0)
		end		

		net.rules[i].bi = uniform(dim, 1, -1, 1):mul(0)
		net.rules[i].Wop = uniform(dim, dim, -1, 1):mul(mul)	-- outer.parent
	end

	-- word/phrase ranking
	net.Wwi = uniform(2*dim, dim, -1, 1):mul(mul)	-- for combining inner, outer meanings
	net.Wwo = uniform(2*dim, dim, -1, 1):mul(mul)
	net.bw = uniform(2*dim, 1, -1, 1):mul(0)
	net.Ws = uniform(1, 2*dim, -1, 1):mul(mul)	-- for scoring

	-- classification
	net.WCat = uniform(nCat, dim, -1, 1):mul(mul)
	net.bCat = uniform(nCat, 1, -1, 1):mul(0) 

	-- wordembedding
	net.L = struct.Lookup --torch.randn(struct.Lookup:size()):mul(0.0001)
	net.func = struct.func
	net.funcPrime = struct.funcPrime

	-- root outer
	net.root_outer = uniform(dim, 1, -1, 1):mul(0.0001)
	
	setmetatable(net, CCIORNN_mt)
	return net
end

-- save net into a bin file
function CCIORNN:save( filename , binary )
	local file = torch.DiskFile(filename, 'w')
	if binary == nil or binary then print(binary) file:binary() end
	file:writeObject(self)
	file:close()
end

-- create net from file
function CCIORNN:load( filename , binary, func, funcPrime )
	local file = torch.DiskFile(filename, 'r')
	if binary == nil or binary then file:binary() end
	local net = file:readObject()
	setmetatable(net, CCIORNN_mt)
	file:close()

	net.func = func or tanh
	net.funcPrime = funcPrime or tanhPrime
	return net
end

-- fold parameters to a vector
function CCIORNN:fold( Model )
	local net = Model or self

	local Params = {
		net.WCat, net.bCat,
		net.Wwi, net.Wwo, net.bw, net.Ws,
		net.root_outer
	}
	
	for i = 1,#net.rules do
		for j = 1,#net.rules[i].Wi do
			Params[#Params+1] = net.rules[i].Wi[j]; 
			Params[#Params+1] = net.rules[i].Wo[j]; 
			Params[#Params+1] = net.rules[i].bo[j]
		end
		Params[#Params+1] = net.rules[i].bi
		Params[#Params+1] = net.rules[i].Wop
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
function CCIORNN:unfold(Theta)
	local Params = {
		net.WCat, net.bCat,
		net.Wwi, net.Wwo, net.bw, net.Ws,
		net.root_outer
	}
	
	for i = 1,#net.rules do
		for j = 1,#net.rules[i].Wi do
			Params[#Params+1] = net.rules[i].Wi[j]; 
			Params[#Params+1] = net.rules[i].Wo[j]; 
			Params[#Params+1] = net.rules[i].bo[j];
		end
		Params[#Params+1] = net.rules[i].bi
		Params[#Params+1] = net.rules[i].Wop
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
--input:
--	tree : compact tree, a list of nodes
--output:
--	Tree

function CCIORNN:forward_inside(tree)
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

function CCIORNN:forward_outside(tree, bag_of_subtrees)
	-- for substitued subtrees
	tree.stt_id = -torch.linspace(1,tree.n_nodes,tree.n_nodes) + tree.n_nodes+1
	--tree.stt_id = torch.rand(tree.n_nodes):mul(#bag_of_subtrees):add(1):floor()
	
	if tree.outer == nil then 
		tree.outer = torch.Tensor(self.dim, tree.n_nodes)
		tree.stt_score = torch.zeros(tree.n_nodes)
		tree.stt_io = torch.zeros(self.Ws:size(2), tree.n_nodes)	-- combination of inner and outer meanings

		-- for gold standard subtrees
		tree.gold_score = torch.zeros(tree.n_nodes)
		tree.gold_io = torch.zeros(self.Ws:size(2), tree.n_nodes)
		tree.stt_error = torch.zeros(tree.n_nodes)
	else
		tree.outer:fill(0)
		tree.stt_score:fill(0)
		tree.stt_io:fill(0)
		tree.gold_score:fill(0)
		tree.gold_io:fill(0)
		tree.stt_error:fill(0)
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
			local child_id = tree.children_id[{j,parent_id}]
			if child_id ~= i then
				input:add(rule.Wo[j] * tree.inner[{{},{child_id}}])
			else
				input:add(rule.bo[j])
			end
		end
		tree.outer[col_i]:copy(self.func(input))

		-- compute stt error / the criterion could be the sizes of subtrees (e.g. containing less than 4 words)
		local len = tree.cover[{2,i}] - tree.cover[{1,i}] + 1
		if #bag_of_subtrees > 0 and len <= bag_of_subtrees.max_phrase_len then
			-- compute gold score
			tree.gold_io[col_i]:copy(self.func(	(self.Wwo * tree.outer[col_i])
											:add(self.Wwi * tree.inner[col_i]):add(self.bw)))
			tree.gold_score[i] = self.Ws * tree.gold_io[col_i]

			-- compute stt score
			local stt_subtree = bag_of_subtrees[tree.stt_id[i]]
			self:forward_inside(stt_subtree)
			tree.stt_io[col_i]:copy(self.func((self.Wwo * tree.outer[col_i])
										:add(self.Wwi * stt_subtree.inner[{{},{1}}]):add(self.bw)))
			tree.stt_score[i] = self.Ws * tree.stt_io[col_i]

			-- error
			tree.stt_error[i] = math.max(0, 1 - tree.gold_score[i] + tree.stt_score[i])
		else
			tree.stt_id[i] = 0
		end
	end
end

function CCIORNN:forward(tree, bag_of_subtrees)
	-- inside 
	self:forward_inside(tree)

	--[[ compute classification error
	tree.cat_predict = safe_compute_softmax(
						(self.WCat*tree.inner)
						:add(torch.repeatTensor(self.bCat, 1, tree.n_nodes)))
	tree.cat_error = (-torch.cmul(tree.category, torch.log(tree.cat_predict))):sum(1)
]]

	-- outside
	self:forward_outside(tree, bag_of_subtrees)
end

--*********************** backpropagate *********************--
-- only for one tree/sentence
-- input:
-- 	tree : result of the parse function
-- output:

function CCIORNN:backpropagate_outside(tree, grad, bag_of_subtrees) --[[, alpha, beta)]]
	local alpha = 0; beta = 1

	if tree.gradZo == nil then
		tree.gradZo = torch.zeros(self.dim, tree.n_nodes)
		tree.grad_i = torch.zeros(self.dim, tree.n_nodes)
	else
		tree.gradZo:fill(0)
		tree.grad_i:fill(0)
	end

	for i = 1,tree.n_nodes do
		if tree.stt_id[i] > 0 then
			local stt_subtree = bag_of_subtrees[tree.stt_id[i]]
			if stt_subtree.grad_i == nil then
				stt_subtree.grad_i = torch.zeros(self.dim, stt_subtree.n_nodes)
				stt_subtree.gradZo = torch.zeros(self.dim, stt_subtree.n_nodes)
			else
				stt_subtree.grad_i:fill(0)
				stt_subtree.gradZo:fill(0)
			end	
		end
	end

	for i = tree.n_nodes, 2, -1 do
		local col_i = {{},{i}}
		local gZo = torch.zeros(self.dim, 1)
		local rule = self.rules[tree.rule_id[i]]

		-- subtree ranking error
		if tree.stt_error[i] > 0 then
			local stt_subtree = bag_of_subtrees[tree.stt_id[i]]
			local gold_io_prime = self.funcPrime(tree.gold_io[col_i])
			local stt_io_prime = self.funcPrime(tree.stt_io[col_i])

			-- Ws
			local gWs = (tree.stt_io[col_i] - tree.gold_io[col_i]):t()
			grad.Ws:add(gWs * beta)

			-- Wwo, Wwi, bw
			local gbw = (stt_io_prime - gold_io_prime):cmul(self.Ws:t())
			grad.bw:add(gbw * beta)
			local gWwo = gbw * tree.outer[col_i]:t()
			grad.Wwo:add(gWwo * beta)

			local gWwi = torch.cmul(self.Ws:t(), stt_io_prime) * stt_subtree.inner[{{},{1}}]:t()
						- torch.cmul(self.Ws:t(), gold_io_prime) * tree.inner[col_i]:t()
			grad.Wwi:add(gWwi * beta)
	
			-- gradZo before multiplied by fprime
			gZo = (self.Wwo:t() * gbw):mul(beta)

			-- update grad_i
			tree.grad_i[col_i]
						:copy((self.Wwi:t() * torch.cmul(self.Ws:t(),-gold_io_prime)):mul(beta))
			stt_subtree.grad_i[{{},{1}}]
						:copy((self.Wwi:t() * torch.cmul(self.Ws:t(),stt_io_prime)):mul(beta))
		end

		local input = nil
		for j = 1, tree.n_children[i] do
			if input == nil then 
				input = tree.gradZo[{{},{tree.children_id[{j,i}]}}]
			else
				input:add(tree.gradZo[{{},{tree.children_id[{j,i}]}}])
			end
		end
		if input ~= nil then 
			gZo:add(rule.Wop:t() * input)
		end
		gZo:cmul(self.funcPrime(tree.outer[col_i]))
		tree.gradZo[col_i]:copy(gZo)

		-- Wop, Wo, bo
		local parent_id = tree.parent_id[i]
		local rule = self.rules[parent_id]
		local grad_rule = grad.rules[parent_id]

		local gWop = gZo * tree.outer[{{},{parent_id}}]:t()
		grad_rule.Wop:add(gWbop)

		for j = 1, tree.n_children[parent_id] do
			local child_id = tree.children_id[{j,parent_id}]
			if child_id ~= i then
				local gWo = gZo * tree.inner[{{},{child_id}}]:t()
				grad_rule.Wo[j]:add(gWo)
			else
				grad_rule.bo[j]:add(gZo)
			end
		end
	end

	-- root
	local input = tree.gradZo[{{},{tree.children_id[{1,1}]}}]
	for j = 2, tree.n_children[1] do
		input:add(tree.gradZo[{{},{tree.children_id[{j,1}]}}])
	end
	local gZo = self.rules[tree.rule_id[1]].Wop:t() * input
	grad.root_outer:add(gZo)
end

function CCIORNN:backpropagate_inside(tree, grad) --[[, alpha, beta)]]
	local alpha = 0; beta = 1

	-- gradient over Z of softmax nodes
	if tree.gradZi == nil then
		tree.gradZi = torch.zeros(self.dim, tree.n_nodes)
	else
		tree.gradZi:fill(0)
	end

	--tree.gradZCat = (tree.cat_predict - tree.category):mul(alpha)

	for i = 1,tree.n_nodes do
		local col_i = {{},{i}}
		--local gZCat = tree.gradZCat[col_i]
		
		local gZi = --[[self.WCat:t() * gZCat +]]  tree.grad_i[col_i]

		-- if not the root
		if tree.parent_id[i] > 0 then
			local parent_id = tree.parent_id[i]
			local rule = self.rules[tree.rule_id[parent_id]]
			
			for j = 1, tree.n_children[parent_id] do
				local child_id = tree.children_id[{j,parent_id}]
				if child_id == i then
					gZi:add(rule.Wi[j] * tree.gradZi[{{},{parent_id}}])
				else 
					gZi:add(rule.Wo[j] * tree.gradZo[{{},{child_id}}])
				end
			end
		end

		-- for internal node
		if tree.n_children[i] > 0 then	
			gZi:cmul(self.funcPrime(tree.inner[col_i]))
			tree.gradZi[col_i]:copy(gZi)

			--[[ WCat, bCat
			grad.WCat:add(gZCat * tree.inner[col_i]:t())
			grad.bCat:add(gZCat)
			]]

			-- weight matrices for inner
			local grad_rule = grad.rules[tree.rule_id[i]]
			for j = 1, tree.n_children[i] do
				local child_id = tree.children_id[j]
				grad_rule.Wi[j]:add(gZi * tree.inner[{{},{child_id}}]:t())
			end
			grad_rule.bi:add(gZi)

		else -- leaf
			tree.gradZi[col_i]:copy(gZi)

			--[[ compute gradient
			grad.WCat:add(gZCat * tree.inner[col_i]:t())
			grad.bCat:add(gZCat)
			]]

			if self.update_L then
				grad.L[{{},{tree.word_id[i]}}]:add(gZi)
			end
		end
	end
end

function CCIORNN:backpropagate(tree, grad, bag_of_subtrees) --[[, alpha, beta)]]
	local alpha = 0; beta = 1

	-- compute costs = alpha * cat cost + (beta) * word_cost
	local cat_cost = 0 --tree.cat_error:sum()
	local word_cost = tree.stt_error:sum()
	local cost = alpha*cat_cost + beta*word_cost

	--print(tree.stt_id)
	self:backpropagate_outside(tree, grad, bag_of_subtrees) --, alpha, beta)
	self:backpropagate_inside(tree, grad) --, alpha, beta)

	for i = 2,tree.n_nodes do
		if tree.stt_id[i] > 0 then
			local stt_subtree = bag_of_subtrees[tree.stt_id[i]]	
			self:backpropagate_inside(stt_subtree, grad) --, alpha, beta)
		end
	end

	return cost
end

function CCIORNN:computeCostAndGrad(treebank, config, bag_of_subtrees)
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

		for j = 1,#rule.Wi do
			grad.rules[i].Wi[j] = torch.zeros(rule.Wi[j]:size())
			grad.rules[i].Wo[j] = torch.zeros(rule.Wo[j]:size())
			grad.rules[i].bo[j] = torch.zeros(rule.bo[j]:size())
		end		

		grad.rules[i].bi = torch.zeros(rule.bi)
		grad.rules[i].Wop = torch.zeros(rule.Wop)
	end

	grad.Wwi = torch.zeros(self.Wwi:size())
	grad.Wwo = torch.zeros(self.Wwo:size())
	grad.bw = torch.zeros(self.bw:size())
	grad.Ws = torch.zeros(self.Ws:size())

	grad.WCat = torch.zeros(self.WCat:size())
	grad.bCat = torch.zeros(self.bCat:size())
	
	if self.update_L then
		grad.L = torch.zeros(self.L:size())
	end
	grad.root_outer = torch.zeros(self.root_outer:size())

	local cost = 0
	local nSample = #treebank
	for i, tree in ipairs(treebank)  do
		self:forward(tree, bag_of_subtrees)
		if parse == false then
			cost = cost + self:backpropagate(tree, grad, bag_of_subtrees) --, config.alpha, config.beta)
		end
	end

	if parse == false then
		local M = self:fold()
		grad = self:fold(grad)

		cost = cost / nSample + config.lambda/2 * torch.pow(M,2):sum()
		grad:div(nSample):add(M * config.lambda)
	end

	return cost, grad, treebank
end
end

-- check gradient
function CCIORNN:checkGradient(treebank, config, bag_of_subtrees)
	local epsilon = 1e-4

	local dim = self.dim
	local wrdDicLen = self.wrdDicLen
	local nCat = self.nCat

	local Theta = self:fold()
	local _, gradTheta = self:computeCostAndGrad(treebank, config, bag_of_subtrees)

	local n = Theta:nElement()
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
function CCIORNN:eval(treebank)
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

function CCIORNN:train_with_adagrad(traintreebank, devtreebank, batchSize, 
									maxepoch, lambda, alpha, beta, prefix,
									adagrad_config, adagrad_state, bag_of_subtrees)
	local nSample = #traintreebank
	--print('accuracy = ' .. self:eval(devtreebank)) io.flush()
	
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
							{lambda = lambda, alpha = alpha, beta = beta}, 
							bag_of_subtrees)
			p:lap("compute grad")

			-- for visualization
			if math.mod(j,2) == 0 then
				print('iter ' .. j .. ': ' .. cost) io.flush()
			end
			
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

function CCIORNN:parse(treebank)
	local old_uL = self.update_L
	self.update_L = false
	_, _, treebank = self:computeCostAndGrad(treebank, {parse=true}, {})
	self.update_L = old_uL
	return treebank
end

--[[*********************************** test ******************************--
	torch.setnumthreads(1)
	word2id = {
		['yet'] = 1,
		['the'] = 2,
		['act'] = 3,
		['is'] = 4,
		['still'] = 5,
		['charming'] = 6,
		['here'] = 7,
		['.'] = 8,
		['a'] = 9,
		['screenplay'] = 10,
		['more'] = 11,
		['ingeniously'] = 12,
		['constructed'] = 13, 
		['than'] = 14, 
		['``'] = 15,
		['memento'] = 16,
		["''"] = 17 }

	struct = {Lookup = torch.randn(3,17), nCategory = 5, func = tanh, funcPrime = tanhPrime}
	net = CCIORNN:new(struct)

	treebank = {}
	treebank[1] = Tree:create_from_string("(3 (2 Yet) (3 (2 (2 the) (2 act)) (3 (4 (3 (2 is) (3 (2 still) (4 charming))) (2 here)) (2 .))))")
	treebank[2] = Tree:create_from_string("(4 (2 (2 a) (2 (2 screenplay) (2 more))) (3 (4 ingeniously) (2 (2 constructed) (2 (2 (2 (2 than) (2 ``)) (2 Memento)) (2 '')))))")
	treebank[3] = Tree:create_from_string("(2 (4 (1 the) (2 act)) (1 screenplay))")
	treebank[4] = Tree:create_from_string("(3 (2 (1 is) (1 more)) (3 (2 than) (3 here)))")

	require "dict"
	dic = Dict:new(huang_template)
	dic.word2id = word2id

	bag_of_subtrees = {}
	for _,tree in ipairs(treebank) do
		for _,subtree in ipairs(tree:all_nodes()) do
			bag_of_subtrees[#bag_of_subtrees+1] = subtree:to_torch_matrices(dic, 5)
		end
	end
	
	for i = 1,#treebank do
		treebank[i] = treebank[i]:to_torch_matrices(dic, 5)
	end

	config = {lambda = 1e-3, alpha = 0, beta = 1}
	net.update_L = true
	net:checkGradient(treebank, config, bag_of_subtrees)
]]
