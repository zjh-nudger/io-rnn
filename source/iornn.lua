require 'tree'
require 'utils'

NPROCESS = 1

--**************** rerursive neural network class ******************--
IORNN = {}
IORNN_mt = {__index = IORNN}

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
function IORNN:new(struct)

	local dim = struct.Lookup:size(1)
	local wrdDicLen = struct.Lookup:size(2)
	local nCat = struct.nCategory

	local net = {dim = dim, wrdDicLen = wrdDicLen, nCat = nCat}
	local mul = 0.1
	
	-- unary branch inner
	net.Wui = uniform(dim, dim, -1, 1):mul(mul)
	net.bui = uniform(dim, 1, -1, 1):mul(0)

	-- binary branch inner
	net.Wbil = uniform(dim, dim, -1, 1):mul(mul)	--left
	net.Wbir = uniform(dim, dim, -1, 1):mul(mul)	--right
	net.bbi = uniform(dim, 1, -1, 1):mul(0)

	-- binary brach outer
	net.Wbol = uniform(dim, dim, -1, 1):mul(mul)	--left sister
	net.Wbor = uniform(dim, dim, -1, 1):mul(mul)	--right sister
	net.Wbop = uniform(dim, dim, -1, 1):mul(mul)	--parent
	net.bbol = uniform(dim, 1, -1, 1):mul(0)
	net.bbor = uniform(dim, 1, -1, 1):mul(0)

	-- word ranking
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
	
	setmetatable(net, IORNN_mt)
	return net
end

-- save net into a bin file
function IORNN:save( filename )
	local file = torch.DiskFile(filename, 'w')
	file:binary()
	file:writeObject(self)
	file:close()
end

-- create net from file
function IORNN:load( filename , func, funcPrime )
	local file = torch.DiskFile(filename, 'r')
	file:binary()
	local net = file:readObject()
	setmetatable(net, IORNN_mt)
	file:close()

	net.func = func or tanh
	net.funcPrime = funcPrime or tanhPrime
	return net
end

-- fold parameters to a vector
function IORNN:fold( Model )
	local Model = Model or {}
	local Params = {
		Model.Wbil or self.Wbil,	--  9
		Model.Wbir or self.Wbir,	-- 18
		Model.bbi or self.bbi,		-- 21
		
		Model.Wbol or self.Wbol,	-- 30
		Model.Wbor or self.Wbor,	-- 39
		Model.Wbop or self.Wbop,	-- 48
		Model.bbol or self.bbol,	-- 51
		Model.bbor or self.bbor,	-- 54

		Model.Wui or self.Wui,		-- 63
		Model.bui or self.bui,		-- 66

		Model.WCat or self.WCat,	-- 81
		Model.bCat or self.bCat,	-- 86

		Model.Wwi or self.Wwi,
		Model.Wwo or self.Wwo,
		Model.bw or self.bw,
		Model.Ws or self.Ws,

		Model.root_outer or self.root_outer
	}
	if self.update_L == true then
		Params[#Params+1] = Model.L or self.L
	end

	local dim = self.dim
	local wrdDicLen = self.wrdDicLen
	local nCat = self.nCat

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
	local Params = {
			self.Wbil, self.Wbir, self.bbi,
			self.Wbol, self.Wbor, self.Wbop, self.bbol, self.bbor,
			self.Wui, self.bui,
			self.WCat, self.bCat,
			self.Wwi, self.Wwo, self.bw, self.Ws,
			self.root_outer
		}
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
function IORNN:forward(tree)

	local Wbil = self.Wbil; local Wbir = self.Wbir; local bbi = self.bbi
	local Wbol = self.Wbol; local Wbor = self.Wbor; local Wbop = self.Wbop; 
	local bbol = self.bbol; local bbor = self.bbor
	local Wui = self.Wui; local bui = self.bui; 
	local WCat = self.WCat; local bCat = self.bCat
	local Wwi = self.Wwi; local Wwo = self.Wwo; local bw = self.bw
	local Ws = self.Ws
	local L = self.L

	local func = self.func
	local funcPrime = self.funcPrime

	local dim = self.dim
	local wrdDicLen = self.wrdDicLen
	local nCat = self.nCat

	local n_nodes = tree.n_nodes

	--################### inside ###############--
	tree.inner = torch.Tensor(dim, tree.n_nodes)
	for i = tree.n_nodes,1,-1 do 
		local col_i = {{},{i}}

		-- for leaves
		if tree.n_children[i] == 0 then
			tree.inner[col_i]:copy(L[{{},{tree.word_id[i]}}])

		-- for internal nodes
		else
			local input
			if tree.n_children[i] == 2 then
				local c1inner = tree.inner[{{},{tree.children_id[{1,i}]}}]
				local c2inner = tree.inner[{{},{tree.children_id[{2,i}]}}]
				input = (Wbil*c1inner):add(Wbir*c2inner):add(bbi)

			elseif tree.n_children[i] == 1 then
				local cinner = tree.inner[{{},{tree.children_id[{1,i}]}}]
				input = (Wui*cinner):add(bui)

			else error('accept only binary trees') end
		
			tree.inner[col_i]:copy(func(input))
		end
	end

	-- compute classification error
	tree.cat_predict = safe_compute_softmax(
						(WCat*tree.inner)
						:add(torch.repeatTensor(bCat, 1, tree.n_nodes)))
	tree.cat_error = (-torch.cmul(tree.category, torch.log(tree.cat_predict))):sum(1)

	--################# outside ################--
	tree.outer = torch.Tensor(dim, tree.n_nodes)
	
	-- for substitued word
	tree.stt_word_id = torch.rand(tree.n_nodes):mul(self.wrdDicLen):add(1):floor()
	--tree.stt_word_id = torch.Tensor(tree.n_nodes):fill(3) -- for gradient checking only
	tree.stt_word_emb = torch.Tensor(dim, tree.n_nodes)
	tree.stt_word_score = torch.zeros(tree.n_nodes)
	tree.stt_word_io = torch.zeros(Ws:size(2), tree.n_nodes)	-- combination of inner and outer meanings

	-- for gold standard word
	tree.word_score = torch.zeros(n_nodes)
	tree.word_io = torch.zeros(Ws:size(2), tree.n_nodes)

	-- process
	tree.outer[{{},{1}}]:copy(self.root_outer)
	tree.word_error = torch.zeros(tree.n_nodes)

	for i = 2,tree.n_nodes do
		local col_i = {{},{i}}
		local input = nil

		if tree.sister_id[i] > 0 then
			input = Wbop * tree.outer[{{},{tree.parent_id[i]}}]

			if tree.child_pos[i] == 1 then
				input:add(Wbol * tree.inner[{{},{tree.sister_id[i]}}]):add(bbol)
			elseif tree.child_pos[i] == 2 then
				input:add(Wbor * tree.inner[{{},{tree.sister_id[i]}}]):add(bbor)
			else
				error('accept only binary tree')
			end
		else
			error("unary branching: not implement yet")
		end

		tree.outer[col_i]:copy(func(input))

		-- leaf: compute word score
		if tree.n_children[i] == 0 then
			tree.word_io[col_i]:copy(func(	(Wwo * tree.outer[col_i])
											:add(Wwi * tree.inner[col_i]):add(bw)))
			tree.word_score[i] = Ws * tree.word_io[col_i]

			tree.stt_word_emb[col_i]:copy(L[{{},{tree.stt_word_id[i]}}])
			tree.stt_word_io[col_i]:copy(func(	
											(Wwo * tree.outer[col_i])
											:add(Wwi * tree.stt_word_emb[col_i]):add(bw)))
			tree.stt_word_score[i] = Ws * tree.stt_word_io[col_i]

			tree.word_error[i] = math.max(0, 1 - tree.word_score[i] + tree.stt_word_score[i])
		end
	end
end

--*********************** backpropagate *********************--
-- only for one tree/sentence
-- input:
-- 	tree : result of the parse function
-- output:
function IORNN:backpropagate(tree, grad, alpha, beta)

	local dim = self.dim
	local wrdDicLen = self.wrdDicLen
	local nCat = self.nCat

	local Wbil = self.Wbil; local Wbir = self.Wbir; local bbi = self.bbi
	local Wbol = self.Wbol; local Wbor = self.Wbor; local Wbop = self.Wbop; 
	local bbol = self.bbol; local bbor = self.bbor
	local Wui = self.Wui; local bui = self.bui
	local WCat = self.WCat; local bCat = self.bCat
	local Wwo = self.Wwo; local Wwi = self.Wwi; local bw = self.bw
	local Ws = self.Ws
	local L = self.L

	local func = self.func
	local funcPrime = self.funcPrime

	-- compute costs = alpha * cat cost + (beta) * word_cost
	local cat_cost = tree.cat_error:sum()
	local word_cost = tree.word_error:sum()

	local cost = alpha*cat_cost + beta*word_cost


	--*************** outside *************-
	local tree_gradZo = torch.zeros(dim, tree.n_nodes)

	for i = tree.n_nodes, 2, -1 do
		local col_i = {{},{i}}
		local gZo = torch.zeros(dim, 1)

		-- leaf: word ranking error
		if tree.n_children[i] == 0 and tree.word_error[i] > 0 then
			local word_io_prime = funcPrime(tree.word_io[col_i])
			local stt_word_io_prime = funcPrime(tree.stt_word_io[col_i])

			-- Ws
			local gWs = (tree.stt_word_io[col_i] - tree.word_io[col_i]):t()
			grad.Ws:add(gWs * beta)

			-- Wwo, Wwi, bw
			local gbw = (stt_word_io_prime - word_io_prime):cmul(Ws:t())
			grad.bw:add(gbw * beta)
			local gWwo = gbw * tree.outer[col_i]:t()
			grad.Wwo:add(gWwo * beta)

			local gWwi = torch.cmul(Ws:t(), stt_word_io_prime) * tree.stt_word_emb[col_i]:t()
						- torch.cmul(Ws:t(), word_io_prime) * tree.inner[col_i]:t()
			grad.Wwi:add(gWwi * beta)
	
			-- gradZo
			gZo = funcPrime(tree.outer[col_i]):cmul(Wwo:t() * gbw):mul(beta)
			tree_gradZo[col_i]:copy(gZo)

			-- update lexsem
			if self.update_L then
				grad.L[{{},{tree.word_id[i]}}]
							:add((Wwi:t() * torch.cmul(Ws:t(),-word_io_prime)):mul(beta))
				grad.L[{{},{tree.stt_word_id[i]}}]
							:add((Wwi:t() * torch.cmul(Ws:t(),stt_word_io_prime)):mul(beta))
			end

		-- nonterminal node
		elseif tree.n_children[i] > 0 then
			if tree.n_children[i] == 1 then
				error("unary branching: not implement yet")
			elseif tree.n_children[i] == 2 then
				gZo = Wbop:t() * (	tree_gradZo[{{},{tree.children_id[{1,i}]}}] 
									+ tree_gradZo[{{},{tree.children_id[{2,i}]}}])
			else
				error('only binary trees')
			end
			gZo:cmul(funcPrime(tree.outer[col_i]))
			tree_gradZo[col_i]:copy(gZo)
		end

		-- Wbop, Wbol, Wbor, bbol, bbor
		-- binary
		if tree.sister_id[i] > 0 then
			if gZo == nil then
				print(i)
				for k,v in pairs(tree) do
					print(k)
					print(v)
				end
			end
			local gWbop = gZo * tree.outer[{{},{tree.parent_id[i]}}]:t()
			grad.Wbop:add(gWbop)

			local gWbo = gZo * tree.inner[{{},{tree.sister_id[i]}}]:t()
			local gbbo = gZo

			-- if this is left child
			if tree.child_pos[i] == 1 then
				grad.Wbol:add(gWbo)
				grad.bbol:add(gbbo)
			else -- right child 
				grad.Wbor:add(gWbo)
				grad.bbor:add(gbbo)
			end
		else
			-- not root , unary
			error("not implement yet")
		end
	end

	-- root
	local gZo = nil
	if tree.n_children[1] == 2 then
		gZo = Wbop:t() * (	tree_gradZo[{{},{tree.children_id[{1,1}]}}] 
							+ tree_gradZo[{{},{tree.children_id[{2,1}]}}] )
	else
		error("not implement yet")
	end
	grad.root_outer:add(gZo)


	--*************** inside ****************
	-- gradient over Z of softmax nodes
	local tree_gradZi = torch.zeros(dim, tree.n_nodes)
	local tree_gradZCat = (tree.cat_predict - tree.category):mul(alpha)

	for i = 1,tree.n_nodes do
		local col_i = {{},{i}}
		local gZCat = tree_gradZCat[col_i]

		-- for internal node
		if tree.n_children[i] > 0 then	
			-- gradZi
			local gZi = WCat:t() * gZCat
			if tree.parent_id[i] > 0 then
				if tree.child_pos[i] == 1 then
					gZi:add(Wbil:t()* tree_gradZi[{{},{tree.parent_id[i]}}])
					if tree.sister_id[i] > 0 then
						gZi:add(Wbor:t() * tree_gradZo[{{},{tree.sister_id[i]}}])
					end
				else
					gZi:add(Wbir:t()* tree_gradZi[{{},{tree.parent_id[i]}}])
					gZi:add(Wbol:t()* tree_gradZo[{{},{tree.sister_id[i]}}])
				end	
			end
			gZi:cmul(funcPrime(tree.inner[col_i]))
			tree_gradZi[col_i]:copy(gZi)

			-- WCat, bCat
			grad.WCat:add(gZCat * tree.inner[col_i]:t())
			grad.bCat:add(gZCat)

			-- binary tree
			-- Wbil, Wbir, bbi
			if tree.n_children[i] == 2 then
				local c1id = tree.children_id[{1,i}]
				local c2id = tree.children_id[{2,i}]
				grad.Wbil:add(gZi * tree.inner[{{},{c1id}}]:t())
				grad.Wbir:add(gZi * tree.inner[{{},{c2id}}]:t())
				grad.bbi:add(gZi)
	
			-- unary tree
			-- Wbui, bui
			elseif tree.n_children[i] == 1 then
				local cid = tree.children_id[{1,i}]
				grad.Wui:add(gZi * tree.inner[{{},{cid}}]:t())
				grad.bui:add(gZi)

			else error('accept only binary trees') end

		else -- leaf
			-- gradZi
			local gZi = WCat:t() * gZCat
			if tree.parent_id[i] > 0 then
				if tree.child_pos[i] == 1 then
					gZi:add(Wbil:t()* tree_gradZi[{{},{tree.parent_id[i]}}])
					if tree.sister_id[i] > 0 then
						gZi:add(Wbor:t() * tree_gradZo[{{},{tree.sister_id[i]}}])
					end
				else
					gZi:add(Wbir:t()* tree_gradZi[{{},{tree.parent_id[i]}}])
					gZi:add(Wbol:t()* tree_gradZo[{{},{tree.sister_id[i]}}])
				end	
			end
			tree_gradZi[col_i]:copy(gZi)

			-- compute gradient
			grad.WCat:add(gZCat * tree.inner[col_i]:t())
			grad.bCat:add(gZCat)
			if self.update_L then
				grad.L[{{},{tree.word_id[i]}}]:add(gZi)
			end
		end
	end
	
	return cost
end

function IORNN:computeCostAndGrad(treebank, config)
	local parse = config.parse or false
	
if NPROCESS > 1 then
else
-- for single process
	local grad = {
		root_outer = torch.zeros(self.root_outer:size()),

		Wbil = torch.zeros(self.Wbil:size()),
		Wbir = torch.zeros(self.Wbir:size()),
		bbi = torch.zeros(self.bbi:size()),
		
		Wui = torch.zeros(self.Wui:size()),
		bui = torch.zeros(self.bui:size()),

		Wbop = torch.zeros(self.Wbop:size()),
		Wbol = torch.zeros(self.Wbol:size()),
		Wbor = torch.zeros(self.Wbor:size()),
		bbol = torch.zeros(self.bbol:size()),
		bbor = torch.zeros(self.bbor:size()),
		
		WCat = torch.zeros(self.WCat:size()),
		bCat = torch.zeros(self.bCat:size()),

		Wwo = torch.zeros(self.Wwo:size()),
		Wwi = torch.zeros(self.Wwi:size()),
		bw = torch.zeros(self.bw:size()),
		Ws = torch.zeros(self.Ws:size())
	}
	if self.update_L then
		grad.L = torch.zeros(self.L:size())
	end

	local cost = 0
	local nSample = #treebank
	for i, tree in ipairs(treebank)  do
		self:forward(tree)
		if parse == false then
			cost = cost + self:backpropagate(tree, grad, config.alpha, config.beta)
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
function IORNN:checkGradient(treebank, config)
	local epsilon = 1e-4

	local dim = self.dim
	local wrdDicLen = self.wrdDicLen
	local nCat = self.nCat

	local Theta = self:fold()
	local _, gradTheta = self:computeCostAndGrad(treebank, config)

	local n = Theta:nElement()
	local numGradTheta = torch.zeros(n)
	for i = 1,n do
		local index = {{i}}
		Theta[index]:add(epsilon)
		self:unfold(Theta)
		local costPlus,_ = self:computeCostAndGrad(treebank, config)
		
		Theta[index]:add(-2*epsilon)
		self:unfold(Theta)
		local costMinus,_ = self:computeCostAndGrad(treebank, config)
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


--***************************** eval **************************
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

--**************************** training ************************--
require 'optim'
require 'xlua'
p = xlua.Profiler()

function IORNN:train_with_adagrad(traintreebank, devtreebank, batchSize, 
									maxepoch, lambda, alpha, beta, prefix,
									adagrad_config, adagrad_state)
	local nSample = #traintreebank
	print('accuracy = ' .. self:eval(devtreebank)) io.flush()
	
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
							{lambda = lambda, alpha = alpha, beta = beta})
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

function IORNN:parse(treebank) 
	_, _, treebank = self:computeCostAndGrad(treebank, {parse=true})
	return treebank
end

--*********************************** test ******************************--
--[[
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
	net = IORNN:new(struct)
	t1 = Tree:create_from_string("(3 (2 Yet) (3 (2 (2 the) (2 act)) (3 (4 (3 (2 is) (3 (2 still) (4 charming))) (2 here)) (2 .))))")
	t2 = Tree:create_from_string("(4 (2 (2 a) (2 (2 screenplay) (2 more))) (3 (4 ingeniously) (2 (2 constructed) (2 (2 (2 (2 than) (2 ``)) (2 Memento)) (2 '')))))")
	t3 = Tree:create_from_string("(2 (4 (1 the) (2 act)) (1 screenplay))")
	t4 = Tree:create_from_string("(3 (2 (1 is) (1 more)) (3 (2 than) (3 here)))")


	require "dict"
	dic = Dict:new(huang_template)
	dic.word2id = word2id
	
	t1 = t1:to_torch_matrices(dic, 5)
	t2 = t2:to_torch_matrices(dic, 5)
	t3 = t3:to_torch_matrices(dic, 5)
	t4 = t4:to_torch_matrices(dic, 5)

	config = {lambda = 1e-3, alpha = 0.8, beta = 0.2}
	net.update_L = false
	net:checkGradient({t1,t2,t3,t4},config)
]]



