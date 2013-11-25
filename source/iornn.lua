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
	
	-- unary branch inner
	net.Wui = uniform(dim, dim, -1, 1):mul(0.01)
	net.bui = uniform(dim, 1, -1, 1):mul(0)

	-- binary branch inner
	net.Wbil = uniform(dim, dim, -1, 1):mul(0.01)	--left
	net.Wbir = uniform(dim, dim, -1, 1):mul(0.01)	--right
	net.bbi = uniform(dim, 1, -1, 1):mul(0)

	-- binary brach outer
	net.Wbol = uniform(dim, dim, -1, 1):mul(0.01)	--left sister
	net.Wbor = uniform(dim, dim, -1, 1):mul(0.01)	--right sister
	net.Wbop = uniform(dim, dim, -1, 1):mul(0.01)	--parent
	net.bbol = uniform(dim, 1, -1, 1):mul(0)
	net.bbor = uniform(dim, 1, -1, 1):mul(0)

	-- word ranking
	net.Wwi = uniform(dim, dim, -1, 1):mul(0.01)	-- for combining inner, outer meanings
	net.Wwo = uniform(dim, dim, -1, 1):mul(0.01)
	net.bw = uniform(dim, 1, -1, 1):mul(0)
	net.Ws = uniform(1, dim, -1, 1):mul(0.01)	-- for scoring
	net.bs = uniform(1, 1, -1, 1):mul(0)		

	-- classification
	net.WCat = uniform(nCat, dim, -1, 1):mul(0.01)
	net.bCat = uniform(nCat, 1, -1, 1):mul(0)

	-- wordembedding
	net.L = torch.randn(struct.Lookup:size()) * 0.001
	net.func = struct.func
	net.funcPrime = struct.funcPrime

	-- root outer
	net.root_outer = uniform(dim, 1, -1, 1):mul(0.001)
	
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
		Model.Wbil or self.Wbil,
		Model.Wbir or self.Wbir,
		Model.bbi or self.bbi,
		
		Model.Wbol or self.Wbol,
		Model.Wbor or self.Wbor,
		Model.Wbop or self.Wbop,
		Model.bbol or self.bbol,
		Model.bbor or self.bbor,

		Model.Wui or self.Wui,
		Model.bui or self.bui,

		Model.WCat or self.WCat,
		Model.bCat or self.bCat,

		Model.Wwi or self.Wwi,
		Model.Wwo or self.Wwo,
		Model.bw or self.bw,
		Model.Ws or self.Ws,
		Model.bs or self.bs,

		Model.L or self.L,
		Model.root_outer or self.root_outer
	}

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
			self.Wwi, self.Wwo, self.bw, self.Ws, self.bs,
			self.L, self.root_outer
		}

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
	local Wbol = self.Wbol; local Wbor = self.Wbor; local Wbop = self.Wbop; local bbol = self.bbol; local bbor = self.bbor
	local Wui = self.Wui; local bui = self.bui; 
	local WCat = self.WCat; local bCat = self.bCat
	local Wwi = self.Wwi; local Wwo = self.Wwo; local bw = self.bw
	local Ws = self.Ws; local bs = self.bs
	local L = self.L

	local func = self.func
	local funcPrime = self.funcPrime

	local dim = self.dim
	local wrdDicLen = self.wrdDicLen
	local nCat = self.nCat

	--################### inside ###############--
	tree.inner = torch.Tensor(dim, tree.n_nodes)
	for i = tree.n_nodes,1,-1 do 
		-- for leaves
		if tree.n_children[i] == 0 then
			tree.inner[{{},{i}}]:copy(L[{{},{tree.word_id[i]}}])

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
		
			tree.inner[{{},{i}}] = func(input) 
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
	tree.stt_word_emb = torch.Tensor(dim, tree.n_nodes)
	tree.stt_word_score = torch.zeros(n_nodes)
	tree.stt_word_io = torch.zeros(dim, tree.n_nodes)	-- combination of inner and outer meanings

	-- for gold standard word
	tree.word_score = torch.zeros(n_nodes)
	tree.word_io = torch.zeros(dim, tree.n_nodes)

	-- process
	tree.outer[{{},{1}}]:copy(self.root_outer)

	for i = 2,tree.n_nodes do
		local input = Wbop * tree.outer[{{},{tree.parent_id[i]}}]
		if tree.left_sister_id[i] > 0 then
			input:add(Wbol * tree.inner[{{},{tree.left_sister_id[i]}}])
			input:add(bbol)
		elseif tree.right_sister_id[i] > 0 then
			input:add(Wbor * tree.inner[{{},{tree.right_sister_id[i]}}])
			input:add(bbor)
		else
			error('accept only binary tree')
		end
		tree.outer[{{},{i}}]:copy(func(input))

		-- leaf: compute word score
		if tree.n_children[i] == 0 then
			tree.word_io[{{},{i}}]:copy(func((Wwo * tree.outer[i]):add(Wwi * tree.inner[i]):add(bw)))
			tree.word_score[i] = (Ws * tree.word_io[{{},{i}}]):add(bs)

			tree.stt_word_emb[{{},{i}}]:copy(L[{{},{tree.stt_word_id[i]}}])
			tree.stt_word_io[{{},{i}}]:copy(func((Wwo * tree.outer[i]):add(Wwi * tree.stt_word_emb[{{},{i}}]):add(bw)))
			tree.stt_word_score[i] = (Ws * tree.stt_word_io[{{},{i}}]):add(bs)
		end
	end
end

--*********************** backpropagate *********************--
-- only for one tree/sentence
-- input:
-- 	tree : result of the parse function
-- output:
function IORNN:backpropagate(tree, grad, alpha)

	local dim = self.dim
	local wrdDicLen = self.wrdDicLen
	local nCat = self.nCat

	local Wbil = self.Wbil; local Wbir = self.Wbir; local bbi = self.bbi
	local Wbol = self.Wbol; local Wbor = self.Wbor; local Wbop = self.Wbop; local bbol = self.bbol; local bbor = self.bbor
	local Wui = self.Wui; local bui = self.bui
	local WCat = self.WCat; local bCat = self.bCat
	local Wwo = self.Wwo; local Wwi = self.Wwi; local bw = self.bw
	local Ws = self.Ws; local bs = self.bs
	local L = self.L

	local func = self.func
	local funcPrime = self.funcPrime

	-- compute costs = alpha * cat cost + (1-alpha) * word_cost
	local cat_cost = tree.cat_error:sum()
	local tree.word_error = tree.stt_word_score + 1 - tree.word_score
	tree.word_error:cmul(torch.ge(tree.word_error,0))
	local word_cost = tree.word_error:sum()

	local cost = alpha*cat_cost + (1-alpha)*word_cost

	-- gradient over Z (inner and outer)
	local gradZi = torch.zeros(dim, tree.n_nodes)
	local gradZo = torch.zeros(dim, tree.n_nodes)

	-- gradient over Z of softmax nodes
	local gradZCat = tree.cat_predict - tree.category

	--*************** outside *************-
	for i = tree.n_nodes, 1, -1 do
		-- leaf: word ranking error
		if tree.word_error[i] > 0 then
			Ws:add(tree.stt_word_io[{{},{i}}]
			
		end
	end


	--*************** inside ****************

	for i = 1,tree.n_nodes do
		local W = support[i].W
		local b = support[i].b
		local gradZp = support[i].gradZp

		gradZCat_i = gradZCat[{{},{i}}]
		inner_i = tree.inner[{{},{i}}]

		-- for internal node
		if tree.n_children[i] > 0 then	
			-- compute gradZ
			local gradZ = 	funcPrime(inner_i):cmul(
							(WCat:t() * gradZCat_i):add(W:t()* gradZp))

			-- compute gradient
			gradWCat:add(gradZCat_i * inner_i:t())
			gradbCat:add(gradZCat_i)

			if tree.n_children[i] == 2 then
				local c1id = tree.children_id[{1,i}]
				local c2id = tree.children_id[{2,i}]
				gradWbil:add(gradZ * tree.inner[{{},{c1id}}]:t())
				gradWbir:add(gradZ * tree.inner[{{},{c2id}}]:t())
				gradbbi:add(gradZ)
			
				-- propagate to its children
				support[c1id] = {W = Wbil, gradZp = gradZ}
				support[c2id] = {W = Wbir, gradZp = gradZ}


			elseif tree.n_children[i] == 1 then
				local cid = tree.children_id[{1,i}]
				gradWui:add(gradZ * tree.inner[{{},{cid}}]:t())
				gradbui:add(gradZ)
	
				-- propagate to its children
				support[cid] = {W = Wui, gradZp = gradZ}

			else error('accept only binary trees') end

		else -- leaf
			-- compute gradZ
			local gradZ = (W:t() * gradZp):add(WCat:t() * gradZCat_i)

			-- compute gradient
			gradWCat:add(gradZCat_i * inner_i:t())
			gradbCat:add(gradZCat_i)
			gradL[{{},{tree.word_id[i]}}]:add(gradZ)
		end
	end
	
	return cost
end

--[[************************ compute cost and gradient *****************--
--input:
--output:
require 'parallel'

-- worker function 
function worker()

	require 'rnn'
	local data = parallel.parent:receive()

	local treebank = data.treebank
	local net = data.net
	local config = data.config
	local fw_only = data.fw_only

	local grad = {
		Wbil = torch.zeros(net.Wbil:size()),
		Wbir = torch.zeros(net.Wbir:size()),
		bbi = torch.zeros(net.bbi:size()),
		Wui = torch.zeros(net.Wui:size()),
		bui = torch.zeros(net.bui:size()),
		WCat = torch.zeros(net.WCat:size()),
		bCat = torch.zeros(net.bCat:size()),
		L = torch.zeros(net.L:size()) }

	local cost = 0
	local timer = torch.Timer()
	treebank = {}
	for i,tree in ipairs(treebank) do
		IORNN.forward(net, tree)
		if not fw_only then
			cost = cost + IORNN.backpropagate(net, tree, grad)
		end
	end
	print('time for child running ' .. timer:time().real) io.flush()

	local stats = nil
	if not fw_only then treebank = nil end

	parallel.parent:send({
			cost = cost, 
			grad = rnn.fold(net, grad), 
			treebank = treebank
		})
end
	
-- parent call
function parent(param)

	local treebank = param.treebank
	local nSample = #treebank
	local net = param.net
	local fw_only = param.fw_only

	-- split data
	local size = math.ceil(nSample / NPROCESS)
	local children = parallel.sfork(NPROCESS)
	children:exec(worker)

	-- send data
	local timer = torch.Timer()
	for i = 1, NPROCESS do
		local data = {treebank = {}, net = param.net, config = param.config, fw_only = fw_only}
		for j = 1,size do
			local id = (i-1)*size+j
			if id > nSample then break end
			data.treebank[j] = treebank[id]
		end
		children[i]:send(data)
	end
	print('time for parent -> children ' .. timer:time().real) io.flush()

	-- receive results
	timer = torch.Timer()
	for i = 1, NPROCESS do
		local reply = children[i]:receive()
		param.totalCost = param.totalCost + reply.cost
		if param.totalGrad == nil then
			param.totalGrad = reply.grad
		else
			param.totalGrad:add(reply.grad)
		end

		if fw_only then
			param.treebank = {}
			for j = 1,#reply.treebank do
				param.treebank[#param.treebank+1] = reply.treebank[j]
			end
		end
	end
	print('time for children -> parent ' .. timer:time().real) io.flush()

	timer = torch.Timer()
	children:sync()
	print('time for sync ' .. timer:time().real) io.flush()

	-- finalize
	local M = param.net:fold()
	param.totalCost = param.totalCost / nSample + param.config.lambda/2 * torch.pow(M,2):sum()
	param.totalGrad:div(nSample):add(M * param.config.lambda)
end
]]

function IORNN:computeCostAndGrad(treebank, config, fw_only)
	
if NPROCESS > 1 then
--[[	local param = {
		net = self,
		config = config,
		treebank = treebank,
		totalCost = 0,
		totalGrad = nil,
		fw_only = fw_only or false
	}


	local ok,err = pcall(parent, param)
	if not ok then 	print(err) parallel.close() end
	
	return param.totalCost, param.totalGrad, param.treebank
]]
else
-- for single process
	local grad = {
		L = torch.zeros(self.L:size()),
		Wbil = torch.zeros(self.Wbil:size()),
		Wbir = torch.zeros(self.Wbir:size()),
		bbi = torch.zeros(self.bbi:size()),
		Wui = torch.zeros(self.Wui:size()),
		bui = torch.zeros(self.bui:size()),
		WCat = torch.zeros(self.WCat:size()),
		bCat = torch.zeros(self.bCat:size()) 
	}

	local cost = 0
	local nSample = #treebank
	for i, tree in ipairs(treebank)  do
		self:forward(tree)
		cost = cost + self:backpropagate(tree, grad)
	end

	local M = self:fold()
	grad = self:fold(grad)

	cost = cost / nSample + config.lambda/2 * torch.pow(M,2):sum()
	grad:div(nSample):add(M * config.lambda)

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

		local diff = math.abs((costPlus - costMinus) / (2*epsilon) - gradTheta[i])
		print('diff ' .. diff)
	end
end

--***************************** eval **************************
function IORNN:eval(treebank)
	_, _, treebank = self:computeCostAndGrad(treebank, {lambda = 0})
	local total = 0
	local correct = 0

	for i,tree in ipairs(treebank) do
		total = total + tree.n_nodes
		local m,_ = tree.cat_predict:max(1)
		pred = 	torch.eq(
					tree.cat_predict, 
					torch.repeatTensor(torch.reshape(m,1,tree.n_nodes), self.nCat,1))
				:double()
		correct = correct + torch.cmul(pred, tree.category):sum()
	end

	return correct / total
end

--******************************* train networks *************************
--[[-- optFunc from 'optim' package
function IORNN:train(traintreebank, validtreebank, batchSize, optFunc, optFuncState, config)
	local nSample = #traintreebank
	local j = 0

	local iter = 1
	local timer = torch.Timer()

	print('accuracy = ' .. self:eval(validtreebank)) io.flush()

	local function func(M)
		print('time for optim ' .. timer:time().real) io.flush()
		self:unfold(M)

		-- extract data
		local timer1 = torch.Timer()
		j = j + 1
		if j > nSample/batchSize then j = 1 end
		local subtreebank = {}
		for k = 1,batchSize do
			subtreebank[k] = traintreebank[k+(j-1)*batchSize]
		end
		print('time to extract data ' .. timer1:time().real) io.flush()

		timer1 = torch.Timer()
		local cost, Grad = self:computeCostAndGrad(subtreebank, config)
		print('time to compute cost & grad ' .. timer1:time().real) io.flush()

		-- for visualization
		if math.mod(iter,1) == 0 then
			print('--- iter: ' .. iter)
			print('cost: ' .. cost)
			io.flush()
		end
		if math.mod(iter,10) == 0 then
			print('accuracy = ' .. self:eval(validtreebank))
			--self:save('model/model.' .. math.floor(iter / 10))
			io.flush()
		end

		iter = iter + 1
		collectgarbage()
		
		timer = torch.Timer()
		return cost, Grad
	end

	local M = optFunc(func, self:fold(), optFuncState, optFuncState)
	self:unfold(M)
end
]]

require 'optim'
require 'xlua'
p = xlua.Profiler()

function IORNN:train_with_adagrad(traintreebank, devtreebank, batchSize, 
								maxit, learn_rate, lambda)
	local nSample = #traintreebank
	local j = 0

	print('accuracy = ' .. self:eval(devtreebank)) io.flush()
	local adagrad_config = {learningRate = learn_rate}
	local adagrad_state = {}

	for iter = 1,maxit do
		local function func(M)
			self:unfold(M)

			-- extract data
			j = j + 1
			if j > nSample/batchSize then j = 1 end
			local subtreebank = {}
			for k = 1,batchSize do
				subtreebank[k] = traintreebank[k+(j-1)*batchSize]
			end
			p:start("compute grad")
			cost, Grad = self:computeCostAndGrad(subtreebank, {lambda = lambda})
			p:lap("compute grad")

			-- for visualization
			if math.mod(iter,1) == 0 then
				print('--- iter: ' .. iter)
				print('cost: ' .. cost)
				io.flush()
			end
			
			return cost, Grad
		end
	
		p:start("optim")
		M,_ = optim.adagrad(func, self:fold(), adagrad_config, adagrad_state)
		self:unfold(M)
		p:lap("optim")

		p:printAll()

		if math.mod(iter,100) == 0 then
			print('accuracy = ' .. self:eval(devtreebank))
			io.flush()
		end

		if math.mod(iter, 1000) == 0 then
			self:save('model/model.' .. math.floor(iter / 1000))
		end

		collectgarbage()
	end
end

--*********************************** test ******************************--
--[[
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
	
	t1 = t1:to_torch_matrices(word2id, 5)
	t2 = t2:to_torch_matrices(word2id, 5)

	config = {lambda = 1e-3}
	net:checkGradient({t1,t2},config)
]]



