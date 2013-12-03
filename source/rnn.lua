require 'tree'
require 'utils'

NPROCESS = 1

--**************** rerursive neural network class ******************--
RNN = {}
RNN_mt = {__index = RNN}

--****************** needed functions ******************--
-- generate a n x m matrix by uniform distibution within range [min,max]
function uniform(n, m, min, max)
	local M = torch.rand(n, m)
	M:mul(max-min):add(min)
	return M
end

-- normalization
-- input: 
-- 	X : n x m matrix, each colum is a n-dim vector
-- 	p : norm
-- output: norm_p of 
function normalize(X, p)
	if p == 1 then
		return torch.cdiv(X, torch.ones(X:size(1),1) * X:sum(1))
	elseif p == 2 then	
		return torch.cdiv(X, torch.ones(X:size(1),1) * torch.pow(X,p):sum(1):sqrt())
	else 
		return torch.cdiv(X, torch.ones(X:size(1),1) * torch.pow(X,p):sum(1):pow(1/p))
	end
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
function RNN:new(struct)

	local dim = struct.Lookup:size(1)
	local wrdDicLen = struct.Lookup:size(2)
	local nCat = struct.nCategory

	local net = {dim = dim, wrdDicLen = wrdDicLen, nCat = nCat}
	local mul = 0.1

	-- unary branch
	net.Wu = uniform(dim, dim, -1, 1):mul(mul)
	net.bu = uniform(dim, 1, -1, 1):mul(0)

	-- binary branch
	net.Wb1 = uniform(dim, dim, -1, 1):mul(mul)
	net.Wb2 = uniform(dim, dim, -1, 1):mul(mul)
	net.bb = uniform(dim, 1, -1, 1):mul(0)

	-- classification
	net.WCat = uniform(nCat, dim, -1, 1):mul(mul)
	net.bCat = uniform(nCat, 1, -1, 1):mul(0)

	-- wordembedding
	net.L = torch.randn(struct.Lookup:size()):mul(0.0001)
	net.func = struct.func
	net.funcPrime = struct.funcPrime
	
	setmetatable(net, RNN_mt)
	return net
end

-- save net into a bin file
function RNN:save( filename )
	local file = torch.DiskFile(filename, 'w')
	file:binary()
	file:writeObject(self)
	file:close()
end

-- create net from file
function RNN:load( filename , func, funcPrime )
	local file = torch.DiskFile(filename, 'r')
	file:binary()
	local net = file:readObject()
	setmetatable(net, RNN_mt)
	file:close()

	net.func = func or tanh
	net.funcPrime = funcPrime or tanhPrime
	return net
end

-- fold parameters to a vector
function RNN:fold( Model )
	local Model = Model or {}
	local Params = {
		Model.Wb1 or self.Wb1,
		Model.Wb2 or self.Wb2,
		Model.bb or self.bb,

		Model.Wu or self.Wu,
		Model.bu or self.bu,

		Model.WCat or self.WCat,
		Model.bCat or self.bCat,

		Model.L or self.L
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
function RNN:unfold(Theta)
	local Params = {
			self.Wb1, self.Wb2, self.bb,
			self.Wu, self.bu,
			self.WCat, self.bCat,
			self.L
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
function RNN:forward(tree)

	local Wb1 = self.Wb1; local Wb2 = self.Wb2; local bb = self.bb
	local Wu = self.Wu; local bu = self.bu; 
	local WCat = self.WCat; local bCat = self.bCat
	local L = self.L

	local func = self.func
	local funcPrime = self.funcPrime

	local dim = self.dim
	local wrdDicLen = self.wrdDicLen
	local nCat = self.nCat

	tree.feature = torch.Tensor(dim, tree.n_nodes)

	-- process
	for i = tree.n_nodes,1,-1 do 
		-- for leaves
		if tree.n_children[i] == 0 then
			tree.feature[{{},{i}}]:copy(L[{{},{tree.word_id[i]}}])

		-- for internal nodes
		else
			local input
			if tree.n_children[i] == 2 then
				local c1feature = tree.feature[{{},{tree.children_id[{1,i}]}}]
				local c2feature = tree.feature[{{},{tree.children_id[{2,i}]}}]
				input = (Wb1*c1feature):add(Wb2*c2feature):add(bb)

			elseif tree.n_children[i] == 1 then
				local cfeature = tree.feature[{{},{tree.children_id[{1,i}]}}]
				input = (Wu*cfeature):add(bu)

			else error('accept only binary trees') end
		
			tree.feature[{{},{i}}] = func(input) 
		end
	end

	-- compute classification error
	tree.predict = safe_compute_softmax(
						(WCat*tree.feature)
						:add(torch.repeatTensor(bCat, 1, tree.n_nodes)))
	tree.ecat = (-torch.cmul(tree.category, torch.log(tree.predict))):sum(1)
end

--*********************** backpropagate *********************--
-- only for one tree/sentence
-- input:
-- 	tree : result of the parse function
-- output:
function RNN:backpropagate(tree, grad)

	local dim = self.dim
	local wrdDicLen = self.wrdDicLen
	local nCat = self.nCat

	local gradL = grad.L
	local gradWb1 = grad.Wb1; local gradWb2 = grad.Wb2; local gradbb = grad.bb
	local gradWu = grad.Wu; local gradbu = grad.bu
	local gradWCat = grad.WCat; local gradbCat = grad.bCat

	local cost = 0

	local Wb1 = self.Wb1; local Wb2 = self.Wb2; local bb = self.bb
	local Wu = self.Wu; local bu = self.bu
	local WCat = self.WCat; local bCat = self.bCat
	local L = self.L

	local func = self.func
	local funcPrime = self.funcPrime

	local cost = tree.ecat:sum()
	local gradZCat = tree.predict - tree.category

	local support = {}
	support[1] = {
		W = torch.zeros(dim,dim),
		b = torch.zeros(dim,1),
		gradZp = torch.zeros(dim,1) }

	for i = 1,tree.n_nodes do
		local W = support[i].W
		local b = support[i].b
		local gradZp = support[i].gradZp

		gradZCat_i = gradZCat[{{},{i}}]
		feature_i = tree.feature[{{},{i}}]

		-- for internal node
		if tree.n_children[i] > 0 then	
			-- compute gradZ
			local gradZ = 	funcPrime(feature_i):cmul(
							(WCat:t() * gradZCat_i):add(W:t()* gradZp))

			-- compute gradient
			gradWCat:add(gradZCat_i * feature_i:t())
			gradbCat:add(gradZCat_i)

			if tree.n_children[i] == 2 then
				local c1id = tree.children_id[{1,i}]
				local c2id = tree.children_id[{2,i}]
				gradWb1:add(gradZ * tree.feature[{{},{c1id}}]:t())
				gradWb2:add(gradZ * tree.feature[{{},{c2id}}]:t())
				gradbb:add(gradZ)
			
				-- propagate to its children
				support[c1id] = {W = Wb1, gradZp = gradZ}
				support[c2id] = {W = Wb2, gradZp = gradZ}


			elseif tree.n_children[i] == 1 then
				local cid = tree.children_id[{1,i}]
				gradWu:add(gradZ * tree.feature[{{},{cid}}]:t())
				gradbu:add(gradZ)
	
				-- propagate to its children
				support[cid] = {W = Wu, gradZp = gradZ}

			else error('accept only binary trees') end

		else -- leaf
			-- compute gradZ
			local gradZ = (W:t() * gradZp):add(WCat:t() * gradZCat_i)

			-- compute gradient
			gradWCat:add(gradZCat_i * feature_i:t())
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
		Wb1 = torch.zeros(net.Wb1:size()),
		Wb2 = torch.zeros(net.Wb2:size()),
		bb = torch.zeros(net.bb:size()),
		Wu = torch.zeros(net.Wu:size()),
		bu = torch.zeros(net.bu:size()),
		WCat = torch.zeros(net.WCat:size()),
		bCat = torch.zeros(net.bCat:size()),
		L = torch.zeros(net.L:size()) }

	local cost = 0
	local timer = torch.Timer()
	treebank = {}
	for i,tree in ipairs(treebank) do
		RNN.forward(net, tree)
		if not fw_only then
			cost = cost + RNN.backpropagate(net, tree, grad)
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

function RNN:computeCostAndGrad(treebank, config, fw_only)
	
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
		Wb1 = torch.zeros(self.Wb1:size()),
		Wb2 = torch.zeros(self.Wb2:size()),
		bb = torch.zeros(self.bb:size()),
		Wu = torch.zeros(self.Wu:size()),
		bu = torch.zeros(self.bu:size()),
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
function RNN:checkGradient(treebank, config)
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
		print('diff ' .. diff)
	end

	local diff = torch.norm(gradTheta - numGradTheta) 
					/ torch.norm(gradTheta + numGradTheta)
	print(diff)
	print("should be < 1e-9")
end

--***************************** eval **************************
function RNN:eval(treebank)
	_, _, treebank = self:computeCostAndGrad(treebank, {lambda = 0})
	local total_all = 0
	local correct_all = 0

	local total_root = 0
	local correct_root = 0

	for i,tree in ipairs(treebank) do
		total_all = total_all + tree.n_nodes
		total_root = total_root + 1

		local m,_ = tree.predict:max(1)
		pred = 	torch.eq(
					tree.predict, 
					torch.repeatTensor(torch.reshape(m,1,tree.n_nodes), self.nCat,1))
				:double()
		correct_all = correct_all + torch.cmul(pred, tree.category):sum()
		correct_root = correct_root + torch.cmul(pred[{{},{1}}], tree.category[{{},{1}}]):sum()
	end

	return correct_all / total_all, correct_root / total_root
end

--******************************* train networks *************************
--[[-- optFunc from 'optim' package
function RNN:train(traintreebank, validtreebank, batchSize, optFunc, optFuncState, config)
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

function RNN:train_with_adagrad(traintreebank, devtreebank, batchSize, 
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

		if math.mod(iter,500) == 0 then
			local acc = self:eval(devtreebank) * 100
			acc = math.floor(acc * 100) / 100
			print('accuracy = ' .. acc)
			io.flush()

			if math.mod(iter, 50000) == 0 then
				self:save('model/model.rnn.' .. iter .. '_' .. acc)
			end
		end

		collectgarbage()
	end
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
	local dic = Dict:new(huang_template)
	dic.word2id = word2id

	struct = {Lookup = torch.randn(3,17), nCategory = 5, func = tanh, funcPrime = tanhPrime}
	net = RNN:new(struct)
	t1 = Tree:create_from_string("(3 (2 Yet) (3 (2 (2 the) (2 act)) (3 (4 (3 (2 is) (3 (2 still) (4 charming))) (2 here)) (2 .))))")
	t2 = Tree:create_from_string("(4 (2 (2 a) (2 (2 screenplay) (2 more))) (3 (4 ingeniously) (2 (2 constructed) (2 (2 (2 (2 than) (2 ``)) (2 Memento)) (2 '')))))")
	t3 = Tree:create_from_string("(2 (4 the) (1 screenplay))")
	t4 = Tree:create_from_string("(3 (2 (1 is) (1 more)) (3 (2 than) (3 here)))")

	require "dict"
	dic = Dict:new(huang_template)
	dic.word2id = word2id
	
	t1 = t1:to_torch_matrices(dic, 5)
	t2 = t2:to_torch_matrices(dic, 5)
	t3 = t3:to_torch_matrices(dic, 5)
	t4 = t4:to_torch_matrices(dic, 5)

	config = {lambda = 1e-3}
	net:checkGradient({t1,t2},config)
]]


