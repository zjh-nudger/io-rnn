require 'tree'

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
	return torch.diag((-torch.pow(tanhX,2)+1):resize(tanhX:size(1)))
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
	
	-- unary branch
	net.Wu = uniform(dim, dim, -1, 1):mul(0.1)
	net.bu = uniform(dim, 1, -1, 1):mul(0.1)

	-- binary branch
	net.Wb1 = uniform(dim, dim, -1, 1):mul(0.1)
	net.Wb2 = uniform(dim, dim, -1, 1):mul(0.1)
	net.bb = uniform(dim, 1, -1, 1):mul(0.1)

	-- classification
	net.WCat = uniform(nCat, dim, -1, 1):mul(0.1)
	net.bCat = uniform(nCat, 1, -1, 1):mul(0.1)

	-- wordembedding
	net.L = normalize(struct.Lookup, 2)
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
	tree.predict = torch.Tensor(nCat, tree.n_nodes)
	tree.ecat = torch.Tensor(tree.n_nodes)

	-- process
	for i = tree.n_nodes,1,-1 do 
		local predict
		local feature
		local ecat

		-- for leaves
		if tree.n_children[i] == 0 then
			feature = L[{{},{node.label}}]:clone()

			-- classify on single words
			predict = normalize((WCat*node.feature):add(bCat):exp(), 1 )
			ecat = (-torch.cmul(node.cat, torch.log(node.predict))):sum() 

		-- for internal nodes
		else
			local input
			if tree.n_children[i] == 2 then
				local c1feature = tree.feature[{{},{tree.children_id[{1,i}}}]
				local c2feature = tree.feature[{{},{tree.children_id[{2,i}}}]
				input = (Wb1*c1feature):add(Wb2*c2feature):add(bb)

			elseif tree.n_children[i] == 1 then
				local cfeature = tree.feature[{{},{tree.children_id[{1,i}}}]
				input = (Wu*cfeature):add(bu)

			else error('accept only binary trees') end
		
			-- cal parent feature
			local feature = func(input) 

			-- compute classification error
			local predict = normalize((WCat*feature):add(bCat):exp(), 1)
			local ecat = (-torch.cmul(node.cat, torch.log(predict))):sum()

			node.feature = feature
			node.predict = predict
			node.ecat = ecat
		end
	end
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

	local support = {}
	support[1] = {
		W = torch.zeros(dim,dim),
		b = torch.zeros(dim,1),
		gradZp = torch.zeros(dim,1) }

	local nNode = #tree

	for i = 1,nNode do
		local node = tree[i]
		local W = support[i].W
		local b = support[i].b
		local gradZp = support[i].gradZp

		-- for internal node
		if #node.childId > 0 then
			-- compute cost
			cost = cost + node.ecat
			
			-- compute gradZ
			local gradZCat = node.predict - node.cat
			local gradZ = 	funcPrime(node.feature):t() * 
							(WCat:t() * gradZCat):add(W:t()* gradZp)

			-- compute gradient
			gradWCat:add(gradZCat * node.feature:t())
			gradbCat:add(gradZCat)

			if #node.childId == 2 then
				local child1 = tree[node.childId[1]]
				local child2 = tree[node.childId[2]]
				gradWb1:add(gradZ * child1.feature:t())
				gradWb2:add(gradZ * child2.feature:t())
				gradbb:add(gradZ)
			
				-- propagate to its children
				support[node.childId[1]] = {W = Wb1, gradZp = gradZ}
				support[node.childId[2]] = {W = Wb2, gradZp = gradZ}


			elseif #node.childId == 1 then
				local child = tree[node.childId[1]]
				gradWu:add(gradZ * child.feature:t())
				gradbu:add(gradZ)
	
				-- propagate to its children
				support[node.childId[1]] = {W = Wu, gradZp = gradZ}

			else error('accept only binary trees') end

		else -- leaf
			-- compute cost
			cost = cost + node.ecat

			-- compute gradZ
			local gradZCat = node.predict - node.cat
			local gradZ = (W:t() * gradZp):add(WCat:t() * gradZCat)

			-- compute gradient
			gradWCat:add(gradZCat * node.feature:t())
			gradbCat:add(gradZCat)
			gradL[{{},{node.label}}]:add(gradZ)
		end
	end
	
	return cost
end

--************************ compute cost and gradient *****************--
--input:
--output:
require 'parallel'

-- worker function 
function worker()

	require 'rnn'
	local data = parallel.parent:receive()

	local raw_treebank = data.raw_treebank
	local net = data.net
	local nSample = #raw_treebank
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
	for i = 1, nSample do
		local tree = Tree:create_from_string(raw_treebank[i])
		RNN.forward(net, tree)
		if not fw_only then
			cost = cost + RNN.backpropagate(net, tree, grad)
		end
		treebank[#treebank+1] = tree
	end
	print('time for child running ' .. timer:time().real) io.flush()

	local stats = nil
	if not fw_only then treebank = nil
	else
--[[
		trueCatCount = torch.zeros(treebank[1][1].nCat)
		correctCatCount = torch.zeros(trueCatCount:nElement())

		for i = 1,nSample do
			local tree = treebank[i]
			for _,node in ipairs(tree) do
				trueCatCount:add(node:cat)
				if (node:cat - node:ecat):sum() == 0 then
				end
			end
		end
]]
	end

	parallel.parent:send({
			cost = cost, 
			grad = rnn.fold(net, grad), 
			treebank = treebank, 
			stats = stats})
end
	
-- parent call
function parent(param)

	local raw_treebank = param.raw_treebank
	local nSample = #raw_treebank
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
			data.raw_treebank[j] = raw_treebank[id]
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

function RNN:computeCostAndGrad(raw_treebank, config, fw_only)
	
if NPROCESS > 1 then
	local param = {
		net = self,
		config = config,
		raw_treebank = raw_treebank,
		totalCost = 0,
		totalGrad = nil,
		fw_only = fw_only or false
	}


	local ok,err = pcall(parent, param)
	if not ok then 	print(err) parallel.close() end
	
	return param.totalCost, param.totalGrad, param.treebank, param.stats

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
	local nSample = #raw_treebank
	local treebank = {} 
	for i = 1,nSample  do
		local tree = Tree:create_from_string(raw_treebank[i])
		self:forward(tree)
		cost = cost + self:backpropagate(tree, grad)
		treebank[#treebank+1] = tree
	end

	return cost/nSample, self:fold(grad):div(nSample), treebank
end
end

-- check gradient
function RNN:checkGradient(raw_treebank, config)
	local epsilon = 1e-4
	local theta = 1e-8

	local dim = self.dim
	local wrdDicLen = self.wrdDicLen
	local nCat = self.nCat

	local Theta = self:fold()
	local _, gradTheta = self:computeCostAndGrad(raw_treebank, config)

	local n = Theta:nElement()
	for i = 1,n do
		local index = {{i}}
		Theta[index]:add(epsilon)
		self:unfold(Theta)
		local costPlus,_ = self:computeCostAndGrad(raw_treebank, config)
		
		Theta[index]:add(-2*epsilon)
		self:unfold(Theta)
		local costMinus,_ = self:computeCostAndGrad(raw_treebank, config)
		Theta[index]:add(epsilon)
		self:unfold(Theta)

		local diff = math.abs((costPlus - costMinus) / (2*epsilon) - gradTheta[i])
		print('diff ' .. diff)
	end
end

--***************************** eval **************************
function RNN:eval(raw_treebank)
	return 0
end

--******************************* train networks *************************
---- optFunc from 'optim' package
function RNN:train(raw_traintreebank, raw_testtreebank, batchSize, optFunc, optFuncState, config)
	local nSample = #raw_traintreebank
	local j = 0

	local iter = 1
	local timer = torch.Timer()
	
	print('accuracy = ' .. self:eval(raw_testtreebank)) io.flush()

	local function func(M)
		print('time for optim ' .. timer:time().real) io.flush()
		self:unfold(M)

		-- extract data
		local timer1 = torch.Timer()
		j = j + 1
		if j > nSample/batchSize then j = 1 end
		local raw_subtreebank = {}
		for k = 1,batchSize do
			raw_subtreebank[k] = raw_traintreebank[k+(j-1)*batchSize]
		end
		print('time to extract data ' .. timer1:time().real) io.flush()

		timer1 = torch.Timer()
		local cost, Grad = self:computeCostAndGrad(raw_subtreebank, config)
		print('time to compute cost & grad ' .. timer1:time().real) io.flush()

		-- for visualization
		if math.mod(iter,1) == 0 then
			print('--- iter: ' .. iter)
			print('cost: ' .. cost)
			io.flush()
		end
		if math.mod(iter,10) == 0 then
			print('accuracy = ' .. self:eval(raw_testtreebank))
			self:save('model.head.' .. math.floor(iter / 10))
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


--*********************************** main ******************************--
function test ()
	local struct = 
	local net = RNN:new()
	local t1 = '(3 (2 Yet) (3 (2 (2 the) (2 act)) (3 (4 (3 (2 is) (3 (2 still) (4 charming))) (2 here)) (2 .))))'
	local t2 = '(4 (2 (2 a) (2 (2 screenplay) (2 more))) (3 (4 ingeniously) (2 (2 constructed) (2 (2 (2 (2 than) (2 ``)) (2 Memento)) (2 '')))))'

	local config = {lambda = 1e-4, alpha = 0.2}
	net:checkGradient({t1},config)
end

-- WARNING: donot uncomment the line below!!!
test()
