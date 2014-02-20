-- IORNNLM using softmax to compute p(w[t+1]|context)

require 'tree'
require 'utils'

NPROCESS = 1

--**************** rerursive neural network class ******************--
SFIORNNLM = {}
SFIORNNLM_mt = {__index = SFIORNNLM}

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
function SFIORNNLM:new(struct)

	local dim = struct.Lookup:size(1)
	local wrdDicLen = struct.Lookup:size(2)
	local nCat = struct.nCategory

	local net = {dim = dim, wrdDicLen = wrdDicLen, nCat = nCat, n_leaves = struct.n_leaves}

	local r = 0.1 --math.sqrt(6 / (dim + dim))
	local rw = 0.1 --4 * math.sqrt(6 / (dim + wrdDicLen))

	net.Wi1	= uniform(dim, dim, -r, r)	-- (inner) left child to parent
	net.Wi2	= uniform(dim, dim, -r, r)	-- (inner) right child to parent
	net.bi	= torch.zeros(dim, 1)-- (inner) bias at parent
	
	net.Wo1	= uniform(dim, dim, -r, r)	-- (outer) left child to right child
	net.bo2	= torch.zeros(dim, 1)		-- (outer) bias at right child
	net.Wop	= uniform(dim, dim, -r, r)	-- (outer) parent to child

	-- word prediction (softmax)
	net.Ww = uniform(wrdDicLen, dim, -rw, rw)
	net.bw = torch.zeros(wrdDicLen, 1)
	
	-- wordembedding
	net.L = struct.Lookup
	net.func = struct.func
	net.funcPrime = struct.funcPrime

	-- root outer
	net.root_outer = uniform(dim, 1, -1e-3, 1e-3)
	
	setmetatable(net, SFIORNNLM_mt)
	return net
end

-- save net into a bin file
function SFIORNNLM:save( filename , binary )
	local file = torch.DiskFile(filename, 'w')
	if binary == nil or binary then print(binary) file:binary() end
	file:writeObject(self)
	file:close()
end

-- create net from file
function SFIORNNLM:load( filename , binary, func, funcPrime )
	local file = torch.DiskFile(filename, 'r')
	if binary == nil or binary then file:binary() end
	local net = file:readObject()
	setmetatable(net, SFIORNNLM_mt)
	file:close()
	return net
end

-- fold parameters to a vector
function SFIORNNLM:fold( Model )
	local net = Model or self

	local Params = {
		net.Ww, net.bw,
		net.root_outer,
		net.Wi1, net.Wi2, net.bi,
		net.Wo1, net.Wop, net.bo2
	}
	
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
function SFIORNNLM:unfold(Theta)
	local Params = {
		self.Ww, self.bw,
		self.root_outer,
		self.Wi1, self.Wi2, self.bi,
		self.Wo1, self.Wop, self.bo2
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

-- create storage for a sentence with n_tokens words 
function SFIORNNLM:create_tree(sen)
	local n_words = #sen
	local n_nodes = 2*n_words - 1
	local tree = {
		n_children = torch.zeros(n_nodes),
		children_id = torch.zeros(2, n_nodes),
		parent_id = torch.zeros(n_nodes),
		sibling_order = torch.zeros(n_nodes),
		word_id = torch.zeros(n_nodes),
		inner = torch.zeros(self.dim, n_nodes),
		n_nodes = n_nodes
	}

	tree.n_children[{{1,n_words-1}}]:fill(2)
	tree.children_id[{1,{1,n_words-1}}]:copy(torch.linspace(2,n_words,n_words-1))
	tree.children_id[{2,{1,n_words-1}}]:copy(-torch.linspace(n_words+1,n_nodes,n_words-1)+3*n_words)
	tree.parent_id[{{2,n_words}}]:copy(torch.linspace(1,n_words-1,n_words-1))
	tree.parent_id[{{n_words+1,-1}}]:copy(-torch.linspace(1,n_words-1,n_words-1)+n_words)
	tree.sibling_order[{{2,n_words}}]:fill(1)
	tree.sibling_order[{{n_words+1,-1}}]:fill(2)
	tree.word_id[{{n_words,-1}}]:copy(torch.Tensor(sen))

--[[
	print(sen)
	for name,value in pairs(tree) do
		print(name)
		print(value)
	end
]]

	return tree
end

function SFIORNNLM:build_treeletbank(treebank)
	local n_treelets = 0
	local n_leaves = self.n_leaves
	for i,tree in ipairs(treebank) do
		n_treelets = n_treelets + (tree.n_nodes + 1)/2 - n_leaves + 1
	end

	local n_nodes = n_leaves * 2 - 1
	local n_children = torch.zeros(n_nodes)
	n_children[{{1,n_leaves-1}}]:fill(2)
	
	local children_id = torch.zeros(2, n_nodes)
	children_id[{1,{1,n_leaves-1}}]:copy(torch.linspace(2,n_leaves,n_leaves-1))
	children_id[{2,{1,n_leaves-1}}]:copy(-torch.linspace(n_leaves+1,2*n_leaves-1,n_leaves-1) + 3*n_leaves)
	
	local parent_id = torch.zeros(n_nodes)
	parent_id[{{2,n_leaves}}]:copy(torch.linspace(1,n_leaves-1,n_leaves-1))
	parent_id[{{n_leaves+1,-1}}]:copy(-torch.linspace(1,n_leaves-1,n_leaves-1)+n_leaves)

	local sibling_order = torch.zeros(n_nodes)
	sibling_order[{{2,n_leaves}}]:fill(1)
	sibling_order[{{n_leaves+1,-1}}]:fill(2)

	local word_id = torch.zeros(n_nodes, n_treelets)
	local inner = torch.zeros(self.dim, n_nodes, n_treelets)
	local gradZi = torch.zeros(self.dim, n_nodes, n_treelets)
	
	local j = 1
	for _,tree in ipairs(treebank) do
		for i = 1,(tree.n_nodes+1)/2-n_leaves+1 do
			inner[{{},{1,n_leaves},j}]:copy(tree.inner[{{},{i,i+n_leaves-1}}])
			local start = tree.n_nodes-i-n_leaves+3
			inner[{{},{n_leaves+1,-1},j}]:copy(tree.inner[{{},{start,start+n_leaves-2}}])
			word_id[{n_leaves,j}] = tree.word_id[i+n_leaves-1]
			word_id[{{n_leaves+1,-1},j}]:copy(tree.word_id[{{start,start+n_leaves-2}}])
			j = j + 1
		end
	end	

	local treeletbank = {
		n_treelets = n_treelets,
		n_nodes = n_nodes,
		n_leaves = n_leaves,
		n_children = n_children,
		children_id = children_id,
		parent_id = parent_id,
		sibling_order = sibling_order,
		word_id = word_id,
		inner = inner,
		gradZi = gradZi
	}

	return treeletbank
end

-- compute inner meaning
function SFIORNNLM:forward_inside(tree)
	for i = tree.n_nodes,1,-1 do
		local col_i = {{},{i}}
		if tree.n_children[i] == 0 then
			--print(col_i)
			--print(tree.word_id[i])
			tree.inner[col_i]:copy(self.L[{{},{tree.word_id[i]}}])
		else
			local input = self.Wi1 * tree.inner[{{},{tree.children_id[{1,i}]}}]
			input:add(self.Wi2 * tree.inner[{{},{tree.children_id[{2,i}]}}])
			input:add(self.bi)
			tree.inner[col_i]:copy(self.func(input))
		end
	end
end

-- compute outer meaning at the right-most leaf
function SFIORNNLM:forward_outside_rml(treeletbank)
	local input = (self.Wo1 * treeletbank.inner[{{},2,{}}])
					:add(torch.repeatTensor(self.bo2 + self.Wop * self.root_outer,
											1,treeletbank.n_treelets))				
	treeletbank.rml_outer = self.func(input)
end

function SFIORNNLM:forward_compute_prediction_prob(treeletbank)
	local z = (self.Ww * treeletbank.rml_outer)
				:add(torch.repeatTensor(self.bw, 1, treeletbank.n_treelets))
	treeletbank.prob = safe_compute_softmax(z)

	treeletbank.error = torch.zeros(treeletbank.n_treelets)
	for i = 1,treeletbank.n_treelets do
		treeletbank.error[i] = -math.log(treeletbank.prob[{treeletbank.word_id[{treeletbank.n_nodes,i}],i}])
	end
end

-- backpropagation from the right-most leaf
function SFIORNNLM:backpropagate_outside_rml(treeletbank, grad)
	local gZw = treeletbank.prob:clone()

	for i = 1,treeletbank.n_treelets do
		gZw[{treeletbank.word_id[{treeletbank.n_nodes,i}],{i}}]:add(-1)
	end

	-- Ww, bw
	grad.bw:add(gZw:sum(2))
	grad.Ww:add(gZw * treeletbank.rml_outer:t())
	
	-- gradZo before multiplied by fprime
	gZo = (self.Ww:t() * gZw):cmul(self.funcPrime(treeletbank.rml_outer))
	treeletbank.gradZo_rml = gZo:clone()

	-- Wop, Wo, bo
	local gWop = gZo * torch.repeatTensor(self.root_outer,1,treeletbank.n_treelets):t()
	grad.Wop:add(gWop)

	grad.Wo1:add(gZo * treeletbank.inner[{{},2,{}}]:t())
	grad.bo2:add(gZo:sum(2))

	-- root
	grad.root_outer:add((self.Wop:t() * gZo):sum(2))
end

function SFIORNNLM:backpropagate_inside(treeletbank, grad)
	for i = 2,treeletbank.n_nodes do
		local col_i = {{},i,{}}
		local parent_id = treeletbank.parent_id[i]
		local gZi = torch.zeros(self.dim, treeletbank.n_treelets)

		if i == 2 then
			gZi:add(self.Wo1:t() * treeletbank.gradZo_rml)
		end
	
		if treeletbank.sibling_order[i] == 1 then
			gZi:add(self.Wi1:t() * treeletbank.gradZi[{{},parent_id,{}}])
		else
			gZi:add(self.Wi2:t() * treeletbank.gradZi[{{},parent_id,{}}])
		end

		-- for internal node
		if treeletbank.n_children[i] > 0 then	
			gZi:cmul(self.funcPrime(treeletbank.inner[col_i]))
			treeletbank.gradZi[col_i]:copy(gZi)

			-- weight matrices for inner
			grad.Wi1:add(gZi * treeletbank.inner[{{},treeletbank.children_id[{1,i}],{}}]:t())
			grad.Wi2:add(gZi * treeletbank.inner[{{},treeletbank.children_id[{2,i}],{}}]:t())
			grad.bi:add(gZi:sum(2))

		else -- leaf
			treeletbank.gradZi[col_i]:copy(gZi)
			if self.update_L then
				for j = 1,treeletbank.n_treelets do
					local word_id = treeletbank.word_id[{i,j}]
					if word_id > 0 then 
						grad.L[{{},{word_id}}]:add(gZi[{{},{j}}])
					end
				end
			end
		end
	end
end

function SFIORNNLM:computeCostAndGrad(senbank, config)
	local parse = config.parse or false
	
if NPROCESS > 1 then
else
	-- create zero grad
	local grad = {}
	grad.Wi1 = torch.zeros(self.Wi1:size())
	grad.Wi2 = torch.zeros(self.Wi2:size())
	grad.bi = torch.zeros(self.bi:size())
	grad.Wo1 = torch.zeros(self.Wo1:size())
	grad.Wop = torch.zeros(self.Wop:size())
	grad.bo2 = torch.zeros(self.bo2:size())

	grad.Ww = torch.zeros(self.Ww:size())
	grad.bw = torch.zeros(self.bw:size())

	if self.update_L then
		grad.L = torch.zeros(self.L:size())
	end
	grad.root_outer = torch.zeros(self.root_outer:size())

	-- process the senbank
	local treebank = {}
	for i, sen in ipairs(senbank) do
		local tree = self:create_tree(sen)
		self:forward_inside(tree)
		treebank[i] = tree
	end
	local treeletbank = self:build_treeletbank(treebank)
	self:forward_outside_rml(treeletbank)
	self:forward_compute_prediction_prob(treeletbank)
	self:backpropagate_outside_rml(treeletbank, grad)
	self:backpropagate_inside(treeletbank, grad)

	local M = self:fold()
	grad = self:fold(grad)

	cost = treeletbank.error:sum() / treeletbank.n_treelets + config.lambda/2 * torch.pow(M,2):sum()
	grad:div(treeletbank.n_treelets):add(M * config.lambda)

	if self.update_L then
		cost = cost + (config.lambda_L - config.lambda)/2 * 
						torch.pow(self.L,2):sum()
		grad[{{-self.L:nElement(),-1}}]:add(self.L * (config.lambda_L-config.lambda))
	end

	return cost, grad, treeletbank.n_treelets
end
end

-- check gradient
function SFIORNNLM:checkGradient(senbank, config)
	local epsilon = 1e-4

	local dim = self.dim
	local wrdDicLen = self.wrdDicLen
	local nCat = self.nCat

	local Theta = self:fold()
	local _, gradTheta = self:computeCostAndGrad(senbank, config)

	local n = Theta:nElement()
	local numGradTheta = torch.zeros(n)
	for i = 1,n do
		local index = {{i}}
		Theta[index]:add(epsilon)
		self:unfold(Theta)
		local costPlus,_ = self:computeCostAndGrad(senbank, config)
		
		Theta[index]:add(-2*epsilon)
		self:unfold(Theta)
		local costMinus,_ = self:computeCostAndGrad(senbank, config)
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


--**************************** training ************************--
require 'optim'
require 'xlua'
p = xlua.Profiler()

function SFIORNNLM:train_with_adagrad(senbank, batchSize, maxepoch, lambda, prefix,
									adagrad_config, adagrad_state)
	local nSample = #senbank
	
	local epoch = 0
	local j = 1e10
	local log_prob = 0
	local ncases = 0

	while true do
		j = j + 1

		-- new epoch
		if j > nSample/batchSize then 
			self:save(prefix .. '_' .. epoch)
			j = 1
			epoch = epoch + 1
			log_prob = 0
			ncases = 0
			if epoch > maxepoch then break end
			print('===== epoch ' .. epoch .. '=====')
		end

		local subsenbank = {}
		for k = 1,batchSize do
			subsenbank[k] = senbank[k+(j-1)*batchSize]
		end
	
		local function func(M)
			self:unfold(M)

			-- extract data
			p:start("compute grad")
			cost, Grad, n_treelets = self:computeCostAndGrad(subsenbank, 
							{lambda = lambda.lambda, lambda_L=lambda.lambda_L}
						)
			p:lap("compute grad")

			-- for visualization
			log_prob = log_prob - cost * n_treelets
			ncases = ncases + n_treelets
			print('iter ' .. j .. ': ' .. cost) io.flush()
			print('entropy ' .. math.log(math.exp(-log_prob/ncases)) / math.log(2))

			-- cut-off
			--Grad[torch.gt(Grad,0.01)] = 0.01
			--Grad[torch.lt(Grad,-0.01)] = -0.01

			return cost, Grad
		end

		p:start("optim")
		M,_ = optim.adagrad(func, self:fold(), adagrad_config, adagrad_state)
--[[		
		local M = self:fold()
		cost,Grad = func(M)
		M:add(Grad:mul(-adagrad_config.learningRate))
]]
		self:unfold(M)
		p:lap("optim")
		p:printAll()

		collectgarbage()
	end

	return adagrad_config, adagrad_state
end

function SFIORNNLM:parse(senbank)
	error('not implement yet')
	local old_uL = self.update_L
	self.update_L = false
	_, _, senbank = self:computeCostAndGrad(senbank, {parse=true})
	self.update_L = old_uL
	return senbank
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
		["''"] = 17, 
		['<s>'] = 18,
		['</s>'] = 19
	}

	require "dict"
	vocaDic = Dict:new(collobert_template)
	vocaDic.word2id = word2id

	struct = {Lookup = torch.randn(2,19), nCategory = 5, func = tanh, funcPrime = tanhPrime}
	net = SFIORNNLM:new(struct)
	
	n_leaves = 4

	senbank = {
		"<s> <s> <s> Yet the act is still charming here . </s>",
		"<s> <s> <s> A screenplay is more ingeniously constructed than `` Memento '' . </s>",
		"<s> <s> <s> The act screenplay is more than here . </s>"
	}

	require 'utils'
	for i = 1,#senbank do
		senbank[i] = split_string(senbank[i])
		local sen = {}
		for j,tok in ipairs(senbank[i]) do
			sen[j] = vocaDic:get_id(tok)
		end
		senbank[i] = sen
	end

	config = {lambda = 0*1e-4, lambda_L = 0*1e-7, n_leaves = 12}
	net.update_L = true
	net:checkGradient(senbank, config)
]]


