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

	local net = {dim = dim, wrdDicLen = wrdDicLen, nCat = nCat}

	local r = math.sqrt(6 / (dim + dim))
	local rw = 4 * math.sqrt(6 / (dim + wrdDicLen))

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

function SFIORNNLM:extend_tree(storage, tree, next_word_id)
	storage.start_pos = storage.start_pos - 1
	storage.end_pos = storage.end_pos + 1

	if storage.start_pos < 1 or storage.end_pos > storage.n_nodes then 
		error('not enough space in storage')
	end
	
	local new_range = {storage.start_pos,storage.end_pos}
	tree.n_nodes = tree.n_nodes + 2

	tree.n_children = storage.n_children[{new_range}]
	tree.n_children[1] = 2	-- new root
	tree.n_children[tree.n_nodes] = 0 -- new (right-most) leaf

	tree.cover = storage.cover[{{},newrage}]
	local right_most_leaf_id = tree.cover[{2,2}]+1
	tree.cover[{1,1}] = 1
	tree.cover[{2,1}] = right_most_leaf_id
	tree.cover[{1,tree.n_nodes}] = right_most_leaf_id
	tree.cover[{2,tree.n_nodes}] = right_most_leaf_id

	tree.children_id = storage.children_id[{{},new_range}]
	tree.children_id:add(1)
	tree.children_id[{1,1}] = 2
	tree.children_id[{2,1}] = tree.n_nodes
	
	tree.parent_id:add(1)
	tree.parent_id = storage.parent_id[{new_range}]
	tree.parent_id[tree.n_nodes] = 1
	tree.parent_id[2] = 1

	tree.sibling_order = storage.sibling_order[{new_range}]
	tree.sibling_order[2] = 1
	tree.sibling_order[tree.n_nodes] = 2

	tree.word_id = storage.word_id[{new_range}]
	tree.word_id[tree.n_nodes] = next_word_id
	
	tree.inner = storage.inner[{{},new_range}]
	tree.gradZi = storage.gradZi[{{},new_range}]
	tree.gradZi:fill(0)

	return tree
end

-- compute inner meaning at the root node
function SFIORNNLM:forward_inside_root(tree)
	tree.inner[{{},{tree.n_nodes}}]:copy(self.L[{{},{tree.word_id[tree.n_nodes]}}])
	local input = self.Wi1 * tree.inner[{{},{2}}]
	input:add(self.Wi2 * tree.inner[{{},{tree.n_nodes}}])
	input:add(self.bi)
	tree.inner[{{},{1}}]:copy(self.func(input))
end

-- compute outer meaning at the right-most leaf
function SFIORNNLM:forward_outside_rml(tree)
	local input = self.Wop * self.root_outer
	input:add(self.Wo1 * tree.inner[{{},{2}}]):add(self.bo2)
	tree.rml_outer = self.func(input)
end

function SFIORNNLM:forward_compute_prediction_prob(tree)
	local z = (self.Ww * tree.rml_outer):add(self.bw)
	tree.prob = safe_compute_softmax(z)
	tree.error = -math.log(tree.prob[{tree.word_id[tree.n_nodes],1}])
end

-- backpropagation from the right-most leaf
function SFIORNNLM:backpropagate_outside_rml(tree, grad)
	local gZw = tree.prob:clone()
	gZw[{tree.word_id[tree.n_nodes],{}}]:add(-1)

	-- Ww, bw
	grad.bw:add(gZw)
	grad.Ww:add(gZw * tree.rml_outer:t())
	
	-- gradZo before multiplied by fprime
	gZo = (self.Ww:t() * gZw):cmul(self.funcPrime(tree.rml_outer))
	tree.gradZo_rml = gZo:clone()

	-- Wop, Wo, bo
	local gWop = gZo * self.root_outer:t()
	grad.Wop:add(gWop)

	grad.Wo1:add(gZo * tree.inner[{{},{2}}]:t())
	grad.bo2:add(gZo)

	-- root
	grad.root_outer:add(self.Wop:t() * gZo)
end

function SFIORNNLM:backpropagate_inside(tree, grad)
	for i = 2,tree.n_nodes do
		local col_i = {{},{i}}
		local parent_id = tree.parent_id[i]
		local gZi = torch.zeros(self.dim, 1)

		if i == 2 then
			gZi:add(self.Wo1:t() * tree.gradZo_rml)
		end
	
		if tree.sibling_order[i] == 1 then
			gZi:add(self.Wi1:t() * tree.gradZi[{{},{parent_id}}])
		else
			gZi:add(self.Wi2:t() * tree.gradZi[{{},{parent_id}}])
		end

		-- for internal node
		if tree.n_children[i] > 0 then	
			gZi:cmul(self.funcPrime(tree.inner[col_i]))
			tree.gradZi[col_i]:copy(gZi)

			-- weight matrices for inner
			grad.Wi1:add(gZi * tree.inner[{{},{tree.children_id[{1,i}]}}]:t())
			grad.Wi2:add(gZi * tree.inner[{{},{tree.children_id[{2,i}]}}]:t())
			grad.bi:add(gZi)

		else -- leaf
			tree.gradZi[col_i]:copy(gZi)
			if self.update_L then
				grad.L[{{},{tree.word_id[i]}}]:add(gZi)
			end
		end
	end
end

-- create storage for a sentence with n_tokens words 
function SFIORNNLM:create_storage_and_tree(n_tokens)
	local n_tokens = n_tokens or 200
	local n_nodes = 2*n_tokens - 1
	local storage = {
		n_children = torch.zeros(n_nodes),
		cover = torch.zeros(2, n_nodes),
		children_id = torch.zeros(2, n_nodes),
		parent_id = torch.zeros(n_nodes),
		sibling_order = torch.zeros(n_nodes),
		word_id = torch.zeros(n_nodes),
		inner = torch.zeros(self.dim, n_nodes),
		gradZi = torch.zeros(self.dim, n_nodes),
		start_pos = (n_nodes + 1)/2,
		end_pos = (n_nodes + 1)/2,
		n_nodes = n_nodes
	}

	local range = {storage.start_pos, storage.end_pos}
	local tree = {
		n_nodes = 1,
		n_children = storage.n_children[{range}],
		cover = storage.cover[{{},range}],
		children_id = storage.children_id[{{},range}],
		parent_id = storage.parent_id[{range}],
		sibling_order = storage.sibling_order[{range}],
		word_id = storage.word_id[{range}],
		inner = storage.inner[{{},range}],
		gradZi = storage.gradZi[{{},range}]
	}

	return storage, tree
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
	local cost = 0
	local n_cases = 0

	for i, sen in ipairs(senbank) do
		local storage, tree = self:create_storage_and_tree(#sen)
		tree.word_id[1] = sen[1] -- should be <s>
		tree.inner[{{},{1}}]:copy(self.L[{{},{tree.word_id[1]}}])
	
		for j = 2,#sen do
			self:extend_tree(storage, tree, sen[j])
			self:forward_inside_root(tree)
			self:forward_outside_rml(tree)
			self:forward_compute_prediction_prob(tree)

			cost = cost + tree.error
			self:backpropagate_outside_rml(tree, grad)
			self:backpropagate_inside(tree, grad)
		end
		n_cases = n_cases + #sen-1
	end

	local M = self:fold()
	grad = self:fold(grad)

	cost = cost / n_cases + config.lambda/2 * torch.pow(M,2):sum()
	grad:div(n_cases):add(M * config.lambda)

	if self.update_L then
		cost = cost + (config.lambda_L - config.lambda)/2 * 
						torch.pow(self.L,2):sum()
		grad[{{-self.L:nElement(),-1}}]:add(self.L * (config.lambda_L-config.lambda))
	end

	return cost, grad
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

		local subsenbank = {}
		for k = 1,batchSize do
			subsenbank[k] = senbank[k+(j-1)*batchSize]
		end
	
		local function func(M)
			self:unfold(M)

			-- extract data
			p:start("compute grad")
			cost, Grad = self:computeCostAndGrad(subsenbank, 
							{lambda = lambda.lambda, lambda_L=lambda.lambda_L}
						)
			p:lap("compute grad")

			-- for visualization
			print('iter ' .. j .. ': ' .. cost) io.flush()
	
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
		['PADDING'] = 18
	}

	require "dict"
	vocaDic = Dict:new(collobert_template)
	vocaDic.word2id = word2id

	senbank = {
		"PADDING Yet the act is still charming here . PADDING",
		"PADDING A screenplay is more ingeniously constructed than `` Memento '' . PADDING",
		"PADDING The act screenplay is more than here . PADDING"
	}

	require 'utils'
	for i = 1,#senbank do
		senbank[i] = split_string(senbank[i])
		for j,tok in ipairs(senbank[i]) do
			senbank[i][j] = vocaDic:get_id(tok)
		end
	end
	--print(senbank)


	struct = {Lookup = torch.randn(2,18), nCategory = 5, func = tanh, funcPrime = tanhPrime}
	net = SFIORNNLM:new(struct)

	--print(net)	

	config = {lambda = 1e-4, lambda_L = 1e-7}
	net.update_L = true
	net:checkGradient(senbank, config)
]]
