require 'depstruct'
require 'utils'
require 'dict'
require 'dp_spec'

require 'xlua'
p = xlua.Profiler()

--**************** inside-outside rerursive neural network class ******************--
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
	return torch.ones(X.size)
end

-- soft sign
function softsign(X)
	return torch.cdiv(X, torch.abs(X):add(1))
end

function softsignPrime(softsignX)
	return torch.abs(softsignX):mul(-1):add(1):pow(2)
end

IORNN.default_func = softsign
IORNN.default_funcPrime = softsignPrime

--************************* construction ********************--

function IORNN:new(input)
	local net = {	dim = input.dim, wdim = input.lookup:size(1), sdim = input.sdim,
					n_prevtrees = input.n_prevtrees,
					voca_dic = input.voca_dic, pos_dic = input.pos_dic, deprel_dic = input.deprel_dic, 
					complete_inside = input.complete_inside }
	net.func = input.func or IORNN.default_func
	net.funcPrime = input.funcPrime or IORNN.default_funcPrime

	print('----------------- NET INFO -------------------')
	print(net)
	print('----------------------------------------------')

	setmetatable(net, IORNN_mt)

	net:init_params(input)
	return net
end

function IORNN:create_weight_matrix(params, index, size1, size2, r)
	local W = nil
	if r then 
		W = params[{{index,index+size1*size2-1}}]:resize(size1,size2):copy(uniform(size1,size2,-r,r))
	else 
		W = params[{{index,index+size1*size2-1}}]:resize(size1,size2)
	end
	return W, index+size1*size2
end

function IORNN:init_params(input)
	local dim	 = self.dim
	local wdim	 = self.wdim
	local sdim	 = self.sdim
	local voca_dic	 = self.voca_dic
	local deprel_dic = self.deprel_dic
	local pos_dic	 = self.pos_dic

	-- create params
	local n_params = 	dim + dim + self.n_prevtrees*dim*dim + 
						dim * (wdim + 1) + 
						2 * dim +
						dim * dim + dim + 
						2 * ( 	dim + 
								dim*dim * (2 + 2*deprel_dic.size) + dim 
							) + 
						(deprel_dic.size + 1) * (dim + 1)  +  
						pos_dic.size 	* (dim + deprel_dic.size + 1) + 
						voca_dic.size 	* (dim + deprel_dic.size + pos_dic.size + 1) + 
						N_CAP_FEAT 		* (dim + deprel_dic.size + pos_dic.size + voca_dic.size + 1) + 
						--N_DIST_FEAT 	* (dim + deprel_dic.size + pos_dic.size + voca_dic.size + N_CAP_FEAT + 1) + 
						pos_dic.size * dim + 
						N_CAP_FEAT * dim + 
						voca_dic.size * wdim
			
	self.params = torch.zeros(n_params)

	--%%%%%%%%%%%%% assign ref %%%%%%%%%%%%%
	local r = 0.1
	local r_small = 1e-3
	local r_tiny = 1e-5

	local index = 1

	-- contextual trees
	self.root_complete_inner, index = self:create_weight_matrix(self.params, index, dim, 1, r)
	self.Wctx_trees = {}
	self.bctx_trees, index = self:create_weight_matrix(self.params, index, dim, 1)
	for i = 1, self.n_prevtrees do
		self.Wctx_trees[i], index = self:create_weight_matrix(self.params, index, dim, dim, math.sqrt(6/(dim+dim)))
	end

--	print(index)
	-- project word embs on to a higher-dim vector space
	self.Wh, index = self:create_weight_matrix(self.params, index, dim, wdim, math.sqrt(6/(wdim+dim)))
	self.bh, index = self:create_weight_matrix(self.params, index, dim, 1)
	
	-- anonymous outer/inner
	self.root_inner, index = self:create_weight_matrix(self.params, index, dim, 1, r)
	self.anon_outer, index = self:create_weight_matrix(self.params, index, dim, 1, r)

	-- weights for combining head	
	self.Wih, index = self:create_weight_matrix(self.params, index, dim, dim, math.sqrt(6/(dim+dim)))
	self.bi, index = self:create_weight_matrix(self.params, index, dim, 1)

	for _,d in ipairs({DIR_L, DIR_R}) do
		self[d] = {}
		local dir = self[d]

		print(index)
		dir.anon_inner, index = self:create_weight_matrix(self.params, index, dim, 1, r)

		-- composition weight matrices
		dir.Wi = {}
		dir.Wo = {}
		for i = 1,deprel_dic.size do
--			print(index .. ' ' .. deprel_dic.id2word[i] .. ' ' .. d)
			dir.Wi[i], index = self:create_weight_matrix(self.params, index, dim, dim, r_small)
			dir.Wo[i], index = self:create_weight_matrix(self.params, index, dim, dim, math.sqrt(6/(dim+dim+pos_dic.size+N_CAP_FEAT+deprel_dic.size)))
		end
		dir.Woh, index = self:create_weight_matrix(self.params, index, dim, dim, math.sqrt(6/(dim+dim+pos_dic.size+N_CAP_FEAT+deprel_dic.size)))
		dir.Wop, index = self:create_weight_matrix(self.params, index, dim, dim, math.sqrt(6/(dim+dim+pos_dic.size+N_CAP_FEAT+deprel_dic.size)))

		dir.bo, index = self:create_weight_matrix(self.params, index, dim, 1)
	end


	-- Pr(deprel | outer, dir)
	self.Wdr, index = self:create_weight_matrix(self.params, index, deprel_dic.size+1, dim, math.sqrt(1/dim)) -- +1 for EOC
	self.bdr, index = self:create_weight_matrix(self.params, index, deprel_dic.size+1, 1)

	-- Pr(POS | deprel, outer, dir)
	self.Wpos, index	= self:create_weight_matrix(self.params, index, pos_dic.size, dim, math.sqrt(1/dim))
	self.Ldrpos, index	= self:create_weight_matrix(self.params, index, pos_dic.size, deprel_dic.size, r)
	self.bpos, index	= self:create_weight_matrix(self.params, index, pos_dic.size, 1)

	-- Pr(word | POS, deprel, outer, dir) -- note: #internal_nodes = #leaves - 1 = voca_dic.size - 1
	-- move this to before self.L
		-- self.Wword, index		= self:create_weight_matrix(self.params, index, voca_dic.size, dim, r)
	self.Ldrword, index		= self:create_weight_matrix(self.params, index, voca_dic.size, deprel_dic.size, r)
	self.Lposword, index	= self:create_weight_matrix(self.params, index, voca_dic.size, pos_dic.size, r)
	self.bword, index		= self:create_weight_matrix(self.params, index, voca_dic.size, 1)

	-- Pr(cap | word, POS, deprel, outer, dir)
	self.Wcap, index		= self:create_weight_matrix(self.params, index, N_CAP_FEAT, dim, math.sqrt(1/dim))
	self.Ldrcap, index		= self:create_weight_matrix(self.params, index, N_CAP_FEAT, deprel_dic.size, r)
	self.Lposcap, index		= self:create_weight_matrix(self.params, index, N_CAP_FEAT, pos_dic.size, r)
	self.Lwordcap, index 	= self:create_weight_matrix(self.params, index, N_CAP_FEAT, voca_dic.size, r)
	self.bcap, index		= self:create_weight_matrix(self.params, index, N_CAP_FEAT, 1)

	--[[ Pr(dist_to_head | cap, word, POS, ...)
	self.Wdist, index	 	= self:create_weight_matrix(self.params, index, N_DIST_FEAT, dim, r)
	self.Ldrdist, index	 	= self:create_weight_matrix(self.params, index, N_DIST_FEAT, deprel_dic.size, r)
	self.Lposdist, index	= self:create_weight_matrix(self.params, index, N_DIST_FEAT, pos_dic.size, r)
	self.Lworddist, index 	= self:create_weight_matrix(self.params, index, N_DIST_FEAT, voca_dic.size, r)
	self.Lcapdist, index	= self:create_weight_matrix(self.params, index, N_DIST_FEAT, N_CAP_FEAT, r)
	self.bdist, index	 	= self:create_weight_matrix(self.params, index, N_DIST_FEAT, 1)
]]
	-- POS tag 
	self.Lpos, index = self:create_weight_matrix(self.params, index, dim, pos_dic.size, r)

	-- capital letter feature
	self.Lcap, index = self:create_weight_matrix(self.params, index, dim, N_CAP_FEAT, r)

	-- for hierarchical softmax
	self.Wword, index = self:create_weight_matrix(self.params, index, voca_dic.size, dim, math.sqrt(6/(dim+dim)))

	--  word embeddings (always always always at the end of the array of params)
	self.L = self.params[{{index,index+voca_dic.size*wdim-1}}]:resize(wdim,voca_dic.size):copy(input.lookup)	-- word embeddings 
	index = index + voca_dic.size*wdim

	if index -1 ~= n_params then error('size not match ' .. (index-1) .. ' ' .. n_params) end
end

function IORNN:create_grad()
	local grad = {}
	local dim = self.dim
	local wdim = self.wdim
	local sdim = self.sdim
	local voca_dic = self.voca_dic
	local deprel_dic = self.deprel_dic
	local pos_dic = self.pos_dic

	grad.params = torch.zeros(self.params:numel())

	--%%%%%%%%%%%%% assign ref %%%%%%%%%%%%%
	local index = 1

	-- contextual trees
	grad.root_complete_inner, index = self:create_weight_matrix(grad.params, index, dim, 1) -- anonymous root complete inner
	grad.Wctx_trees = {}
	grad.bctx_trees, index = self:create_weight_matrix(grad.params, index, dim, 1)
	for i = 1, self.n_prevtrees do
		grad.Wctx_trees[i], index = self:create_weight_matrix(grad.params, index, dim, dim)
	end

	-- project word embs on to a higher-dim vector space
	grad.Wh, index = self:create_weight_matrix(grad.params, index, dim, wdim)	
	grad.bh, index = self:create_weight_matrix(grad.params, index, dim, 1)
	
	-- anonymous outer/inner
	grad.root_inner, index = self:create_weight_matrix(grad.params, index, dim, 1)
	grad.anon_outer, index = self:create_weight_matrix(grad.params, index, dim, 1)

	grad.Wih, index = self:create_weight_matrix(grad.params, index, dim, dim)
	grad.bi, index = self:create_weight_matrix(grad.params, index, dim, 1)

	for _,d in ipairs({DIR_L, DIR_R}) do
		grad[d] = {}
		local dir = grad[d]
	
		dir.anon_inner, index = self:create_weight_matrix(grad.params, index, dim, 1)

		-- composition weight matrices
		dir.Wi = {}
		dir.Wo = {}
		for i = 1,deprel_dic.size do
			dir.Wi[i], index = self:create_weight_matrix(grad.params, index, dim, dim)
			dir.Wo[i], index = self:create_weight_matrix(grad.params, index, dim, dim)
		end
		dir.Woh, index = self:create_weight_matrix(grad.params, index, dim, dim)
		dir.Wop, index = self:create_weight_matrix(grad.params, index, dim, dim)
		dir.bo, index = self:create_weight_matrix(grad.params, index, dim, 1)
	end

	-- Pr(deprel | outer)
	grad.Wdr, index = self:create_weight_matrix(grad.params, index, deprel_dic.size+1, dim)
	grad.bdr, index = self:create_weight_matrix(grad.params, index, deprel_dic.size+1, 1)

	-- Pr(POS | deprel, outer)
	grad.Wpos, index	= self:create_weight_matrix(grad.params, index, pos_dic.size, dim)
	grad.Ldrpos, index	= self:create_weight_matrix(grad.params, index, pos_dic.size, deprel_dic.size)
	grad.bpos, index	= self:create_weight_matrix(grad.params, index, pos_dic.size, 1)

	-- Pr(word | POS, deprel, outer)
	-- move this down
		-- grad.Wword, index		= self:create_weight_matrix(grad.params, index, voca_dic.size, dim)
	grad.Ldrword, index		= self:create_weight_matrix(grad.params, index, voca_dic.size, deprel_dic.size)
	grad.Lposword, index	= self:create_weight_matrix(grad.params, index, voca_dic.size, pos_dic.size)
	grad.bword, index		= self:create_weight_matrix(grad.params, index, voca_dic.size, 1)

	-- Pr(cap | word, POS, deprel, outer, dir)
	grad.Wcap, index		= self:create_weight_matrix(grad.params, index, N_CAP_FEAT, dim)
	grad.Ldrcap, index		= self:create_weight_matrix(grad.params, index, N_CAP_FEAT, deprel_dic.size)
	grad.Lposcap, index		= self:create_weight_matrix(grad.params, index, N_CAP_FEAT, pos_dic.size)
	grad.Lwordcap, index 	= self:create_weight_matrix(grad.params, index, N_CAP_FEAT, voca_dic.size)
	grad.bcap, index		= self:create_weight_matrix(grad.params, index, N_CAP_FEAT, 1)

	--[[ Pr(dist_to_head | cap, word, POS, ...)
	grad.Wdist, index	 	= self:create_weight_matrix(grad.params, index, N_DIST_FEAT, dim)
	grad.Ldrdist, index	 	= self:create_weight_matrix(grad.params, index, N_DIST_FEAT, deprel_dic.size)
	grad.Lposdist, index	= self:create_weight_matrix(grad.params, index, N_DIST_FEAT, pos_dic.size)
	grad.Lworddist, index 	= self:create_weight_matrix(grad.params, index, N_DIST_FEAT, voca_dic.size)
	grad.Lcapdist, index	= self:create_weight_matrix(grad.params, index, N_DIST_FEAT, N_CAP_FEAT)
	grad.bdist, index	 	= self:create_weight_matrix(grad.params, index, N_DIST_FEAT, 1)
]]
	-- POS tag 
	grad.Lpos, index = self:create_weight_matrix(grad.params, index, dim, pos_dic.size)

	-- capital letter feature
	grad.Lcap, index = self:create_weight_matrix(grad.params, index, dim, N_CAP_FEAT)

	--
	grad.Wword, index = self:create_weight_matrix(grad.params, index, voca_dic.size, dim)

	--  word embeddings (always always always at the end of the array of params)
	grad.L = grad.params[{{index,index+voca_dic.size*wdim-1}}]:resize(wdim,voca_dic.size)	-- word embeddings 
	index = index + voca_dic.size*wdim

	if index-1 ~= grad.params:numel() then
		error('index not match')
	end

	grad.params:fill(0)
	return grad
end

-- save net into a file
function IORNN:save( filename , binary )
	local file = torch.DiskFile(filename, 'w')
	if binary == true then file:binary() end

	file:writeObject(self)
	file:close()
end

-- create net from file
function IORNN:load( filename , binary, func, funcPrime )
	local file = torch.DiskFile(filename, 'r')
	if binary == true then file:binary() end

	local net = file:readObject()
	file:close()

	setmetatable(net, IORNN_mt)
	setmetatable(net.voca_dic, Dict_mt)
	setmetatable(net.pos_dic, Dict_mt)
	setmetatable(net.deprel_dic, Dict_mt)
	if net.n_prevtrees == nil then 
		net.n_prevtrees = #net.Wctx_trees
	elseif net.n_prevtrees ~= #net.Wctx_trees then
		error('not match #prev-trees')
	end
	return net
end


--************************ forward **********************--

function IORNN:forward_inside(tree)
	if tree.inner == nil then
		tree.inner = torch.zeros(self.dim, tree.n_nodes)
		tree.complete_inner = torch.zeros(self.dim, tree.n_nodes)
	else
		tree.inner:fill(0)
		tree.complete_inner:fill(0)
	end

	-- computing inners for words
	local input = (self.Wh * self.L:index(2, tree.word:long()))
					:add(self.Lpos:index(2, tree.pos:long()))
					:add(self.Lcap:index(2, tree.cap:long()))
					:add(torch.repeatTensor(self.bh, 1, tree.n_nodes))

	tree.inner:copy(self.func(input))
	tree.inner[{{},{1}}]:copy(self.root_inner)

	-- computing complete inners for 'phrases'
	if self.complete_inner ~= CMPL_INSIDE_NONE then
		for i = tree.n_nodes,1,-1 do
			local col_i = {{},{i}}

			local n_children = tree[DIR_L].n_children[i] + tree[DIR_R].n_children[i]
			-- if this is a leaf
			if n_children == 0 then
				tree.complete_inner[col_i]:copy(tree.inner[col_i])

			-- else, compute [phrase] inner
			else
				local input_kids = torch.zeros(self.dim, 1)
		
				for _,dir in ipairs({DIR_L, DIR_R}) do 
					for j = 1,tree[dir].n_children[i] do
						local child = tree[dir].children[{j,i}]
						input_kids:addmm(self[dir].Wi[tree.deprel[child]], tree.complete_inner[{{},{child}}])
					end
				end
				input_kids:div(n_children)
				tree.complete_inner[col_i]:copy(self.func((self.Wih * tree.inner[col_i]):add(input_kids):add(self.bi)))
			end
		end
	end
end

function clean_tree(tree) 
	tree.inner = nil
	tree.complete_inner = nil
	tree.outer = nil
	tree.cstr_outer = nil
	tree[DIR_L].EOC_outer = nil
	tree[DIR_R].EOC_outer = nil
	tree.gradi = nil
	tree.gradcomplete_i = nil
end

function IORNN:forward_outside(tree, ctx_trees, complete_inside)
	tree.ctx_trees = ctx_trees

	if tree.outer == nil then 
		tree.outer = torch.zeros(self.dim, tree.n_nodes)
		tree.cstr_outer = torch.zeros(self.dim, tree.n_nodes) -- outer rep. during construction
		tree[DIR_L].EOC_outer = torch.zeros(self.dim, tree.n_nodes)
		tree[DIR_R].EOC_outer = torch.zeros(self.dim, tree.n_nodes)
	else
		tree.outer		:fill(0)
		tree.cstr_outer	:fill(0)
		tree[DIR_L].EOC_outer:fill(0)
		tree[DIR_R].EOC_outer:fill(0)
	end

	for i = 1,tree.n_nodes do
		local col_i = {{},{i}}

		-- compute full outer
		if i == 1 then -- ROOT
			if #ctx_trees == 0 then 
				tree.outer[col_i] = self.anon_outer
			else 
				local input = torch.zeros(self.dim, 1)
				for j,ctx_tree in ipairs(ctx_trees) do
					if ctx_tree == 0 then
						input:addmm(self.Wctx_trees[j], self.root_complete_inner)
					else
						input:addmm(self.Wctx_trees[j], ctx_tree.complete_inner[{{},{1}}])
					end
				end
				input:add(self.bctx_trees)
				tree.outer[col_i] = self.func(input)
			end

		else
			local parent = tree.parent[i]
			local dir	 = tree.dir[i]
			local input_parent = 	(self[dir].Woh * tree.inner[{{},{parent}}])
									:addmm(self[dir].Wop, tree.outer[{{},{parent}}])
									:add(self[dir].bo)
			local n_sister = tree[DIR_L].n_children[parent] + tree[DIR_R].n_children[parent] - 1
			if n_sister == 0 then
				tree.outer[col_i] = self.func(input_parent:add(self[dir].anon_inner))
			else
				local input = torch.zeros(self.dim, 1)
				for _,sis_dir in ipairs({DIR_L, DIR_R}) do
					for j = 1, tree[sis_dir].n_children[parent] do
						local sister = tree[sis_dir].children[{j,parent}]
						if sister < i then -- the fragment rooted at this sister node is complete
							if complete_inside == CMPL_INSIDE_LEFT2RIGHT then
								input:addmm(self[sis_dir].Wo[tree.deprel[sister]], tree.complete_inner[{{},{sister}}])
							else
								input:addmm(self[sis_dir].Wo[tree.deprel[sister]], tree.inner[{{},{sister}}])
							end
						elseif sister > i then -- this fragement rooted at this sister node is not complete
							if complete_inside == CMPL_INSIDE_RIGHT2LEFT then
								input:addmm(self[sis_dir].Wo[tree.deprel[sister]], tree.complete_inner[{{},{sister}}])
							else
								input:addmm(self[sis_dir].Wo[tree.deprel[sister]], tree.inner[{{},{sister}}])
							end
						end
					end
				end
				tree.outer[col_i] = self.func(input_parent:add(input:div(n_sister)))
			end
		end
	
		-- compute children's constr. outers and EOC outer
		local input	= torch.zeros(self.dim, 1)
		local n_left = 0

			-- left
			local input_head = 	(self[DIR_L].Woh * tree.inner[col_i])
								:addmm(self[DIR_L].Wop, tree.outer[col_i])
								:add(self[DIR_L].bo)

			if tree[DIR_L].n_children[i] == 0 then
				tree[DIR_L].EOC_outer[col_i] = self.func(input_head+self[DIR_L].anon_inner)
				input:copy(self[DIR_L].anon_inner)
				n_left = 1
			else 
				local left_sister	= nil
			
				-- compute outer rep. for its children
				for j = 1, tree[DIR_L].n_children[i] do
					local child = tree[DIR_L].children[{j,i}]
					local col_c = {{},{child}}

					-- compute constructed outer
					if left_sister  then 
						input:addmm(self[DIR_L].Wo[tree.deprel[left_sister]], tree.inner[{{},{left_sister}}])
						tree.cstr_outer[col_c] = self.func(torch.div(input, j-1):add(input_head))
					else 
						tree.cstr_outer[col_c] = self.func(input_head + self[DIR_L].anon_inner)
					end
					left_sister = child			
				end

				-- compute outer rep. for EOC
				n_left = tree[DIR_L].n_children[i]
				input:addmm(self[DIR_L].Wo[tree.deprel[left_sister]], tree.inner[{{},{left_sister}}])
				tree[DIR_L].EOC_outer[col_i] = self.func(torch.div(input,n_left):add(input_head))
			end

		-- right
			local input_head = 	(self[DIR_R].Woh * tree.inner[col_i])
								:addmm(self[DIR_R].Wop, tree.outer[col_i])
								:add(self[DIR_R].bo)

			if tree[DIR_R].n_children[i] == 0 then
				tree[DIR_R].EOC_outer[col_i] = self.func((self[DIR_R].anon_inner + input):div(n_left+1):add(input_head))
			else 
				local left_sister	= nil
			
				-- compute outer rep. for its children
				for j = 1, tree[DIR_R].n_children[i] do
					local child = tree[DIR_R].children[{j,i}]
					local col_c = {{},{child}}

					-- compute constructed outer
					if left_sister then 
						input:addmm(self[DIR_R].Wo[tree.deprel[left_sister]], tree.inner[{{},{left_sister}}])
						tree.cstr_outer[col_c] = self.func(torch.div(input, n_left + j-1):add(input_head))
					else 
						tree.cstr_outer[col_c] = self.func((self[DIR_R].anon_inner+input):div(n_left+1):add(input_head))
					end
					left_sister = child			
				end

				-- compute outer rep. for EOC
				input:addmm(self[DIR_R].Wo[tree.deprel[left_sister]], tree.inner[{{},{left_sister}}])
				tree[DIR_R].EOC_outer[col_i] = self.func(torch.div(input,n_left + tree[DIR_R].n_children[i]):add(input_head))
			end

	end

	-- compute probabilities
	-- Pr(deprel | outer)
	tree.deprel_score	= (self.Wdr * tree.cstr_outer):add(torch.repeatTensor(self.bdr, 1, tree.n_nodes))
	tree.deprel_prob	= safe_compute_softmax(tree.deprel_score)

	tree[DIR_L].EOC_score	= (self.Wdr * tree[DIR_L].EOC_outer):add(torch.repeatTensor(self.bdr, 1, tree.n_nodes))
	tree[DIR_L].EOC_prob	= safe_compute_softmax(tree[DIR_L].EOC_score)
	tree[DIR_R].EOC_score	= (self.Wdr * tree[DIR_R].EOC_outer):add(torch.repeatTensor(self.bdr, 1, tree.n_nodes))
	tree[DIR_R].EOC_prob	= safe_compute_softmax(tree[DIR_R].EOC_score)

	-- Pr(pos | deprel, outer)
	tree.pos_score	= 	(self.Wpos * tree.cstr_outer)
						:add(self.Ldrpos:index(2, tree.deprel:long()))
						:add(torch.repeatTensor(self.bpos, 1, tree.n_nodes))
	tree.pos_prob	= safe_compute_softmax(tree.pos_score)

	-- Pr(word | pos, deprel, outer)
	--[[
	tree.word_score	= 	(self.Wword * tree.cstr_outer)
						:add(self.Ldrword:index(2, tree.deprel:long()))
						:add(self.Lposword:index(2, tree.pos:long()))
						:add(torch.repeatTensor(self.bword, 1, tree.n_nodes))
	tree.word_prob	= safe_compute_softmax(tree.word_score)
	]]
	tree.word_score = {}
	tree.word_prob	= {}
	for i = 2,tree.n_nodes do
		local word = tree.word[i]
		local len = self.voca_dic.code_len[word]
		local path = self.voca_dic.path[{word,{1,len}}]
		tree.word_score[i] = (self.Wword:index(1,path:long()) * tree.cstr_outer[{{},{i}}])
								:add(self.Ldrword:index(1,path:long())[{{},{tree.deprel[i]}}])
								:add(self.Lposword:index(1,path:long())[{{},{tree.pos[i]}}])
								:add(self.bword:index(1,path:long()))
								:cmul(self.voca_dic.code[{{word},{1,len}}])
		tree.word_prob[i] = logistic(tree.word_score[i])
	end

	-- Pr(cap | word, pos, deprel, outer)
	tree.cap_score	= 	(self.Wcap * tree.cstr_outer)
						:add(self.Ldrcap:index(2, tree.deprel:long()))
						:add(self.Lposcap:index(2, tree.pos:long()))
						:add(self.Lwordcap:index(2, tree.word:long()))
						:add(torch.repeatTensor(self.bcap, 1, tree.n_nodes))
	tree.cap_prob	= safe_compute_softmax(tree.cap_score)

	--[[ Pr(dist | cap, word, pos, deprel, outer)
	tree.dist_score	= 	(self.Wdist * tree.cstr_outer)
						:add(self.Ldrdist:index(2, tree.deprel:long()))
						:add(self.Lposdist:index(2, tree.pos:long()))
						:add(self.Lworddist:index(2, tree.word:long()))
						:add(self.Lcapdist:index(2, tree.cap:long()))
						:add(torch.repeatTensor(self.bdist, 1, tree.n_nodes))
	tree.dist_prob	= safe_compute_softmax(tree.dist_score)
]]
	-- compute error
	tree.total_err = 0
	for i = 2, tree.n_nodes do
		tree.total_err = tree.total_err - math.log(tree.deprel_prob[{tree.deprel[i],i}])
										- math.log(tree.pos_prob[{tree.pos[i],i}])
										--- math.log(tree.word_prob[{tree.word[i],i}])
										- torch.log(tree.word_prob[i]):sum()
										- math.log(tree.cap_prob[{tree.cap[i],i}])
										-- - math.log(tree.dist_prob[{tree.dist[i],i}])
	end
	tree.total_err = tree.total_err - torch.log(tree[DIR_L].EOC_prob[{self.deprel_dic.size+1,{}}]):sum()
									- torch.log(tree[DIR_R].EOC_prob[{self.deprel_dic.size+1,{}}]):sum()
	return tree.total_err
end

--*********************** backpropagate *********************--
function IORNN:backpropagate_outside(tree, grad, complete_inside)
	ctx_trees = tree.ctx_trees

	if tree.gradi == nil then
		tree.gradi			= torch.zeros(self.dim, tree.n_nodes)
		tree.gradcomplete_i = torch.zeros(self.dim, tree.n_nodes)
	end

	local grado		= torch.zeros(self.dim, tree.n_nodes)
	local gradcstro	= torch.zeros(self.dim, tree.n_nodes)
	local gradEOCo	= { [DIR_L] = torch.zeros(self.dim, tree.n_nodes),
						[DIR_R]	= torch.zeros(self.dim, tree.n_nodes) }


	local gZdr		= tree.deprel_prob	:clone()
	local gZpos		= tree.pos_prob		:clone()
	local gZword	= {} --tree.word_prob	:clone()
	local gZcap		= tree.cap_prob		:clone()
	--local gZdist	= tree.dist_prob	:clone()
	local gZEOC		= {	[DIR_L] = tree[DIR_L].EOC_prob	:clone(),
						[DIR_R] = tree[DIR_R].EOC_prob	:clone() }

	for i = 2, tree.n_nodes do
		gZdr[{tree.deprel[i],i}]	= gZdr[{tree.deprel[i],i}]	- 1
		gZpos[{tree.pos[i],i}]		= gZpos[{tree.pos[i],i}]		- 1

		--gZword[{tree.word[i],i}]	= gZword[{tree.word[i],i}]	- 1
		local word = tree.word[i]
		local len = self.voca_dic.code_len[word]
		gZword[i] = (tree.word_prob[i] - 1):cmul(self.voca_dic.code[{word,{1,len}}])

		gZcap[{tree.cap[i],i}]		= gZcap[{tree.cap[i],i}]	- 1
		--gZdist[{tree.dist[i],i}]	= gZdist[{tree.dist[i],i}]	- 1
	end
	gZdr[{{},{1}}]	:fill(0) -- don't take ROOT into account
	gZpos[{{},{1}}]	:fill(0)
	--gZword[{{},{1}}]:fill(0)
	gZcap[{{},{1}}]	:fill(0)
	--gZdist[{{},{1}}]:fill(0)

	gZEOC[DIR_L][{self.deprel_dic.size+1,{}}]:add(-1)
	gZEOC[DIR_R][{self.deprel_dic.size+1,{}}]:add(-1)

	-- for Pr( . | context)
	grad.Wdr	:addmm(gZdr, tree.cstr_outer:t())
				:addmm(gZEOC[DIR_L], tree[DIR_L].EOC_outer:t())
				:addmm(gZEOC[DIR_R], tree[DIR_R].EOC_outer:t())
	grad.bdr	:add(gZdr:sum(2))
				:add(gZEOC[DIR_L]:sum(2))
				:add(gZEOC[DIR_R]:sum(2))
	gradcstro	:addmm(self.Wdr:t(), gZdr)
	gradEOCo[DIR_L]:addmm(self.Wdr:t(), gZEOC[DIR_L])
	gradEOCo[DIR_R]:addmm(self.Wdr:t(), gZEOC[DIR_R])

	grad.Wpos	:addmm(gZpos, tree.cstr_outer:t())
	grad.bpos	:add(gZpos:sum(2))
	gradcstro	:addmm(self.Wpos:t(), gZpos)

--	grad.Wword		:addmm(gZword, tree.cstr_outer:t())
--	grad.bword		:add(gZword:sum(2))
--	tree.gradcstro	:addmm(self.Wword:t(), gZword)

	grad.Wcap	:addmm(gZcap, tree.cstr_outer:t())
	grad.bcap	:add(gZcap:sum(2))
	gradcstro	:addmm(self.Wcap:t(), gZcap)

	--grad.Wdist		:addmm(gZdist, tree.cstr_outer:t())
	--grad.bdist		:add(gZdist:sum(2))
	--tree.gradcstro	:addmm(self.Wdist:t(), gZdist)

	for i = 2,tree.n_nodes do
		grad.Ldrpos[{{},{tree.deprel[i]}}]	:add(gZpos[{{},{i}}])

		-- for word
		local word = tree.word[i]
		local len = self.voca_dic.code_len[word]
		local path = self.voca_dic.path[{word,{1,len}}]
		gradcstro[{{},{i}}]:addmm(self.Wword:index(1,path:long()):t(), gZword[i])
		grad.Wword:indexCopy(1, path:long(), grad.Wword:index(1, path:long()):addmm(gZword[i],tree.cstr_outer[{{},{i}}]:t()))
		grad.bword:indexCopy(1, path:long(), grad.bword:index(1, path:long()):add(gZword[i]))

		local graddr = grad.Ldrword[{{},{tree.deprel[i]}}]
		graddr:indexCopy(1, path:long(), graddr:index(1, path:long()):add(gZword[i]))
		local gradpos = grad.Lposword[{{},{tree.pos[i]}}]
		gradpos:indexCopy(1, path:long(), gradpos:index(1, path:long()):add(gZword[i]))

		grad.Ldrcap[{{},{tree.deprel[i]}}]	:add(gZcap[{{},{i}}])
		grad.Lposcap[{{},{tree.pos[i]}}]	:add(gZcap[{{},{i}}])
		grad.Lwordcap[{{},{tree.word[i]}}]	:add(gZcap[{{},{i}}])

		--grad.Ldrdist[{{},{tree.deprel[i]}}]	:add(gZdist[{{},{i}}])
		--grad.Lposdist[{{},{tree.pos[i]}}]	:add(gZdist[{{},{i}}])
		--grad.Lworddist[{{},{tree.word[i]}}]	:add(gZdist[{{},{i}}])
		--grad.Lcapdist[{{},{tree.cap[i]}}]	:add(gZdist[{{},{i}}])
	end

	-- backward 
	gradZcstro = gradcstro:cmul(self.funcPrime(tree.cstr_outer))

	gradZEOCo = {}
	for _,dir in ipairs({DIR_L, DIR_R}) do
		gradZEOCo[dir]	= gradEOCo[dir]:cmul(self.funcPrime(tree[dir].EOC_outer))
		grad[dir].Woh	:addmm(gradZEOCo[dir], tree.inner:t())
		grad[dir].Wop	:addmm(gradZEOCo[dir], tree.outer:t())
		grad[dir].bo	:add(gradZEOCo[dir]:sum(2))

		tree.gradi	:addmm(self[dir].Woh:t(), gradZEOCo[dir])
		grado		:addmm(self[dir].Wop:t(), gradZEOCo[dir])
	end

	-- for cstr outer
	for i = tree.n_nodes, 1, -1 do
		local col_i = {{},{i}}

		-- for EOC 
		-- left
			local gz = gradZEOCo[DIR_L][col_i]
			n_left = 1

			if tree[DIR_L].n_children[i] == 0 then
				grad[DIR_L].anon_inner:add(gz)
			else 
				n_left = tree[DIR_L].n_children[i]
				local t = 1/n_left
				for j = 1,tree[DIR_L].n_children[i] do
					local child = tree[DIR_L].children[{j,i}]
					local col_c = {{},{child}}
					grad[DIR_L].Wo[tree.deprel[child]]	:addmm(t, gz, tree.inner[col_c]:t())
					tree.gradi[col_c]					:addmm(t, self[DIR_L].Wo[tree.deprel[child]]:t(), gz)
				end
			end
		-- right
			local gz = gradZEOCo[DIR_R][col_i]
			n_right = 1

			if tree[DIR_R].n_children[i] == 0 then
				grad[DIR_R].anon_inner:add(1/(n_left+1),gz)
			else 
				n_right = tree[DIR_R].n_children[i]
				local t = 1/ (n_right + n_left)
				for j = 1,tree[DIR_R].n_children[i] do
					local child = tree[DIR_R].children[{j,i}]
					local col_c = {{},{child}}
					grad[DIR_R].Wo[tree.deprel[child]]	:addmm(t, gz, tree.inner[col_c]:t())
					tree.gradi[col_c]					:addmm(t, self[DIR_R].Wo[tree.deprel[child]]:t(), gz)
				end
			end

			if tree[DIR_L].n_children[i] == 0 then
				grad[DIR_L].anon_inner:add(1/(n_left+n_right), gz)
			else
				local t = 1/(n_right + n_left)
				for j = 1,tree[DIR_L].n_children[i] do
					local child = tree[DIR_L].children[{j,i}]
					local col_c = {{},{child}}
					grad[DIR_L].Wo[tree.deprel[child]]	:addmm(t, gz, tree.inner[col_c]:t())
					tree.gradi[col_c]					:addmm(t, self[DIR_L].Wo[tree.deprel[child]]:t(), gz)
				end
			end

		-- for children's constr outers
		-- left
			for j = 1,tree[DIR_L].n_children[i] do
				local child = tree[DIR_L].children[{j,i}]
				local col_c = {{},{child}}
				local gz = gradZcstro[col_c]

				grad[DIR_L].Woh:addmm(gz, tree.inner[col_i]:t())
				grad[DIR_L].Wop:addmm(gz, tree.outer[col_i]:t())
				grad[DIR_L].bo :add(gz)

				tree.gradi[col_i]:addmm(self[DIR_L].Woh:t(), gz)
				grado[col_i]:addmm(self[DIR_L].Wop:t(), gz)
	
				if j == 1 then 
					grad[DIR_L].anon_inner:add(gz)
				else
					local t = 1 / (j-1)
					for k = 1,j-1 do
						local sister = tree[DIR_L].children[{k,i}]
						local col_s = {{},{sister}}
						grad[DIR_L].Wo[tree.deprel[sister]]	:addmm(t, gz, tree.inner[col_s]:t())
						tree.gradi[col_s]					:addmm(t, self[DIR_L].Wo[tree.deprel[sister]]:t(), gz)
					end
				end
			end
		-- right
			for j = 1,tree[DIR_R].n_children[i] do
				local child = tree[DIR_R].children[{j,i}]
				local col_c = {{},{child}}
				local gz = gradZcstro[col_c]

				grad[DIR_R].Woh:addmm(gz, tree.inner[col_i]:t())
				grad[DIR_R].Wop:addmm(gz, tree.outer[col_i]:t())
				grad[DIR_R].bo :add(gz)

				tree.gradi[col_i]:addmm(self[DIR_R].Woh:t(), gz)
				grado[col_i]:addmm(self[DIR_R].Wop:t(), gz)

				n_right = 1
	
				if j == 1 then 
					grad[DIR_R].anon_inner:add(1/(n_left+1), gz)
				else
					n_right = j - 1
					local t = 1 / (n_left+n_right)
					for k = 1,j-1 do
						local sister = tree[DIR_R].children[{k,i}]
						local col_s = {{},{sister}}
						grad[DIR_R].Wo[tree.deprel[sister]]	:addmm(t, gz, tree.inner[col_s]:t())
						tree.gradi[col_s]					:addmm(t, self[DIR_R].Wo[tree.deprel[sister]]:t(), gz)
					end
				end

				if tree[DIR_L].n_children[i] == 0 then
					grad[DIR_L].anon_inner:add(1/(n_left+n_right), gz)
				else
					local t = 1/(n_right + n_left)
					for k = 1,tree[DIR_L].n_children[i] do
						local sister = tree[DIR_L].children[{k,i}]
						local col_s = {{},{sister}}
						grad[DIR_L].Wo[tree.deprel[sister]]	:addmm(t, gz, tree.inner[col_s]:t())
						tree.gradi[col_s]					:addmm(t, self[DIR_L].Wo[tree.deprel[sister]]:t(), gz)
					end
			
				end
			end

		-- for full outer
		if i == 1 then
			if #ctx_trees == 0 then
				grad.anon_outer:add(grado[{{},{1}}])
			else
				local gradZoroot = grado[{{},{1}}]:cmul(self.funcPrime(tree.outer[{{},{1}}]))
				grad.bctx_trees:add(gradZoroot)
				for j,ctx_tree in ipairs(ctx_trees) do
					if ctx_tree == 0 then
						grad.Wctx_trees[j]:addmm(gradZoroot, self.root_complete_inner:t())
						grad.root_complete_inner:addmm(self.Wctx_trees[j]:t(), gradZoroot)
					else
						grad.Wctx_trees[j]:addmm(gradZoroot, ctx_tree.complete_inner[{{},{1}}]:t())
						if ctx_tree.gradiroot == nil then 
							ctx_tree.gradiroot = self.Wctx_trees[j]:t() * gradZoroot
						else
							ctx_tree.gradiroot:addmm(self.Wctx_trees[j]:t(), gradZoroot)
						end
					end
				end
			end

		else 
			local parent = tree.parent[i]
			local dir = tree.dir[i]
			local col_p = {{},{parent}}
			local gz = grado[col_i]:cmul(self.funcPrime(tree.outer[col_i]))

			grad[dir].Woh:addmm(gz, tree.inner[col_p]:t())	
			grad[dir].Wop:addmm(gz, tree.outer[col_p]:t())
			grad[dir].bo :add(gz)

			tree.gradi[col_p]:addmm(self[dir].Woh:t(), gz)
			grado[col_p]:addmm(self[dir].Wop:t(), gz)
			
			local n_sister = tree[DIR_L].n_children[parent] + tree[DIR_R].n_children[parent] - 1
			if n_sister == 0 then
				grad[dir].anon_inner:add(gz)
			else
				local t = 1 / n_sister
				for _,c_dir in ipairs({DIR_L, DIR_R}) do
					for j = 1,tree[c_dir].n_children[parent] do
						local sister = tree[c_dir].children[{j,parent}]
						if sister < i then
							local col_s = {{},{sister}}
							if complete_inside == CMPL_INSIDE_LEFT2RIGHT then 
								grad[c_dir].Wo[tree.deprel[sister]]	:addmm(t, gz, tree.complete_inner[col_s]:t())
								tree.gradcomplete_i[col_s]			:addmm(t, self[c_dir].Wo[tree.deprel[sister]]:t(), gz)
							else
								grad[c_dir].Wo[tree.deprel[sister]]	:addmm(t, gz, tree.inner[col_s]:t())
								tree.gradi[col_s]					:addmm(t, self[c_dir].Wo[tree.deprel[sister]]:t(), gz)
							end

						elseif sister > i then
							local col_s = {{},{sister}}
							if complete_inside == CMPL_INSIDE_RIGHT2LEFT then
								grad[c_dir].Wo[tree.deprel[sister]]	:addmm(t, gz, tree.complete_inner[col_s]:t())
								tree.gradcomplete_i[col_s]			:addmm(t, self[c_dir].Wo[tree.deprel[sister]]:t(), gz)					
							else
								grad[c_dir].Wo[tree.deprel[sister]]	:addmm(t, gz, tree.inner[col_s]:t())
								tree.gradi[col_s]					:addmm(t, self[c_dir].Wo[tree.deprel[sister]]:t(), gz)
							end
						end
					end
				end
			end	
		end
	end
end

function IORNN:backpropagate_inside(tree, grad, complete_inside)
	if tree.gradi ~= nil then
		grad.root_inner:add(tree.gradi[{{},{1}}])

		local gradZi = tree.gradi:cmul(self.funcPrime(tree.inner))
		grad.Wh	:addmm(gradZi[{{},{2,-1}}], self.L:index(2, tree.word[{{2,-1}}]:long()):t())
		grad.bh	:add(gradZi[{{},{2,-1}}]:sum(2))

		for i = 2, tree.n_nodes do
			local col = {{},{i}}
			local gz = gradZi[col]
			grad.L[{{},{tree.word[i]}}]:addmm(self.Wh:t(), gz)
			grad.Lpos[{{},{tree.pos[i]}}]:add(gz)
			grad.Lcap[{{},{tree.cap[i]}}]:add(gz)
		end
		tree.gradi = nil
	end

	if complete_inside ~= CMPL_INSIDE_NONE and (tree.gradcomplete_i ~= nil or tree.gradiroot ~= nil) then
		if tree.gradcomplete_i == nil then
			tree.gradcomplete_i = torch.zeros(self.dim, tree.n_nodes)
		end
		if tree.gradiroot ~= nil then
			tree.gradcomplete_i[{{},{1}}]:copy(tree.gradiroot)
		end

		for i = 1, tree.n_nodes do 
			local n_children = tree[DIR_L].n_children[i] + tree[DIR_R].n_children[i]
			local gradZi = tree.gradcomplete_i[{{},{i}}]:cmul(self.funcPrime(tree.complete_inner[{{},{i}}]))

			-- for head
			local gradZhead = gradZi
			if n_children > 0 then
				gradZhead = (self.Wih:t() * gradZi):cmul(self.funcPrime(tree.inner[{{},{i}}]))
				grad.Wih:addmm(gradZi, tree.inner[{{},{i}}]:t())
			end

			if i == 1 then
				grad.root_inner:addmm(self.Wih:t(), gradZi)
			else
				grad.Wh	:addmm(gradZhead, self.L[{{},{tree.word[i]}}]:t())
				grad.bh	:add(gradZhead)
				grad.L[{{},{tree.word[i]}}]:addmm(self.Wh:t(), gradZhead)
				grad.Lpos[{{},{tree.pos[i]}}]:add(gradZhead)
				grad.Lcap[{{},{tree.cap[i]}}]:add(gradZhead)
			end

			-- for children
			if n_children > 0 then
				grad.bi:add(gradZi)
				gradZi:div(n_children)

				for _,dir in ipairs({DIR_L,DIR_R}) do
					for j = 1, tree[dir].n_children[i] do
						local child = tree[dir].children[{j,i}]
						grad[dir].Wi[tree.deprel[child]]:addmm(gradZi, tree.complete_inner[{{},{child}}]:t())
						tree.gradcomplete_i[{{},{child}}]:addmm(self[dir].Wi[tree.deprel[child]]:t(), gradZi)
					end
				end
			end
		end
		
		tree.gradcomplete_i = nil
		tree.gradiroot = nil
	end
end

function IORNN:compute_log_prob(dsbank, ctx_trees)
	local scores = {}
	local trees = {}
	for i, ds in ipairs(dsbank) do
		local tree = ds:to_torch_matrix_tree()
		self:forward_inside(tree)
		if self.complete_inside == CMPL_INSIDE_2WAY then
			scores[i] = - (self:forward_outside(tree, ctx_trees, CMPL_INSIDE_LEFT2RIGHT) 
						+ self:forward_outside(tree, ctx_trees, CMPL_INSIDE_RIGHT2LEFT)) / 2
		else
			scores[i] = -self:forward_outside(tree, ctx_trees, self.complete_inside) 
		end
		trees[i] = tree
	end

	return scores, trees
end

function IORNN:computeCostAndGrad(treebank, start_id, end_id, config, grad)

	--p:start('compute cost and grad')	

	grad.params:fill(0)  -- always make sure that this grad is intialized with 0

	local cost = 0
	local nSample = 0
	local tword = {}
	local tpath = {}
	
	local total_trees_id = {}

	--p:start('process dsbank')
	for i = start_id,end_id do
		tree = treebank[i]
		total_trees_id[i] = 1

		-- extract contextual trees
		local ctx_trees = {}
		for t = 1, self.n_prevtrees do
			local j = i -1 - self.n_prevtrees + t
			if j < 1 or treebank.doc_id[j] ~= treebank.doc_id[i] then
				ctx_trees[t] = 0
			else
				ctx_trees[t] = treebank[j]
				total_trees_id[j] = 1
				if j < start_id then
					self:forward_inside(ctx_trees[t])
				end
			end
		end

		self:forward_inside(tree)

		if self.complete_inside == CMPL_INSIDE_2WAY then
			cost = cost + self:forward_outside(tree, ctx_trees, CMPL_INSIDE_LEFT2RIGHT) 
			self:backpropagate_outside(tree, grad, CMPL_INSIDE_LEFT2RIGHT)
			cost = cost + self:forward_outside(tree, ctx_trees, CMPL_INSIDE_RIGHT2LEFT) 
			self:backpropagate_outside(tree, grad, CMPL_INSIDE_RIGHT2LEFT)	
		else
			cost = cost + self:forward_outside(tree, ctx_trees, self.complete_inside) 
			self:backpropagate_outside(tree, grad, self.complete_inside)
		end

		nSample = nSample + tree.n_nodes
		for k = 2,tree.wnode:numel() do -- do not take the root into account
			local word = tree.word[tree.wnode[k]]
			tword[word] = 1
			for j=1,self.voca_dic.code_len[word] do
				tpath[self.voca_dic.path[{word,j}]] = 1
			end
		end
	end

	for tree_id,_ in pairs(total_trees_id) do
		local tree = treebank[tree_id]
		self:backpropagate_inside(tree, grad, self.complete_inside)
		clean_tree(tree)
	end

	--p:lap('process dsbank') 

	--p:start('compute grad')
	local wparams = self.params[{{1,-1-self.dim*self.voca_dic.size-self.wdim*self.voca_dic.size}}]
	local grad_wparams = grad.params[{{1,-1-self.dim*self.voca_dic.size-self.wdim*self.voca_dic.size}}]
	cost = cost / nSample + config.lambda/2 * torch.pow(wparams,2):sum()
	grad_wparams:div(nSample):add(wparams * config.lambda)
	
	for wid,_ in pairs(tword) do
		cost = cost + torch.pow(self.L[{{},{wid}}],2):sum() * config.lambda_L/2
		grad.L[{{},{wid}}]:div(nSample):add(config.lambda_L, self.L[{{},{wid}}])
	end 

	for wid,_ in pairs(tpath) do
		cost = cost + torch.pow(self.Wword[{{wid},{}}],2):sum() * config.lambda/2
		grad.Wword[{{wid},{}}]:div(nSample):add(config.lambda, self.Wword[{{wid},{}}])
	end 

	--p:lap('compute grad')

	--p:lap('compute cost and grad') 
	--p:printAll()

	return cost, grad, tword, tpath
end

-- make sure gradients are computed correctly
function IORNN:checkGradient(treebank, config)
	local epsilon = 1e-4

	local dim = self.dim
	local wrdDicLen = self.wrdDicLen
	local nCat = self.nCat

	local Theta = self.params
	local grad = self:create_grad()
	local _, gradTheta = self:computeCostAndGrad(treebank, 1, #treebank, config, grad)
	gradTheta = gradTheta.params
	
	local n = Theta:nElement()
	print(n)
	local numGradTheta = torch.zeros(n)
	for i = 1,n do
		local index = {{i}}
		Theta[index]:add(epsilon)
		local grad = self:create_grad()
		local costPlus,_ = self:computeCostAndGrad(treebank, 1, #treebank, config, grad)
		
		Theta[index]:add(-2*epsilon)
		local costMinus,_ = self:computeCostAndGrad(treebank, 1, #treebank, config, grad)
		Theta[index]:add(epsilon)

		numGradTheta[i] = (costPlus - costMinus) / (2*epsilon) 

		local diff = math.abs(numGradTheta[i] - gradTheta[i])
		if diff > 0 and math.abs(gradTheta[i]) < 1e-7 then 
			print('diff ' .. i .. ' ' .. diff .. ' : ' .. numGradTheta[i] .. ' ' .. gradTheta[i])
		end
		if diff > 5e-9  then 
			print('diff ' .. i .. ' ' .. diff .. ' : ' .. numGradTheta[i] .. ' ' .. gradTheta[i] .. ' errrrrrrrrrrrrrr')
		end
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
	local voca_dic_lr = config.voca_dic_learningRate or 1e-3

	local lrd = config.learningRateDecay or 0
	state.evalCounter = state.evalCounter or 0
	local nevals = state.evalCounter

	-- (1) evaluate f(x) and df/dx
	local cost, grad, tword, tpath = func()

	-- (3) learning rate decay (annealing)
	local weight_clr	= weight_lr / (1 + nevals*lrd)
	local voca_dic_clr	= voca_dic_lr / (1 + nevals*lrd)

	-- (4) parameter update with single or individual learning rates
	if not state.paramVariance then
		state.paramVariance = self:create_grad()
		state.paramStd = self:create_grad()
	end

	-- for weights
	local wparamindex = {{1,-1-self.dim*self.voca_dic.size-self.wdim*self.voca_dic.size}}
	state.paramVariance.params[wparamindex]:addcmul(1,grad.params[wparamindex],grad.params[wparamindex])
	torch.sqrt(state.paramStd.params[wparamindex],state.paramVariance.params[wparamindex])
	self.params[wparamindex]:addcdiv(-weight_clr, grad.params[wparamindex],state.paramStd.params[wparamindex]:add(1e-10))

	-- for word embeddings
	for wid,_ in pairs(tword) do
		local col_i = {{},{wid}}
		state.paramVariance.L[col_i]:addcmul(1,grad.L[col_i],grad.L[col_i])
		torch.sqrt(state.paramStd.L[col_i],state.paramVariance.L[col_i])
		self.L[col_i]:addcdiv(-voca_dic_clr, grad.L[col_i],state.paramStd.L[col_i]:add(1e-10))
	end

	for wid,_ in pairs(tpath) do
		local row_i = {{wid},{}}
		state.paramVariance.Wword[row_i]:addcmul(1,grad.Wword[row_i],grad.Wword[row_i])
		torch.sqrt(state.paramStd.Wword[row_i],state.paramVariance.Wword[row_i])
		self.Wword[row_i]:addcdiv(-weight_clr, grad.Wword[row_i],state.paramStd.Wword[row_i]:add(1e-10))
	end

	-- (5) update evaluation counter
	state.evalCounter = state.evalCounter + 1
end

function IORNN:train_with_adagrad(traintreebank, batchSize, 
									maxepoch, lambda, prefix,
									adagrad_config, adagrad_state, 
									devdsbank_path, kbestdevdsbank_path)
	local nSample = #traintreebank

	local grad = self:create_grad()
	
	local epoch = 0
	local j = 0
	local percent = 0
	local percent_stick = 0

	epoch = epoch + 1
	print('===== epoch ' .. epoch .. '=====')
	print(get_current_time())
	
	while true do
		j = j + 1
		local start_id = (j-1)*batchSize + 1
		local end_id = math.min(nSample, j*batchSize)

		if start_id > nSample then 
			self:save(prefix .. '_' .. epoch)
			os.execute('th eval_depparser_rerank.lua '..prefix..'_'..epoch..' '..devdsbank_path..' '..kbestdevdsbank_path..' /tmp/dev &')

			j = 1 
			start_id = (j-1)*batchSize + 1
			end_id = math.min(nSample, j*batchSize)

			epoch = epoch + 1
			percent_stick = 0
			if epoch > maxepoch then break end
			print('===== epoch ' .. epoch .. '=====')
			print(get_current_time())
		end

	
		local function func()
			cost, grad, tword, tpath  = self:computeCostAndGrad(traintreebank, start_id, end_id,  
							{lambda = lambda.lambda, lambda_L=lambda.lambda_L}, grad)

			print('batch ' .. j .. ': ' .. cost) io.flush()		
			return cost, grad, tword, tpath
		end

		--p:start("optim")
		self:adagrad(func, adagrad_config, adagrad_state)
		
		--p:lap("optim")
		--p:printAll()

		percent = end_id * 100 / nSample
		if percent >= percent_stick then 
			print(get_current_time() .. '      ' .. string.format('%.1f%%',percent))
			percent_stick = percent_stick + 5
		end 

		collectgarbage()
	end

	return adagrad_config, adagrad_state
end


--[[ ********************************** test ******************************--
require 'depparser_rerank'
torch.setnumthreads(1)

local voca_dic = Dict:new()
voca_dic:load('../data/wsj-dep/toy/dic/words.lst')
local pos_dic = Dict:new()
pos_dic:load('../data/wsj-dep/toy/dic/pos.lst')
local deprel_dic = Dict:new()
deprel_dic:load('../data/wsj-dep/toy/dic/deprel.lst')
local lookup = torch.rand(2, voca_dic.size)

dim = 3
L = torch.rand(2, voca_dic.size)

print('training...')
local net = IORNN:new({ dim = dim, voca_dic = voca_dic, pos_dic = pos_dic, deprel_dic = deprel_dic,
						lookup = L, func = IORNN.default_func, funcPrime = IORNN.default_funcPrime })

local parser = Depparser:new(voca_dic, pos_dic, deprel_dic)
local dsbank,_ = parser:load_dsbank('../data/wsj-dep/toy/data/train.conll')

config = {lambda = 1e-4, lambda_L = 1e-7}
net.update_L = true
net:checkGradient(dsbank, config)
]]
