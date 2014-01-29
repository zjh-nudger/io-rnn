require 'utils'
require 'dict'

Tree = {}
Tree_mt = {__index = Tree}

--*************** construction **************
function Tree:new(label, cover)
	local t = {}

	t.label = label
	t.children = {}
	t.cover = cover or {}

	setmetatable(t, Tree_mt)
	return t
end

function Tree:create_from_string(input , leafId)
	local str = input:sub(2,input:len()-1)
	local leafId = leafId or 1

	-- read label
	local i = str:find(' ')
	local t = Tree:new(str:sub(1,i-1))

	-- read children
	i = i + 1
	
	-- check if a leaf
	if str:sub(i,i) ~= '(' then
		t.children[1] = Tree:new(str:sub(i))
		t.children[1].cover = {leafId, leafId}
		t.cover = {leafId, leafId}
		leafId = leafId + 1

	else
		local countOpen = 1
		str = str:sub(i)
		local j = 2
		while true do
			local c = str:sub(j,j)
			if c == '(' then 
				countOpen = countOpen + 1
			elseif c == ')' then
				countOpen = countOpen - 1
			end

			if countOpen == 0 then
				t.children[#t.children+1],leafId = 
						Tree:create_from_string(str:sub(1,j) , leafId) 
				if j + 1 > str:len() then break end
				j = str:find('(',j+1,true)
				str = str:sub(j)
				countOpen = 1
				j = 2
			else 
				j = j + 1
			end
		end
		t.cover = {t.children[1].cover[1], t.children[#t.children].cover[2]}
	end

	return t, leafId
end

--******************** deep copy ****************
function Tree:clone()
	local nChild = #self.children
	local newtree = Tree:new(self.label, self.cover)
	
	-- clone children
	for i = 1,nChild do
		newtree.children[i] = self.children[i]:clone()
	end

	return newtree
end

--***************** convert to string ****************
function Tree:to_string( onlyLeaves ) 
	local onlyLeaves = onlyLeaves or false 
	local str = self.label .. (self.sublabel or '')

	local nChild = #self.children
	if nChild == 0 then
		if onlyLeaves then 
			return str .. ' '
		else 
			return str
		end
	else
		if onlyLeaves == false then 
			str = '(' .. str 
		else 
			str = ''
		end
		for i = 1,nChild do
			if onlyLeaves == false then 
				str = str .. ' ' .. self.children[i]:to_string(onlyLeaves)
			else 
				str = str .. self.children[i]:to_string(onlyLeaves) 
			end
		end
		if onlyLeaves == false then str = str .. ')' end
	end
	return str
end

--******************* extract leaves ***************
function Tree:leaves(ret)
	local ret = ret or {}
	local nChild = #self.children
	if nChild == 0 then
		ret[#ret+1] = self
	else
		for i = 1,nChild do
			self.children[i]:leaves(ret)
		end
	end
	return ret
end

--*********************** binarize tree ******************--
function Tree:binarize(left_branching, strict)
	local strict = strict or false
	local left_branching = left_branching or true
	local nChild = #self.children
       
	if strict then
		while nChild == 1 do
			if #self.children[1].children > 0 then
				self.label = self.label .. '^' .. self.children[1].label
			else 
				self.label = self.children[1].label
			end
			self.children = self.children[1].children
			nChild = #self.children
		end
	end

	if nChild > 2 then
		-- newlabel = '@[this_node_label]'
		local newlabel = self.label
		if newlabel:sub(1,1) ~= '@' then newlabel = '@' .. newlabel end

		-- create new child
		local lChild, rChild
		if left_branching then
			rChild = self.children[nChild]
			lChild = Tree:new(newlabel)
			for i = 1,nChild-1 do
				lChild.children[i] = self.children[i]
			end
			lChild.cover[1] = lChild.children[1].cover[1]
			lChild.cover[2] = lChild.children[nChild-1].cover[2]
		else
			lChild = self.children[1]
			rChild = Tree:new(newlabel)
			for i = 2, nChild do
				rChild.children[i-1] = self.children[i]
			end
			rChild.cover[1] = rChild.children[1].cover[1]
			rChild.cover[2] = rChild.children[nChild-1].cover[2]
		end
		
		-- update children list for this node
		self.children = {lChild, rChild}
	end
       
	-- binarize subtrees
	for i = 1,#self.children do
		self.children[i]:binarize(left_branching, strict)
	end
end
                                
--********************** get all nodes ******************
-- pre-order traverse
function Tree:all_nodes(ret)
	local ret = ret or {}
	local nChild = #self.children
	ret[#ret+1] = self

	for i = 1,nChild do
		self.children[i]:all_nodes( ret )
	end
	return ret
end

function Tree:to_flat_form(ret)
	local ret = ret or {}
	local nChild = #self.children
	ret[#ret+1] = {label = self.label, cat = self.cat, childId = {}, cover = self.cover}
	local id = #ret

	for i = 1,nChild do
		ret,cid = self.children[i]:to_flat_form(ret)
		ret[id].childId[i] = cid
	end
	return ret, id
end

--****************** load treebank from file ****************
function Tree:load_treebank(filename)
	local treebank = {}
	for line in io.lines(filename) do
		--print(line)
		treebank[#treebank+1] = Tree:create_from_string(line)
		--print(treebank[#treebank]:to_string())
	end
	return treebank
end
--[[
function Tree:to_stanford_sa_form()
	if #self.children > 0 then
		self.cat = tonumber(self.label)
		self.label = "X"
		
		-- merge leaf
		if #self.children == 1 and #self.children[1].children == 0 then
			self.label = self.children[1].label
			self.children = {}
		else 
			for _,child in ipairs(self.children) do
				child:to_stanford_sa_form()
			end
		end
	end
end
]]

function Tree:to_torch_matrices(vocaDic, ruleDic, grammar) --, nCat)
	require "utils"

	local nodes = self:to_flat_form()
	local nnodes = #nodes

	local n_children = torch.zeros(nnodes)
	local children_id = torch.zeros(20, nnodes)
	local parent_id = torch.zeros(nnodes)
	--local category = torch.zeros(nCat, nnodes)
	local word_id = torch.zeros(nnodes)
	local rule_id = torch.zeros(nnodes)
	local cover = torch.zeros(2, nnodes)
	local sibling_order = torch.zeros(nnodes)
	
	for i,node in ipairs(nodes) do
		n_children[i] = #node.childId
		cover[{1,i}] = node.cover[1]
		cover[{2,i}] = node.cover[2]

		for j,cid in ipairs(node.childId) do
			if j > 20 then 
				print(#node.childId)
				print(self:to_string())
			end
			children_id[{j,i}] = cid
			parent_id[cid] = i
			sibling_order[cid] = j
		end

		-- uncomment these lines in the case of supervised learning
		--cat = torch.zeros(nCat)
		--cat[tonumber(node.cat)+1] = 1
		--category[{{},i}]:copy(cat)
		
		if #node.childId == 0 then
			word_id[i] = vocaDic:get_id(node.label)
			rule_id[i] = 0
		else 
			word_id[i] = 0
			if grammar == "CCG" then
				local comps = split_string(nodel.label, '[^.]+')
				rule_id[i] = ruleDic:get_id(comps[2])
			else
				local str = node.label
				for j,cid in ipairs(node.childId) do
					if #nodes[cid].childId == 0 then 
						str = str .. '\t' .. '[word]'
					else
						str = str .. '\t' .. nodes[cid].label
					end
				end
				rule_id[i] = ruleDic:get_id(str)
				if rule_id[i] == ruleDic.word2id["UNKNOWN"] then
					str,_ = string.gsub(str, "[^ \t]+", "X")
					rule_id[i] = ruleDic:get_id(str)
				end
			end

			if rule_id[i] == 1 then 
				error('invalid rule in')
			end
		end
	end

	return 	{	
				n_nodes = nnodes,
				n_children = n_children,
				cover = cover,
				children_id = children_id, 
				parent_id = parent_id,
				--category = category,
				word_id = word_id,
				rule_id = rule_id,
				sibling_order = sibling_order
			}
end

function extract_all_phrases(tree, dic, phrases, node_id)
	local phrases = phrases or {}
	local node_id = node_id or 1

	if tree.n_children[node_id] == 0 then
		phrases[node_id] = dic.id2word[tree.word_id[node_id]]
	else
		local str = ''
		for i = 1,tree.n_children[node_id] do
			phrases = extract_all_phrases(tree, dic, phrases, 
										tree.children_id[{i,node_id}])
			str = str .. ' ' .. phrases[tree.children_id[{i,node_id}]]
		end
		phrases[node_id] = string.sub(str, 2)
	end

	return phrases
end



--*********** test **************
--[[
local string = "(TOP (S (NP (NP (JJ Influential) (NNS members)) (PP (IN of) (NP (DT the) (NNP House) (NNP Ways) (CC and) (NNP Means) (NNP Committee)))) (VP (VBD introduced) (NP (NP (NN legislation)) (SBAR (WHNP (WDT that)) (S (VP (MD would) (VP (VB restrict) (SBAR (WHADVP (WRB how)) (S (NP (DT the) (JJ new) (NN savings-and-loan) (NN bailout) (NN agency)) (VP (MD can) (VP (VB raise) (NP (NN capital)))))) (, ,) (S (VP (VBG creating) (NP (NP (DT another) (JJ potential) (NN obstacle)) (PP (TO to) (NP (NP (NP (DT the) (NN government) (POS 's)) (NN sale)) (PP (IN of) (NP (JJ sick) (NNS thrifts)))))))))))))) (. .)))"

print(string)

tree = Tree:create_from_string(string)
print(tree:to_string())

tree:binarize(true, true)
print(tree:to_string())
]]
--[[
for line in io.lines(arg[1]) do
	local tree = Tree:create_from_string(line)
	tree:binarize(true, false)
	print(tree:to_string())
end
]]
