require 'utils'

Tree = {}
Tree_mt = {__index = Tree}

--*************** construction **************
function Tree:new(label, cat, cover)
	local t = {}

	t.label = label
	t.cat = cat
	t.children = {}
	t.cover = cover or {}

	setmetatable(t, Tree_mt)
	return t
end

function Tree:create_from_string( input , leafId )
	local str = input:sub(2,input:len()-1)
	local leafId = leafId or 1

	-- read label
	local i = str:find(' ')
	local lc = split_string(str:sub(1,i-1), "[^#]+")
	local t = Tree:new(lc[1], lc[2])

	-- read children
	i = i + 1
	
	-- check if this is POS
	if str:sub(i,i) ~= '(' then
		t.children[1] = Tree:new( str:sub(i)  )
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
	local newtree = Tree:new( self.label, self.sublabel, self.cover )
	
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
	ret[#ret+1] = {label = self.label, cat = self.cat, childId = {}}
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

function Tree:to_torch_matrices(word2id, nCat)
	require "utils"

	local nodes = self:to_flat_from()
	local nnodes = #nodes

	local n_children = torch.Tensor(nnodes)
	local children_id = torch.Tensor(2, nodes)
	local category = torch.Tensor(nCat, nnodes)
	local label_id = torch.Tensor(nnodes)
	
	for i,node in ipairs(nodes) do
		n_children[i] = #node.childId

		for j,cid in ipairs(node.childId) do
			children_id[{j,i}] = cid
		end

		cat = torch.zeros(nCat)
		cat[tonumber(node.cat)+1] = 1
		category[{{},i}]:copy(cat)
		
		if #node.childId == 0 then
			label_id = get_word_id(node.label)
		else 
			label_id = -1
		end
	end

	return 	{	
				n_nodes = nnodes,
				n_children = n_children, 
				children_id = children_id, 
				category = category,
				label_id = label_id 
			}
end
