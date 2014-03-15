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
	local leafId = leafId or 1
	local input = trim_string(input)

	if string.sub(input,1,1) ~= '(' then
		return Tree:new(input, {leafId, leafId}), leafId+1
	end

	local str = input:sub(2,input:len()-1)

	-- read label
	local i = str:find(' ')
	local t = Tree:new(str:sub(1,i-1))

	-- read children
	i = i + 1
	
	local countOpen = 0
	str = str:sub(i)
	local j = 1
	while true do
		local c = str:sub(j,j)
		local next_c = str:sub(j+1,j+1)
		if c == '' then break end

		if c == '(' then 
			countOpen = countOpen + 1
		elseif c == ')' then
			countOpen = countOpen - 1
		end

		if countOpen == 0 and next_c == ' ' or next_c == '' then
			t.children[#t.children+1],leafId = 
					Tree:create_from_string(str:sub(1,j) , leafId) 
			str = str:sub(j+1)
			countOpen = 0
			j = 1
		else 
			j = j + 1
		end
	end
	t.cover = {t.children[1].cover[1], t.children[#t.children].cover[2]}

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

function Tree:collapse_unarybranch_not_POS()
    for _,child in ipairs(self.children) do
        child:collapse_unarybranch_not_POS()
    end

    if #self.children == 1 then
        local child = self.children[1]
        if #child.children > 0 then
            self.children = child.children
            self.label = child.label
        end
    end
end


function Tree:create_CoNLL2005_SRL(tokens)
	require 'utils'

	-- read tree & srls
	local str = ''
	local srl = {}
	for i = 7,#tokens[1] do
		srl[i-6] = {}
	end

	for i,tok in ipairs(tokens) do
		if 		tok[2] == '(' then tok[2] = '-LRB-'
		elseif 	tok[2] == ')' then tok[2] = '-RRB-'
		elseif 	tok[2] == '[' then tok[2] = '-LSB-'
		elseif 	tok[2] == ']' then tok[2] = '-RSB-'
		elseif 	tok[2] == '{' then tok[2] = '-LCB-'
		elseif 	tok[2] == '}' then tok[2] = '-RCB-'
		end

		if      tok[1] == '(' then tok[1] = '-LRB-'
		elseif  tok[1] == ')' then tok[1] = '-RRB-'
		elseif  tok[1] == '[' then tok[1] = '-LSB-'
		elseif  tok[1] == ']' then tok[1] = '-RSB-'
		elseif  tok[1] == '{' then tok[1] = '-LCB-'
		elseif  tok[1] == '}' then tok[1] = '-RCB-'
		end

		if tok[1] == '%' then tok[1] = '%%' end

		str = str .. string.gsub(tok[3], '[*]', '('..tok[2]..' '..tok[1]..')')

		for j = 7,#tok do
			if tok[j]:sub(1,1) == '(' then
				role = {label = split_string(tok[j], '[^()*]+')[1], cover = {i,i}}
				srl[j-6][#srl[j-6]+1] = role
			end
			if tok[j]:sub(tok[j]:len()) == ')' then 
				local role = srl[j-6][#srl[j-6]]
				if role.label ~= 'V' then
					role.cover[2] = i
				end
			end
		end
	end
	str = string.gsub(str, '[(]', ' ('):sub(2)
	--print(str)
	--print(srl)

	-- turn to torch_matrix
	local tree = Tree:create_from_string(str)
	return tree, srl
end

--------------------------------------------------------------------------------

function Tree:to_torch_matrices(vocaDic, ruleDic, has_head)
	require "utils"

	local has_head = has_head or false

	local nodes = self:to_flat_form()
	local nnodes = #nodes
	local grammar = ruleDic.gramamr

	local n_children = torch.zeros(nnodes)
	local children_id = torch.zeros(20, nnodes)
	local parent_id = torch.zeros(nnodes)
	
	local word_id = torch.zeros(nnodes)
	local rule_id = torch.zeros(nnodes)
	local cover = torch.zeros(2, nnodes)
	local sibling_order = torch.zeros(nnodes)
	local head_word_id = torch.zeros(nnodes)

	if has_head then
		--print('head')
		for _,node in ipairs(nodes) do
			local comp = split_string(node.label, '[^-]+')
			local pos = tonumber(comp[#comp])
			if pos == nil then error('has head???') end

			node.label = comp[1]
			for i = 2,#comp-1 do node.label = node.label .. '-' .. comp[i] end
			node.head_word_pos = pos
		end
	end

	local words = {}
	for _,node in ipairs(nodes) do
		if #node.childId == 0 then
			words[#words+1] = node.label
		end
	end

	--print(words)
	--print(nodes)

	for i,node in ipairs(nodes) do
		n_children[i] = #node.childId
		cover[{1,i}] = node.cover[1]
		cover[{2,i}] = node.cover[2]
		head_word_id[i] = vocaDic:get_id(words[node.head_word_pos])

		for j,cid in ipairs(node.childId) do
			if j > 20 then 
				print(#node.childId)
				print(self:to_string())
				error('invalid rule')
			end
			children_id[{j,i}] = cid
			parent_id[cid] = i
			sibling_order[cid] = j
		end
		
		if #node.childId == 0 then
			word_id[i] = vocaDic:get_id(node.label)
			rule_id[i] = 0
		else 
			word_id[i] = 0
			if grammar == "CCG" then
				local str = nil
				if #node.childId == 1 then
					str = node.label..'\tX'
				elseif #node.childId == 2 then
					str = node.label..'\tX\tX'
				end
				rule_id[i] = ruleDic:get_id(str)

				if rule_id[i] == ruleDic.word2id["UNKNOWN"] then
					if #node.childId == 1 then
						str = 'X\tX'
					elseif #node.childId == 2 then
						str = 'X\tX\tX'
					end
					rule_id[i] = ruleDic:get_id(str)
				end
				
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
				print(node.label) 
				error('invalid rule in')
			end
		end
	end

	return {	
				n_nodes			= nnodes,
				n_children		= n_children,
				cover			= cover,
				children_id		= children_id, 
				parent_id		= parent_id,
				word_id			= word_id,
				head_word_id	= head_word_id,
				rule_id			= rule_id,
				sibling_order	= sibling_order
		}
end

function Tree:copy_torch_matrix_tree(tree)
 	return {	
				n_nodes			= tree.n_nodes,
				n_children		= tree.n_children:clone(),
				cover			= tree.cover:clone(),
				children_id		= tree.children_id:clone(), 
				parent_id		= tree.parent_id:clone(),
				word_id			= tree.word_id:clone(),
				head_word_id	= tree.head_word_id:clone(),
				rule_id			= tree.rule_id:clone(),
				sibling_order	= tree.sibling_order:clone()
		}
end

function assign_srl_role(tree, role, classDic, node_id)
	local node_id = node_id or 1
	if tree.cover[{2,node_id}] >= role.cover[1] and tree.cover[{1,node_id}] <= role.cover[2] then
		if role.cover[1] <= tree.cover[{1,node_id}] and role.cover[2] >= tree.cover[{2,node_id}] then
			tree.class_gold[{classDic:get_id(role.label),node_id}] = 1
			tree.class_gold[{classDic:get_id('NULL'),node_id}] = 0
		else
			for i = 1,tree.n_children[node_id] do
				assign_srl_role(tree, role, classDic, tree.children_id[{i,node_id}])
			end
		end
	end
end

function Tree:add_srl_torch_matrix_tree(tree, srl, classDic)
	tree.target_verb = torch.zeros(tree.n_nodes)
	tree.class_gold = torch.zeros(classDic:size(), tree.n_nodes):byte()
	tree.class_gold[{{classDic:get_id('NULL')},{}}] = 1
	for _,role in ipairs(srl) do

		-- if this is the target verb
		if role.label == 'V' then
			for i = 1,tree.n_nodes do
				if tree.cover[{1,i}] == role.cover[1] and tree.cover[{2,i}] == role.cover[2] 
				and tree.n_children[i] == 1 and tree.n_children[tree.children_id[{1,i}]] == 0 then
					tree.target_verb[i] = 1
					break
				end
			end

		-- other roles
		else
			assign_srl_role(tree, role, classDic)
		end
	end

	return tree
end

--[[ test 
tokens = {}
for line in io.lines('../data/SRL/toy/train.txt') do
	tokens[#tokens+1] = split_string(line, '[^ ]+')
end

local tree, srl = Tree:create_CoNLL2005_SRL(tokens)
print(tree:to_string())
print(srl)
]]
--[[
	local f = io.open(arg[2], 'w')
	local treebank = {}

	-- load trees
	local tokens = {}
	for line in io.lines(arg[1]) do
		line = trim_string(line)
		if line ~= '' then  -- continue read tokens
			tokens[#tokens+1] = split_string(line, '[^ ]+')

		-- line == '' means end of sentence
		else -- process the whole sentence
			local tree = nil
			local tree_torch = nil
			local srls = nil
			if pcall(function() 
					tree, srls = Tree:create_CoNLL2005_SRL(tokens)
					f:write(tree:to_string() .. '\n')
				end) 
			then
			else 
				print('error: ')
				print(tokens)
			end
			tokens = {}
		end
	end
	f:close()
]]

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
--local string = "(TOP (S (NP (NP (JJ Influential) (NNS members)) (PP (IN of) (NP (DT the) (NNP House) (NNP Ways) (CC and) (NNP Means) (NNP Committee)))) (VP (VBD introduced) (NP (NP (NN legislation)) (SBAR (WHNP (WDT that)) (S (VP (MD would) (VP (VB restrict) (SBAR (WHADVP (WRB how)) (S (NP (DT the) (JJ new) (NN savings-and-loan) (NN bailout) (NN agency)) (VP (MD can) (VP (VB raise) (NP (NN capital)))))) (, ,) (S (VP (VBG creating) (NP (NP (DT another) (JJ potential) (NN obstacle)) (PP (TO to) (NP (NP (NP (DT the) (NN government) (POS 's)) (NN sale)) (PP (IN of) (NP (JJ sick) (NNS thrifts)))))))))))))) (. .)))"
string = '(S1-2 (NP-2 (NNP-1 Two-Way-1) (NNP-2 Street-2)))'
print(string)

tree = Tree:create_from_string(string)
print(tree:to_string())
tree = tree:to_torch_matrices('x', 'y', true)
]]

--[[
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
--[[
--local string = "(rp (fa (fa Since (ba (ba (lex Taiwan) (conj and (lex (fa South Korea)))) (fa have (ba (fa neither (fa indigenous coal)) (conj nor (lex (fa natural gas))))))) (lp , (ba (lex (fa nuclear power)) (fa will (fa be (fa the (fa favoured (fa electricity (fa generation path))))))))) .)"
local string = "(rp (ba You (fa can (ba (ba (fa (fa keep it) (fa at (lex home))) (fa with (lex (fa other (fa important papers))))) (lp , (conj or (ba (fa (fa give it) (fa to (ba (fa your executor) (lp , (ba (fa your (fa professional adviser)) (lp , (conj or (fa your bank)))))))) (lex (fa to (ba look after))))))))) .)"
print(string)
tree = Tree:create_from_string(string)
print(tree:to_string())

-- load word emb and grammar rules
print('load wordembeddngs...')
local f = torch.DiskFile(arg[1], 'r')
local vocaDic = f:readObject(); setmetatable(vocaDic, Dict_mt)
local wembs = f:readObject()
f:close()

print('load grammar rules...')
ruleDic = Dict:new(cfg_template)
ruleDic:load(arg[2])

tree = tree:to_torch_matrices(vocaDic, ruleDic, 'CCG')
print(tree)

]]
