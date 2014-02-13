require 'dict'
require 'utils'
require 'tree'
require 'iornn'

torch.setnumthreads(1)
grammar = 'CCG'

function load_gold()
	-- load parses
	local parses = {}
	local i = 1
	local id = -1
	for line in io.lines(parses_path) do
		if math.mod(i, 2) == 1 then 
			id = tonumber(split_string(line, "[0-9]+")[1])
		else
			if pcall(function() 
					local tree = Tree:create_from_string(line)
					--print(tree:to_string())
					parses[id] = tree:to_torch_matrices(vocaDic, ruleDic, grammar)
					if rank_func_name == 'context' then 
						parses[id] = net:parse({parses[id]})[1]
					end
				end) == false then 
				parses[id] = nil
			end
		end
		i = i + 1
	end

	-- load target word position
	local tw_positions = {}
	local i = 1
	local id = -1
	for line in io.lines(tw_position_path) do
		if math.mod(i, 2) == 1 then 
			id = tonumber(line)
		else 
			tw_positions[id] = tonumber(line)
		end
		i = i + 1
	end

	-- load gold 
	local cases = {}
	local pre_word_id = -1
	local cand_list = { name = {} , number = 0 }

	local iter = 0
	for line in io.lines(gold_path) do
		iter = iter + 1
		--if iter > 300 then break end

		local comps = split_string(line)
		local word_id = vocaDic:get_id(comps[1])
		local case_id = tonumber(comps[2])

		if word_id ~= pre_word_id then
			cand_list = { name = {} , number = 0 }
			pre_word_id = word_id
		end

		-- extract gold rank
		local name = {}
		local weight = torch.zeros(#comps / 2 - 1)

		for i = 3,#comps,2 do
			local id = (i-1) / 2
			name[id] = vocaDic:get_id(comps[i])
			if cand_list.name[name[id]] == nil then 
				cand_list.name[name[id]] = 1
				cand_list.number = cand_list.number + 1
			end
			--print(comps)
			weight[{id}] = tonumber(comps[i+1])
		end

		if #name > 0 and word_id > 0 and parses[case_id] ~= nil then
			cases[#cases+1] = {
						target = word_id,
						gold_rank = {name = name, weight = weight}, 
						cand_list = cand_list,
						parse = parses[case_id],
						tw_position = tw_positions[case_id] }
		end
	end
	--print(cases)
	return cases
end

function compute_gap( gold_rank, cand_rank )
	cand_rank.weight = torch.zeros(#cand_rank.name)
	for i,na in ipairs(cand_rank.name) do
		for j,ma in ipairs(gold_rank.name) do
			if na == ma then
				cand_rank.weight[i] = gold_rank.weight[j]
			end
		end
	end

	local a = gold_rank.weight --print(a)
	local b = cand_rank.weight --print(b)
	local Ia = torch.gt(a, 0):double() --print(Ia)
	local Ib = torch.gt(b, 0):double() --print(Ib)
	local a_bar = 	torch.cumsum(a, 1)
					:cdiv(torch.linspace(1,a:numel(),a:numel())) 
					--print(a_bar)
	local b_bar = 	torch.cumsum(b, 1)
					:cdiv(torch.linspace(1,b:numel(),b:numel())) 
					--print(b_bar)

	local gap = torch.cmul(Ib,b_bar):sum() / torch.cmul(Ia,a_bar):sum() 
				--print(gap)
	return gap
end

function compute_avg_gap( cases )
	local sum_gap = 0
	for iter,case in pairs(cases) do
		sum_gap = sum_gap + compute_gap(case.gold_rank, case.cand_rank)
	end
	return sum_gap / #cases
end

function compute_score( x, y, measure)
	local measure = measure or 'cos'
	if measure == 'cos' then
		return torch.cdiv(x, torch.repeatTensor(x:norm(2,1),x:size(1),1))
			:cmul(torch.cdiv(y, torch.repeatTensor(y:norm(2,1),y:size(1),1)))
			:sum(1)[{1,{}}]
	elseif measure == 'dot' then
		return torch.cmul(x,y):sum(1)[{1,{}}]
	else
		error(measure .. ' is invalid')
	end
end

function eval( cases , rank_function )
	print('ranking...')
	for iter,case in pairs(cases) do
		local name = {}
		--if math.mod(iter, 50) == 0 then print(iter) end

		for ca,_ in pairs(case.cand_list.name) do
			name[#name+1] = ca
		end

		rank, score = rank_function(case, name)
		case.cand_rank = {name = {}}
		case.cand_score = score
		for i = 1,#name do
			case.cand_rank.name[i] = name[rank[{i}]]
		end
	end

	print('compute GAP...')
	return compute_avg_gap(cases)
end

function rank_random( case , cand_names )
	return torch.randperm(#cand_names), torch.rand(#cand_names)
end

function rank_wo_context( case, cand_names )
	--print(cand_names)
	local cand_embs = emb:index(2, torch.LongTensor(cand_names))
	local target_emb = emb[{{},{case.target}}]:clone()
	local score = compute_score(cand_embs, 
						torch.repeatTensor(target_emb, 1, #cand_names))
	-- sort cand
	_,rank = score:sort(1, true) 
	return rank, score
end

function rank_context( case, cand_names )
	local cand_embs = net.L:index(2, torch.LongTensor(cand_names))
	local score = compute_score_iornn(case, cand_embs)
	_,rank = score:sort(1, true)
	return rank, score
end

function compute_score_iornn(case, cand_embs)
	local tree = case.parse
	local outer = nil
	local inner = nil
	local ncand = cand_embs:size(2)
	
	local leaf_count = 0
	for i = 1,tree.n_nodes do
		if tree.n_children[i] == 0 then
			leaf_count = leaf_count + 1
			if leaf_count == case.tw_position then
				outer = tree.outer[{{},{i}}]:clone()
				inner = tree.inner[{{},{i}}]:clone()
				break
			end
		end
	end
--[[
	local sem1 = torch.Tensor(2*net.dim,1)
	sem1[{{1,net.dim},{1}}]:copy(inner)
	sem1[{{net.dim+1,2*net.dim},{1}}]:copy(outer*alpha)

	local sem2 = torch.Tensor(2*net.dim,ncand)
	sem2[{{1,net.dim},{}}]:copy(cand_embs)
	sem2[{{net.dim+1,2*net.dim},{}}]:copy(torch.repeatTensor(outer*alpha,1,ncand))
	return compute_score(torch.repeatTensor(sem1, 1, ncand), sem2)
]]	
	-- inner score
	local inner_score = compute_score(cand_embs, 
						torch.repeatTensor(inner, 1, ncand))

	-- outer score
	local small_WwiL = net.Wwi * cand_embs
	local word_io = tanh(
						torch.repeatTensor(net.Wwo*outer, 1, ncand)
						:add(small_WwiL)
						:add(torch.repeatTensor(net.bw, 1, ncand)))
	local outer_score = (net.Ws * word_io):reshape(ncand)

	return outer_score*alpha +  inner_score*(1-alpha)

--[[
	local sem1 = inner + outer * alpha
	local sem2 = cand_embs + torch.repeatTensor(outer*alpha, 1, ncand)
	return compute_score(torch.repeatTensor(sem1, 1, ncand), sem2)
]]
end

if #arg == 5 then

	vocaDic_emb_path	= arg[1]
	rule_path = arg[2]
	gold_path 		= arg[3] .. '/gold.txt'
	tw_position_path = arg[3] .. '/lexsub_word_pos.txt'
	parses_path = arg[3] .. '/lexsub_parse.ccg.penn.txt'
	net_path = arg[4]
	rank_func_name = arg[5]

	-- load vocaDic & emb
	print('load vocaDic & emb...')
	f = torch.DiskFile(vocaDic_emb_path, 'r')
	vocaDic = f:readObject()
	setmetatable(vocaDic, Dict_mt)
	emb = f:readObject()
	f:close()

	-- load grammar rules
	print('load grammar rules...')
	ruleDic = Dict:new(cfg_template)
	ruleDic:load(rule_path)

	-- load net
	print('load net...')
	net = IORNN:load(net_path)

	-- load gold
	print('load gold and context...')
	cases = load_gold()

	-- eval
	if rank_func_name == 'random' then
		rank_function = rank_random
	elseif rank_func_name == 'nocontext' then
		rank_function = rank_wo_context
	elseif rank_func_name == 'context' then
		rank_function = rank_context
	end

	alpha = 0
	for al = 0,1,0.1 do
		alpha = al
		print(alpha .. ' : ' .. eval(cases, rank_function))
	end

else
	print('<vocaDic_emb_path> <rule path> <lex sub dir> <net path> <rank func name>')
end
