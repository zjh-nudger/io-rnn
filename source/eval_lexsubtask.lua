require 'dict'
require 'utils'
require 'tree'
require 'iornn'

torch.setnumthreads(1)

function load_gold()
	-- load parses
	local parses = {}
	local i = 1
	local id = -1
	for line in io.lines(parses_path) do
		if math.mod(i, 2) == 1 then 
			id = tonumber(split_string(line, "[0-9]+")[1])
		else 
			local tree = Tree:create_from_string(line)
			parses[id] = tree:to_torch_matrix()
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
		--if iter > 200 then break end

		local comps = split_string(line)
		local word_id = dic:get_id(comps[1])
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
			name[id] = dic:get_id(comps[i])
			if cand_list.name[name[id]] == nil then 
				cand_list.name[name[id]] = 1
				cand_list.number = cand_list.number + 1
			end
			--print(comps)
			weight[{id}] = tonumber(comps[i+1])
		end

		if #name > 0 and word_id > 0 then
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
		if math.mod(iter, 50) == 0 then print(iter) end

		for ca,_ in pairs(case.cand_list.name) do
			name[#name+1] = ca
		end

		rank = rank_function(case, name)
		case.cand_rank = {name = {}}
		for i = 1,#name do
			case.cand_rank.name[i] = name[rank[{i}]]
		end
	end

	print('compute GAP...')
	return compute_avg_gap(cases)
end

function rank_random( case , cand_names )
	return torch.randperm(#cand_names)
end

function rank_wo_context( case, cand_names )
	--print(cand_names)
	local cand_embs = emb:index(2, torch.LongTensor(cand_names))
	local target_emb = emb[{{},{case.target}}]
	local score = compute_score(cand_embs, 
						torch.repeatTensor(target_emb, 1, #cand_names))
	-- sort cand
	_,rank = score:sort(1, true) 
	return rank	
end

function rank_context( case, cand_names )
	local cand_embs = net.L:index(2, torch.LongTensor(cand_names))
	local score = compute_score_iornn(net, case.parse, case.tw_position, cand_embs)
	_,rank = score:sort(1, true)
	return rank
end

function compute_score_iornn(case, cand_embs)
	local tree = net:parse({case.parse})[1]
	local outer = nil
	local inner = nil
	local ncand = cand_embs:size(2)
	
	local leaf_count = 0
	for i = 1,tree.n_nodes do
		if tree.n_children[i] == 0 then
			leaf_count = leaf_count + 1
			if leaf_count == case.tw_position then
				outer = tree.outer[{{},{i}}]
				inner = tree.inner[{{},{i}}]
				break
			end
		end
	end
		
	-- inner score
	local inner_score = compute_score(cand_embs, 
						torch.repeatTensor(inner, 1, ncand))

	-- outer score
	local small_WwiL = net.Wwi * cand_embs
	local word_io = func(
						torch.repeatTensor(net.Wwo*outer, 1, ncand)
						:add(small_WwiL)
						:add(torch.repeatTensor(net.bw, 1, ncand)))
	local outer_score = (Ws * word_io):reshape(n)

	return outer_score
end

if #arg == 2 then

	dic_emb_path	= arg[1]
	gold_path 		= arg[2]
	tw_position_path = arg[3]
	parses_path = arg[4]

	-- load dic & emb
	print('load dic & emb...')
	f = torch.DiskFile(dic_emb_path, 'r')
	dic = f:readObject()
	setmetatable(dic, Dict_mt)
	emb = f:readObject()
	f:close()

	-- load gold
	print('load gold and context...')
	cases = load_gold()

	-- eval
	rank_function = rank_wo_context
	print(eval(cases, rank_function))

else
	print('<dic_emb_path> <gold_path>')
end
