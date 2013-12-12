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
		if math.mod(i, 3) == 1 then 
			id = tonumber(split_string(line, "[0-9]+")[1])
			parses[id] = {}
		else
			if line ~= "(TOP())" then  
				local tree = Tree:create_from_string(line)				
				parses[id][math.mod(i-1,3)] = tree:to_torch_matrices(dic, 1)
			end
		end
		i = i + 1
	end


	-- load word position 
	local tw_positions = {}
	local i = 1
	local id = -1
	for line in io.lines(tw_position_path) do
		if math.mod(i, 3) == 1 then 
			id = tonumber(line)
			tw_positions[id] = {}
		else 
			tw_positions[id][math.mod(i-1,3)] = tonumber(line)
		end
		i = i + 1
	end

	-- load human rates
	local cases = {}
	local iter = 0
	for line in io.lines(human_score_path) do
		iter = iter + 1
		--if iter > 10 then break end

		local comps = split_string(line, "[^\t]+")
		local case_id = tonumber(comps[1])
		local word_id = torch.Tensor({dic:get_id(comps[2]), dic:get_id(comps[4])})

		-- extract human rates
		local human_rates = torch.zeros(#comps - 8)

		for i = 9,#comps do
			human_rates[i-8] = tonumber(comps[i])
		end

		if word_id[1] ~= 1 and word_id[2] ~= 1 and parses[case_id][1] ~= nil and parses[case_id][2] ~= nil then
			cases[#cases+1] = {
						word_id = word_id,
						human_rates = human_rates,
						word = {comps[2], comps[4]},
						parse = parses[case_id],
						tw_position = tw_positions[case_id] 
					}
			--print('-------')
			--print(case_id)
			--print(comps[2] .. ' ' .. comps[4])
			--print(word_id)
		else
			--print(case_id)
		end
	end

	return cases
end

function compute_rho(gold_rate, cand_rate)
	return spearman_rho(gold_rate, cand_rate)
end

function compute_score(x, y, measure)
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

function eval( cases , rate_function )
	print('rating...')
	local ncases = #cases
	local human_rate = torch.zeros(ncases)
	local cand_rate = torch.zeros(ncases)

	for i,case in ipairs(cases) do
		if math.mod(i,100) == 0 then print(i) end
		human_rate[i] = case.human_rates:mean()
		cand_rate[i] = rate_function(case)

		--print('-----')
		--print(case.word)
		--print(case.word_id)
		--print(human_rate[i])
		--print(cand_rate[i] * 10)
	end

	return compute_rho(human_rate, cand_rate)
end

function rate_random( case )
	return math.random()
end

function rate_wo_context( case )
	local emb1 = emb[{{},{case.word_id[1]}}]:clone()
	local emb2 = emb[{{},{case.word_id[2]}}]:clone()
	return compute_score(emb1, emb2)
end

function rate_context( case )
	local trees = net:parse(case.parse)
	
	local outer = {}
	local inner = {}

	for k = 1,2 do	
		local tree = trees[k]
		local leaf_count = 0
		for i = 1,tree.n_nodes do
			if tree.n_children[i] == 0 then
				leaf_count = leaf_count + 1
				if leaf_count == case.tw_position[k] then
					outer[k] = tree.outer[{{},{i}}]:clone()
					inner[k] = tree.inner[{{},{i}}]:clone()
					break
				end
			end
		end
	end
		
	-- inner score
	--local inner_score = compute_score(inner[1], inner[2])
	--local outer_score = compute_score(outer[1], outer[2])

	--local alpha = 0.5
	--return outer_score*alpha *  inner_score*(1-alpha)
	
	--local sem = torch.Tensor(net.dim*2,2)
	--for k = 1,2 do
	--	sem[{{1,net.dim},{k}}]:copy(inner[k])
	--	sem[{{net.dim+1,2*net.dim},{k}}]:copy(outer[k])
	--end
	--return compute_score(sem[{{},{1}}], sem[{{},{2}}])

	local sem = torch.Tensor(net.dim,2)
	alpha = 0.1
	sem[{{},{1}}]:copy(inner[1] + outer[1]*alpha)
	sem[{{},{2}}]:copy(inner[2] + outer[2]*alpha)

	return compute_score(sem[{{},{1}}], sem[{{},{2}}])

end

if #arg == 4 then
	dic_emb_path	= arg[1]
	human_score_path = arg[2] .. '/ratings.txt'
	tw_position_path = arg[2] .. '/word_pos.txt'
	parses_path = arg[2] .. '/parse.txt'
	net_path = arg[3]
	rate_func_name = arg[4]

	-- load dic & emb
	print('load dic & emb...')
	f = torch.DiskFile(dic_emb_path, 'r')
	dic = f:readObject()
	setmetatable(dic, Dict_mt)
	--emb = f:readObject()
	f:close()

	--print(dic:size())
	--print(emb:size())

	-- load net
	print('load net...')
	net = IORNN:load(net_path)
	emb = net.L

	-- load gold
	print('load gold and context...')
	cases = load_gold()

	-- eval
	if rate_func_name == "random" then 
		rate_function = rate_random
	elseif rate_func_name == "nocontext" then
		rate_function = rate_wo_context
	elseif rate_func_name == "context" then
		rate_function = rate_context
	end
	print(eval(cases, rate_function))

else
	print('<dic_emb_path> <corpus dir> <net path> <rate func name>')
end
