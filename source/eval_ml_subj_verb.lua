require 'dict'
require 'utils'
require 'cutils'
require 'tree'
require 'iornn'

torch.setnumthreads(1)

function load_gold()
	-- load human rates
	local cases = {}
	local iter = 0
	for line in io.lines(human_score_path) do
		iter = iter + 1
		--if iter > 10 then break end

		local comps = split_string(line)

		local str = '(TOP (NP '..comps[3].. ') (VP '..comps[2]..'))'
		local parse = Tree:create_from_string(str):to_torch_matrices(dic, 1)

		local str = '(TOP (NP '..comps[3].. ') (VP '..comps[4]..'))'
		local parse_lm = Tree:create_from_string(str)
							:to_torch_matrices(dic, 1)

		local case_id = line --str .. comps[4]
		local human_rate = tonumber(comps[5])
		local is_high = comps[6] == 'high'

		if cases[case_id] == nil then
			cases[case_id] = {
						parse = parse,
						parse_lm = parse_lm,
						verb_id = dic:get_id(comps[2]),
						noun_id = dic:get_id(comps[3]),
						landmark_id = dic:get_id(comps[4]),
						human_rates = {human_rate},
						is_high = is_high 
					}
		else
			local human_rates = cases[case_id].human_rates
			human_rates[#human_rates+1] = human_rate
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
	local human_rates = {}
	local cand_rates = {}
	
	for _,case in pairs(cases) do
		human_rates[#human_rates+1] = torch.Tensor(case.human_rates):mean()
		cand_rates[#cand_rates+1] = rate_function(case)
	end
	return compute_rho(torch.Tensor(human_rates), torch.Tensor(cand_rates))
end

function rate_random( case )
	return math.random()
end

function rate_nocomp(case) 
	local sem1 = net.L[{{},{case.verb_id}}]
	local sem2 = net.L[{{},{case.landmark_id}}]:clone()
	return compute_score(sem1, sem2)[1]
end

function rate_add(case) 
	local sem1 = net.L[{{},{case.verb_id}}] + net.L[{{},{case.noun_id}}]
	local sem2 = net.L[{{},{case.landmark_id}}]:clone()
	return compute_score(sem1, sem2)[1]
end

function rate_multiply(case)
	local sem1 = torch.cmul(net.L[{{},{case.verb_id}}], 
							net.L[{{},{case.noun_id}}])
	local sem2 = net.L[{{},{case.landmark_id}}]:clone()
	return compute_score(sem1, sem2)[1]
end

function rate_iornn( case )
	local tree = net:parse({case.parse})[1]
	local tree_lm = net:parse({case.parse_lm})[1]
	local sem1 = tree.inner[{{},{1}}]
	--local sem2 = tree_lm.inner[{{},{1}}]
	local sem2 = tanh(net.L[{{},{case.landmark_id}}])
	return compute_score(sem1, sem2)[1]
end

if #arg == 3 then
	dic_emb_path = arg[1]
	human_score_path = arg[2]
	net_path = arg[3]

	-- load dic & emb
	print('load dic & emb...')
	f = torch.DiskFile(dic_emb_path, 'r')
	dic = f:readObject()
	setmetatable(dic, Dict_mt)
	f:close()

	-- load net
	print('load net...')
	net = IORNN:load(net_path)
	emb = net.L

	-- load gold
	print('load gold and context...')
	cases = load_gold()

	-- eval
	rate_func_list = {
						['nocomp'] = rate_nocomp,
						['add'] = rate_add,
						['multiply'] = rate_multiply,
						['iornn'] = rate_iornn
					}
	for fname, rate_function in pairs(rate_func_list) do
		print(fname .. ' : ' .. eval(cases, rate_function))
	end

else
	print('<dic_emb_path> <corpus dir> <net path>')
end