require 'dict'
require 'utils'
require 'tree'
require 'iornn'
require 'cutils'

torch.setnumthreads(1)

function load_gold(test_type)
	-- load human rates
	local cases = {}
	grammar = 'CCG'

	for line in io.lines(human_score_path) do
		local comps = split_string(line)

		if comps[2] == test_type then
			local str = nil
			if test_type == 'adjectivenouns' then
				str = {'(fa '..comps[4].. ' ' ..comps[5]..')', 
						'(fa '..comps[6]..' '..comps[7]..')' }
			elseif test_type == 'verbobjects' then 
				str = {'(fa '..comps[5].. ' (lex '..comps[4]..'))',
						'(fa '..comps[7].. ' (lex '..comps[6]..'))'}
			elseif test_type == 'compoundnouns' then
				str = {'(fa '..comps[4].. ' '..comps[5]..')', 
						'(fa '..comps[6]..' '..comps[7]..')' }
			end
			--print(str)
			local parse = {Tree:create_from_string(str[1])
								:to_torch_matrices(vocaDic, ruleDic, grammar),
							Tree:create_from_string(str[2])
								:to_torch_matrices(vocaDic, ruleDic, grammar) }


			local case_id = line --str[1] .. ' ; ' .. str[2]
			local human_rate = tonumber(comps[8])
			local level = tonumber(comps[3])

			if cases[case_id] == nil then
				cases[case_id] = {
						parse = parse,
						human_rates = {human_rate},
						level = level,
						compound = { 
								{vocaDic:get_id(comps[4]),vocaDic:get_id(comps[5])},
								{vocaDic:get_id(comps[6]),vocaDic:get_id(comps[7])}}
					}
			else
				local human_rates = cases[case_id].human_rates
				human_rates[#human_rates+1] = human_rate
			end
		end
	end
	--print(cases)
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

function rate_add(case) 
	local sem1 = net.L[{{},{case.compound[1][1]}}] + 
				net.L[{{},{case.compound[1][2]}}]
	local sem2 = net.L[{{},{case.compound[2][1]}}] + 
				net.L[{{},{case.compound[2][2]}}]
	return compute_score(sem1, sem2)[1]
end

function rate_multiply(case)
	local sem1 = torch.cmul(net.L[{{},{case.compound[1][1]}}],
							net.L[{{},{case.compound[1][2]}}])
	local sem2 = torch.cmul(net.L[{{},{case.compound[2][1]}}],
							net.L[{{},{case.compound[2][2]}}])
	return compute_score(sem1, sem2)[1]
end

function rate_iornn( case )
	local trees = net:parse(case.parse)
	local sem1 = trees[1].inner[{{},{1}}]
	local sem2 = trees[2].inner[{{},{1}}]
	return compute_score(sem1, sem2)[1]
end

if #arg == 4 then
	vocaDic_emb_path = arg[1]
	rule_path =  arg[2]
	human_score_path = arg[3]
	net_path = arg[4]

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

	local rules = {}
	for _,str in ipairs(ruleDic.id2word) do
		local comps = split_string(str, "[^ \t]+")
		local rule = {lhs = comps[1], rhs = {}}
		for i = 2,#comps do
			rule.rhs[i-1] = comps[i]
		end
		rules[#rules+1] = rule
	end

	-- load net
	print('load net...')
	net = IORNN:load(net_path)
	--net.L = emb

	-- test
	func_list = {
					['add'] = rate_add,
					['multiply'] = rate_multiply,
					['iornn'] = rate_iornn 
				}

	for _,test_type in ipairs({'verbobjects', 'compoundnouns', 'adjectivenouns'}) do
		print('===== ' .. test_type .. ' =====')
		cases = load_gold(test_type)

		for f_name,rate_function in pairs(func_list) do
			rho = eval(cases, rate_function)
			print(f_name .. ' : ' .. rho)
		end
	end

else
	print('<vocaDic_emb_path> <rule path> <data path> <net path>')
end
