-- evluate IORNN on Microsoft Research Sentence Completion Challenge data

require 'dict'
require 'utils'
require 'tree'
require 'iornn'
require 'cutils'

torch.setnumthreads(1)
grammar = 'CFG'

function load_gold()
	-- load human rates
	local cases = {}
	
	local i = -1
	local case = nil
	local fault = false
	for line in io.lines(data_path) do
		i = i + 1
		local j = math.mod(i,5) + 1
		if j == 1 then 
			case = {
				parse = Tree:create_from_string(line)
						:to_torch_matrices(vocaDic, ruleDic, grammar),
				choice = {}
			}
		elseif j == 2 then
			case.position = tonumber(line)
		elseif j == 3 then
			case.answer = vocaDic:get_id(line)
			if case.answer == 1 then
				fault = true
			end
		elseif j == 4 then
			local comps = split_string(line, '[^|]+')
			for k = 1,5 do
				case.choice[k] = vocaDic:get_id(trim_string(comps[k]))
				if case.choice[k] == 1 then 
					faults = true
				end
			end
		elseif j == 5 then
			--if faults == false then
				cases[#cases+1] = case
			--end
			faults = false
		end
	end

	print(#cases)
	--print(faults)
	return cases
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
	local n_correct = 0
	for _,case in pairs(cases) do
		local choice = rate_function(case)
		--print('----------')
		--print(choice)
		--print(case.answer)
		if choice == case.answer then 
			n_correct = n_correct + 1
		end
	end
	return n_correct / #cases
end

function rate_random( case )
	return math.random()
end

function rate_totsim(case)
	local scores = torch.zeros(#case.choice)
	local tree = case.parse
	local n_leaves = 0
	for i = 1,tree.n_nodes do
		if tree.n_children[i] == 0 then
			n_leaves = n_leaves + 1
			if n_leaves ~= case.position then
				for j,c in ipairs(case.choice) do
					scores[j] = scores[j] + 
							compute_score(net.L[{{},{tree.word_id[i]}}], net.L[{{},{c}}])[1]
				end
			end
		end
	end

	_,t = scores:max(1)
	return case.choice[t[1]]
end

function rate_iornn( case )
	local tree = net:parse({case.parse})[1]

	local pos
	local n_leaves = 0
	for i = 1,tree.n_nodes do
		if tree.n_children[i] == 0 then
			n_leaves = n_leaves + 1
			if n_leaves == case.position then
				pos = i
				break
			end
		end
	end
	local outer = tree.outer[{{},{pos}}]
	
	local best_score = -1e100
	local best_choice = 0
	for _,c in ipairs(case.choice) do
		local score = net.Ws * tanh(
						(net.Wwo*outer)
						:add(net.Wwi*net.L[{{},{c}}])
						:add(net.bw))
		score = score[{1,1}]

		if score > best_score then
			best_choice = c
			best_score = score
		end
	end

	return best_choice
end

if #arg == 4 then
	vocaDic_emb_path = arg[1]
	rule_path =  arg[2]
	data_path = arg[3]
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
	ruleDic = Dict:new(grammar_template)
	ruleDic:load(rule_path)

	-- load data
	print('load data')
	cases = load_gold(test_type)

	-- load net
	print('load net...')
	net = IORNN:load(net_path) 
	net.L = emb

	-- test
	func_list = {
					['totsim'] = rate_totsim, -- total similarity
					['iornn'] = rate_iornn 
				}


	for f_name,rate_function in pairs(func_list) do
		rho = eval(cases, rate_function)
		print(f_name .. ' : ' .. rho)
	end

else
	print('[vocaDic] [rule] [data] [net]')
end
