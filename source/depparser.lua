require 'depstruct'
require 'utils'
require 'dict'
require 'xlua'
require 'dp_spec'

p = xlua.Profiler()

Depparser = {}
Depparser_mt = { __index = Depparser }

function Depparser:new(voca_dic, pos_dic, deprel_dic)
	local parser = { voca_dic = voca_dic, pos_dic = pos_dic, deprel_dic = deprel_dic}
	setmetatable(parser, Depparser_mt)
	return parser
end

function Depparser:load_possible_word_dr(path)
	self.word_deprel = {}
	for line in io.lines(path) do
		local comps = split_string(line)
		local word_id = self.voca_dic:get_id(comps[1])
		local cap_id = self.voca_dic:get_cap_feature(comps[1])
		local deprel_id = self.deprel_dic:get_id(comps[2])
		l = self.word_deprel[word_id* 10 + cap_id]
		if l == nil then 
			l = {}
			self.word_deprel[word_id* 10 + cap_id] = l
		end
		l[deprel_id] = 1
	end
end

function Depparser:clone_state(state) 
	local new_state = { n_words = state.n_words,
						stack = state.stack:clone(), stack_pos = state.stack_pos, 
						buffer = state.buffer:clone(), buffer_pos = state.buffer_pos,
						head = state.head:clone(), deprel = state.deprel:clone(), 
						score = state.score }
	return new_state
end

function Depparser:trans(sent, state, decision_scores, net)

	local i = nil; j = nil
	if state.stack_pos > 0 then
		i = state.stack[state.stack_pos]
	end
	if state.buffer_pos <= state.buffer:numel() then
		j = state.buffer[state.buffer_pos]
	end

	-- remove impossible deprel
	if i ~= nil and i > 1 then
		local word_id = sent.word_id[i]
		local cap_id = sent.cap_id[i]
		local drs = self.word_deprel[word_id*10 + cap_id]
		if drs then
			for k = 1,self.deprel_dic.size do
				if drs[k] == nil then
					--decision_scores[k] = 0
				end
			end
		end
	end

	if j ~= nil then
		local word_id = sent.word_id[j]
		local cap_id = sent.cap_id[j]
		local drs = self.word_deprel[word_id*10 + cap_id]
		if drs then
			for k = 1,self.deprel_dic.size do
				if drs[k] == nil then
					--decision_scores[k+self.deprel_dic.size] = 0
				end
			end
		end
	end

	-- remove impossible actions
	if i == nil or j == nil then
		decision_scores[{{1,2*self.deprel_dic.size}}]:fill(0)
	end
	if i == 1 then -- ROOT
		decision_scores[{{1,self.deprel_dic.size}}]:fill(0)
	end
	if j == nil then
		decision_scores[2*self.deprel_dic.size+1] = 0
	end

	local score,action = decision_scores:max(1)
	action = action[1]

	-- left arc
	if action <= self.deprel_dic.size then
		--print('la')
		local tree = net:merge_treelets(state.treelets[j], 
										state.treelets[i], action)
		state.treelets[j] = tree
		state.head[i] = j
		state.deprel[i] = action
		state.stack_pos = state.stack_pos - 1

	-- right arc
	elseif action <= 2*self.deprel_dic.size then
		--print('ra')
		action = action - self.deprel_dic.size
		local tree = net:merge_treelets(state.treelets[i], 
										state.treelets[j], action)
		state.treelets[i] = tree
		state.head[j] = i
		state.deprel[j] = action
		state.stack_pos = state.stack_pos - 1
		state.buffer[state.buffer_pos] = i

	else -- shift
		--print('sh')
		state.stack_pos = state.stack_pos + 1
		state.stack[state.stack_pos] = j
		state.buffer_pos = state.buffer_pos + 1
	end

	return state
end

function Depparser:parse(nets, sentbank)
	local statebanks = {}
	for j=1,#nets do
		statebanks[j] = {}
	end

-- init
	for i,sent in ipairs(sentbank) do
		local n_words = sent.n_words
		for j,bank in ipairs(statebanks) do
			local state = { n_words = n_words,
							stack = torch.LongTensor(n_words):fill(-1), 
							stack_pos = 1, 
							buffer = torch.linspace(1,n_words,n_words), 
							buffer_pos = 2,
							head = torch.LongTensor(n_words):fill(-1),
							deprel = torch.LongTensor(n_words):fill(-1),
							treelets = nets[j]:create_treelets(sent),
							score = 0
						}
			state.stack[1] = 1 -- ROOT

			local bank = statebanks[j]
			bank[#bank+1] = state
		end
	end

-- parse
	local not_done = torch.Tensor(#sentbank):fill(1)

	while not_done:sum() ~= 0 do
		-- collect state and sent & extract features
		local decision_scores = nil
		for j,statebank in ipairs(statebanks) do
			local active_states = {}
			for i,sent in ipairs(sentbank) do
				if not_done[i] == 1 then
					active_states[#active_states+1] = statebank[i]
				end
			end

			-- compute prediction score
			if decision_scores == nil then
				decision_scores = nets[j]:predict_action(active_states)
			else
				decision_scores:add(nets[j]:predict_action(active_states))
			end
		end
		decision_scores:div(#nets)

		-- process states
		local k = 1
		for i,sent in ipairs(sentbank) do
			if not_done[i] == 1 then
				for j,statebank in ipairs(statebanks) do
					local state = statebank[i]
					local decision_score = decision_scores[{{},k}]
					state = self:trans(sent, state, decision_score, nets[j])
					if state.buffer_pos > sent.n_words then
						not_done[i] = 0
						if state.stack_pos ~= 1 or state.stack[state.stack_pos] ~= 1 then -- not valid projective dep-struct
							state.head[state.head:eq(-1)] = 1
							state.deprel[state.deprel:eq(-1)] = self.deprel_dic:get_id(ROOT_LABEL)
						end
					end
				end
				k = k + 1
			end
		end
	end
	
	return statebanks[1]
end

function Depparser:extract_training_states(ds)
	-- init
	local n_words = ds.n_words
	local state = { n_words = n_words,
					stack = torch.LongTensor(n_words+1):fill(-1), 
					stack_pos = 1, 
					buffer = torch.linspace(1,n_words,n_words), 
					buffer_pos = 2,
					head = torch.LongTensor(n_words):fill(-1),
					deprel = torch.LongTensor(n_words):fill(-1),
					score = 0
				}
	state.stack[1] = 1 -- ROOT
	local states = {}

	-- extract 
	while state.buffer_pos <= n_words do
		states[#states+1] = self:clone_state(state)

		if state.stack_pos < 0 then 
			error('invalid dep-struct')
		end

		local i = nil
		if state.stack_pos > 0 then
			i = state.stack[state.stack_pos]
		end
		j = state.buffer[state.buffer_pos]

		local action = nil

		-- check if left-arc
		if i ~= 0 and ds.head_id[i] == j then
			action = ds.deprel_id[i]
			state.stack_pos = state.stack_pos - 1
			state.head[i] = j
			state.deprel[i] = ds.deprel_id[i]
		else
			-- check the condition of right-arc
			local ok = true
			for k = 1,ds.n_words do
				if ds.head_id[k] == j then
					if state.head[k] ~= j then 
						ok = false
						break
					end
				end
			end

			-- check if righ-arc
			if j > 0 and ds.head_id[j] == i and ok then 
				action = ds.deprel_id[j] + self.deprel_dic.size -- plus offset
				state.stack_pos = state.stack_pos - 1
				state.buffer[state.buffer_pos] = i
				state.head[j] = i
				state.deprel[j] = ds.deprel_id[j]

			else -- shift
				action = self.deprel_dic.size * 2 + 1 -- offset
				state.stack_pos = state.stack_pos + 1
				state.stack[state.stack_pos] = j
				state.buffer_pos = state.buffer_pos + 1
			end
		end	
		
		-- add examples
		states[#states].action = action
	end

	return states
end

function Depparser:load_treebank(path)
	local treebank = {}
	local raw = {}

	tokens = {}
	for line in io.lines(path) do
		--print(line)
		line = trim_string(line)
		if line == '' then
			if pcall( function() 
					ds,sent = Depstruct:create_from_strings(tokens, 
						self.voca_dic, self.pos_dic, self.deprel_dic) 
				end ) 
			then
				treebank[#treebank+1] = ds
				raw[#raw+1] = sent
			else 
				print('error')
				print(tokens)
			end
			tokens = {}
		else 
			tokens[#tokens+1] = line
		end
	end

	return treebank, raw
end

function Depparser:train(net, traintrebank_path, devtreebank_path, model_dir)
	print('load train treebank')
	local traintreebank,_ = self:load_treebank(traintrebank_path)
	
	net.update_L = TRAIN_UPDATE_L

	-- shuf the traintreebank
	print('shufing train treebank')
	local new_i = torch.randperm(#traintreebank)
	temp = {}
	for i = 1,#traintreebank do
		temp[i] = traintreebank[new_i[i]]
	end
	traintreebank = temp

	-- train
	local adagrad_config = {	weight_learningRate	= TRAIN_WEIGHT_LEARNING_RATE,
								voca_learningRate	= TRAIN_VOCA_LEARNING_RATE	}
	local adagrad_state = {}

	net:save(model_dir .. '/model_0')
	local prefix = model_dir..'/model'

	print('train net')
	adagrad_config, adagrad_state = net:train_with_adagrad(traintreebank, TRAIN_BATCHSIZE,
										TRAIN_MAX_N_EPOCHS, {lambda = TRAIN_LAMBDA, lambda_L = TRAIN_LAMBDA_L}, 
										prefix, adagrad_config, adagrad_state, 
										self, devtreebank_path)
	return net
end

-- should not call it directly when training, there's a mem-leak problem!!!
function Depparser:eval(nets, path, output)
	print('load data')
	local treebank, raw = self:load_treebank(path)

	print('parsing...')
	local parses = self:parse(nets, treebank)
	
	if output then
		print('print to '..output)
		local f = io.open(output, 'w')
		for i, parse in ipairs(parses) do
			local sent = raw[i]
			for j = 2,#sent do
				f:write((j-1)..'\t'..sent[j]..'\t_\t_\t_\t_\t'..(parse.head[j]-1)..'\t'..self.deprel_dic.id2word[parse.deprel[j]]..'\t_\t_\n')
			end
			f:write('\n')	
		end
		f:close()
	end

	-- compute scores
	local total = 0
	local label = 0
	local unlabel = 0

	for i,parse in ipairs(parses) do
		local gold = treebank[i]
		if parse.n_words ~= gold.n_words then
			error('not match')
		else
			total = total + parse.n_words - 1
			for j = 2,parse.n_words do
				if parse.head[j] == gold.head_id[j] then
					unlabel = unlabel + 1
					if parse.deprel[j] == gold.deprel_id[j] then
						label = label + 1
					end
				end
			end
		end
	end

	local LAS = label / total
	local UAS = unlabel / total
	local str = 'LAS = ' .. string.format("%.2f",LAS*100)..'\nUAS = ' ..string.format("%.2f",UAS*100)
	print(str)

	-- mail
	if EVAL_EMAIL_ADDR and self.mail_subject then
		os.execute('echo "'..str..'" | mail -s '..self.mail_subject..' '..EVAL_EMAIL_ADDR)
	end
end

