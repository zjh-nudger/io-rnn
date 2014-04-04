require 'depstruct'
require 'svm'
require 'utils'
require 'dict'
require 'xlua'

p = xlua.Profiler()

Depparser = {}
Depparser_mt = { __index = Depparser }

torch.setnumthreads(1)

-- setting for beam-search
Depparser.MAX_NSTATES = 1 -- any larger than 1 performs worse

Depparser.LA_ID = 1
Depparser.RA_ID = 2
Depparser.SH_ID = 3

function Depparser:new(voca_dic, pos_dic, deprel_dic)
	local parser = { voca_dic = voca_dic, pos_dic = pos_dic, 
					deprel_dic = deprel_dic }
	setmetatable(parser, Depparser_mt)
	return parser
end

function Depparser:trans(sent, state, decision_scores)

	local i = nil; j = nil
	if state.stack_pos > 0 then
		i = state.stack[state.stack_pos]
	end
	if state.buffer_pos <= state.buffer:numel() then
		j = state.buffer[state.buffer_pos]
	end

	-- remove impossible actions
	if i == nil or j == nil then
		decision_scores[{{1,2*self.deprel_dic.size}}]:fill(0)
	end
	if i == 0 then
		decision_scores[{{1,self.deprel_dic.size}}]:fill(0)
	end
	if j == nil then
		decision_scores[2*self.deprel_dic+1] = 0
	end

	local _,action = decision_scores:max(1)
	action = action[1]

	-- left arc
	if action <= self.deprel_dic.size then
		local tree = self.net:merge_treelets(state.treelets[j], 
											state.treelets[i], action)
		state.treelets[j] = tree
		state.head[i] = j
		state.deprel[i] = action
		state.stack_pos = state.stack_pos - 1

	-- right arc
	elseif action <= 2*self.deprel_dic.size then
		action = action - self.deprel_dic.size
		local tree = self.net:merge_treelets(state.treelets[i], 
											state.treelets[j], action)
		state.treelets[i] = tree
		state.head[j] = i
		state.deprel[j] = action
		state.stack_pos = state.stack_pos - 1
		state.buffer[state.buffer_pos] = i

	else -- shift
		state.stack_pos = state.stack_pos + 1
		state.stack[state.stack_pos] = j
		state.buffer_pos = state.buffer_pos + 1
	end

	return state
end

function Depparser:parse(sentbank)
	local statebank = {}

-- init
	for i,sent in ipairs(sentbank) do
		local n_words = sent.n_words
		local state = { stack = torch.LongTensor(n_words+1):fill(-1), 
						stack_pos = 1, 
						buffer = torch.linspace(1,n_words,n_words), 
						buffer_pos = 1,
						head = torch.LongTensor(n_words):fill(-1),
						deprel = torch.LongTensor(n_words):fill(-1),
						treelets = self.net:create_treelets(sent),
						score = 0
					}
		state.stack[1] = 0 -- ROOT
		statebank[#statebank+1] = state
	end

-- parse
	local not_done = torch.Tensor(#statebank):fill(1)

	while not_done:sum() ~= 0 do
		-- collect state and sent & extract features
		local active_states = {}
		for i,sent in ipairs(sentbank) do
			if not_done[i] == 1 then
				active_states[#active_states+1] = statebank[i]
			end
		end

		-- compute prediction score
		p:start('decision') 
		local decision_scores = self.net:predict_action(active_states)
		p:lap('decision')

		-- process states
		p:start('trans')
		local k = 1
		for i,sent in ipairs(sentbank) do
			if not_done[i] == 1 then
				local state = statebank[i]
				local decision_score = decision_scores[{{},k}]
				k = k + 1

				state = self:trans(sent, state, decision_score)
			end
		end
		p:lap('trans')
		p:printAll()
	end
	
	return statebank
end

function Depparser:extract_training_states(ds)
	-- init
	local n_words = ds.n_words
	local state = { stack = torch.LongTensor(n_words+1):fill(-1), 
					stack_pos = 1, 
					buffer = torch.linspace(1,n_words,n_words), 
					buffer_pos = 1,
					head = torch.LongTensor(n_words):fill(-1),
					deprel = torch.LongTensor(n_words):fill(-1),
					score = 0
				}
	state.stack[1] = 0 -- ROOT
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

function Depparser:load_train_treebank(treebank_path)
	local treebank = {}
	local i = 0

	local tokens = {}
	for line in io.lines(treebank_path) do
		line = trim_string(line)
		if line == '' then
			if pcall( function() 
				ds,sent = Depstruct:create_from_strings(tokens, 
					self.voca_dic, self.pos_dic, self.deprel_dic) 
				end ) 
			then
				local states = self:extract_training_states(ds)
				local tree = ds:to_torch_matrix_tree()
				tree.states = states
				treebank[#treebank+1] = tree
			else 
				print('error')
				print(tokens)
			end
			tokens = {}
		else 
			tokens[#tokens+1] = line
		end
	end
	return treebank
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

function Depparser:eval(path, output)
	local treebank, raw = self:load_treebank(path)
	local parses = self:parse(treebank)
	
	local f = io.open(output, 'w')
	for i, parse in ipairs(parses) do
		local sent = raw[i]
		if parse.head == nil then 
			parse = { 	head = torch.zeros(#sent) , 
						deprel = torch.zeros(#sent):fill(self.deprel_dic:get_id('root')) } 
		end

		for j = 1,#sent do
			f:write(j..'\t'..sent[j]..'\t_\t_\t_\t_\t'..parse.head[j]..'\t'..self.deprel_dic.id2word[parse.deprel[j]]..'\t_\t_\n')
		end
		f:write('\n')	
	end
	f:close()
end

--[[
if #arg == 1 then
	print('load dics')
	local voca_dic = Dict:new(collobert_template)
	voca_dic:load('../data/wsj-dep/stanford/dic/collobert/words.lst')
 
	local pos_dic = Dict:new()
	pos_dic:load("../data/wsj-dep/stanford/dic/pos.lst")

	local deprel_dic = Dict:new()
	deprel_dic:load('../data/wsj-dep/stanford/dic/deprel.lst')

	print('training...')
	local parser = Depparser:new(voca_dic, pos_dic, deprel_dic)
	parser:train('../data/wsj-dep/stanford/data/train.conll')
	collectgarbage()

	parser:eval('../data/wsj-dep/stanford/data/dev.conll', arg[1])
else
	print('[output]')
end
]]
--[[
print('take a sentence')
local sent = 'This is a long sentence .'
sent = split_string(sent)
for i,tok in ipairs(sent) do
	sent[i] = voca_dic:get_id(tok)
end
sent = { word_id = torch.Tensor(sent):long() }
sent.n_words = sent.word_id:numel()

print('parsing...')
parser:parse(sent)
for i = 1,10 do
	print(parser.states[i].head)
end
]]
