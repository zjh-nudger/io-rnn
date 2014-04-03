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
	local parser = { voca_dic = voca_dic, pos_dic = pos_dic, deprel_dic = deprel_dic }
	setmetatable(parser, Depparser_mt)
	return parser
end

function Depparser:clone_state(state) 
	local new_state = { stack = state.stack:clone(), stack_pos = state.stack_pos, 
						buffer = state.buffer:clone(), buffer_pos = state.buffer_pos,
						head = state.head:clone(), deprel = state.deprel:clone(), 
						score = state.score }
	return new_state
end

function Depparser:trans(sent, state)
	local new_states = {}
	
	local i = nil; j = nil
	if state.stack_pos > 0 then
		i = state.stack[state.stack_pos]
	end
	if state.buffer_pos <= state.buffer:numel() then
		j = state.buffer[state.buffer_pos]
	end

	-- left arc
	if i ~= nil and j ~= nil and i ~= 0 then
		for l = 1,self.deprel_dic.size do
			local class = self.label_map[l]
			if class ~= nil then
				local state_la = self:clone_state(state)
				state_la.head[i] = j
				state_la.deprel[i] = l
				state_la.stack_pos = state_la.stack_pos - 1
				state_la.score = state.score + state.decision_scores[class]
				new_states[#new_states+1] = state_la
			end
		end
	end

	-- right arc
	if i ~= nil and j ~= nil then
		for l = 1,self.deprel_dic.size do
			local class = self.label_map[l+self.deprel_dic.size]
			if class ~= nil then
				local state_ra = self:clone_state(state)
				state_ra.head[j] = i
				state_ra.deprel[j] = l
				state_ra.stack_pos = state_ra.stack_pos - 1
				state_ra.buffer[state_ra.buffer_pos] = i
				state_ra.score = state.score + state.decision_scores[self.label_map[l+self.deprel_dic.size]]
				new_states[#new_states+1] = state_ra
			end
		end
	end

	-- shift
	if j ~= nil then
		local state_sh = self:clone_state(state)
		state_sh.stack_pos = state_sh.stack_pos + 1
		state_sh.stack[state_sh.stack_pos] = j
		state_sh.buffer_pos = state_sh.buffer_pos + 1
		state_sh.score = state.score + state.decision_scores[self.label_map[2*self.deprel_dic.size+1]]
		new_states[#new_states+1] = state_sh
	end

	return new_states
end

function Depparser:parse(sentbank)
	local statebank = {}

-- init
	for i,sent in ipairs(sentbank) do
		local n_words = sent.n_words
		local state = { stack = torch.LongTensor(n_words+1):fill(-1), stack_pos = 1, 
						buffer = torch.linspace(1,n_words,n_words), buffer_pos = 1,
						head = torch.LongTensor(n_words):fill(-1),
						deprel = torch.LongTensor(n_words):fill(-1),
						score = 0
					}
		state.stack[1] = 0 -- ROOT
		local states = { state }
		statebank[#statebank+1] = states
	end

-- parse
	local not_done = torch.Tensor(#statebank):fill(1)

	while not_done:sum() ~= 0 do
		-- collect state and sent & extract features
		p:start('feature')
		local data = {}
		for i,sent in ipairs(sentbank) do
			if not_done[i] == 1 then
				for j,state in ipairs(statebank[i]) do
					data[#data+1] = self:get_feature(sent, state)
				end
			end
		end
		p:lap('feature')

		-- compute prediction score
		p:start('decision') 
		_,_,dec = liblinear.predict(data, self.model)
		dec = safe_compute_softmax(dec:t()):t():log()
		local k = 1
		for i,sent in ipairs(sentbank) do
			if not_done[i] == 1 then
				for j,state in ipairs(statebank[i]) do
					state.decision_scores = dec[{k,{}}]
					k = k + 1
				end
			end
		end
		p:lap('decision')
	
		-- process sentences 
		p:start('trans')
		for i,states in ipairs(statebank) do
			local sent = sentbank[i]

			if not_done[i] == 1 then
				not_done[i] = 0 
				local new_states = {}
				local scores = torch.Tensor(1000 * #states)

				for _,state in ipairs(states) do
					if state.buffer_pos == sent.n_words + 1 then -- buff empty
						if state.stack_pos == 1 and state.stack[state.stack_pos] == 0 then -- valid projective dep-struct
							new_states[#new_states+1] = state
							scores[#new_states] = state.score
						end
					else 
						not_done[i] = 1
						local local_new_states = self:trans(sent, state)
						for _,st in ipairs(local_new_states) do
							new_states[#new_states+1] = st
							scores[#new_states] = st.score
						end
					end
				end
	
				-- extract best states
				local n = #new_states
				states = {}
				if n > 0 then
					top_scores,top_id = scores[{{1,n}}]:sort(1,true)
					for i = 1,math.min(Depparser.MAX_NSTATES,n) do
						states[i] = new_states[top_id[i]]
					end		
				end
				statebank[i] = states
			end
		end
		p:lap('trans')
		p:printAll()
	end
	
	-- get the best parses
	local ret = {}
	for i,states in ipairs(statebank) do
		if #states > 0 then 
			ret[i] = states[1]
		else 
			ret[i] = {}
		end
	end

	return ret
end

function Depparser:extract_training_states(ds)
	-- init
	local n_words = ds.n_words
	local state = { stack = torch.LongTensor(n_words+1):fill(-1), stack_pos = 1, 
					buffer = torch.linspace(1,n_words,n_words), buffer_pos = 1,
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
			if pcall( function() ds,sent = Depstruct:create_from_strings(tokens, self.voca_dic, self.pos_dic, self.deprel_dic) end ) then
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
			if pcall( function() ds,sent = Depstruct:create_from_strings(tokens, self.voca_dic, self.pos_dic, self.deprel_dic) end ) then
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
