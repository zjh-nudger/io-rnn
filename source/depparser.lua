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

FEAT_FORM	= 1
FEAT_POS	= 2
FEAT_DRLDEP	= 3
FEAT_DRRDEP	= 4
FEAT_FORM_H	= 5

function Depparser:get_feature_stack_buffer(sent, state, position, features, mem)
	if mem == state.stack and position > 0 and position <= state.stack_pos
	or mem == state.buffer and position <= sent.n_words and position >= state.buffer_pos then
		local i = mem[position]
		if features[FEAT_FORM] ~= nil then
			if i == 0 then features[FEAT_FORM] = self.voca_dic.size + 1 
			else features[FEAT_FORM] = sent.word_id[i] end
		end
		if features[FEAT_POS] ~= nil then
			if i == 0 then features[FEAT_POS] = self.pos_dic.size + 1
			else features[FEAT_POS] = sent.pos_id[i] end
		end
		if features[FEAT_DRLDEP] ~= nil then
			for k = 1,i-1 do
				if state.head[k] == i then
					features[FEAT_DRLDEP] = state.deprel[k]
					break
				end
			end
		end
		if features[FEAT_DRRDEP] ~= nil then
			for k = sent.n_words,i+1,-1 do
				if state.head[k] == i then
					features[FEAT_DRRDEP] = state.deprel[k]
					break
				end
			end
		end
		if features[FEAT_FORM_H] ~= nil and i > 0 then
			local h = state.head[i]
			if h == 0 then features[FEAT_FORM_H] = self.voca_dic.size + 1 
			elseif h > 0 then features[FEAT_FORM_H] = sent.word_id[h] end
		end
	end

	return features
end

function Depparser:get_feature(sent, state)
	local fstop		= {[FEAT_FORM] = -1, [FEAT_POS] = -1, [FEAT_DRLDEP] = -1, [FEAT_DRRDEP] = -1, [FEAT_FORM_H] = 1}
	local fstop1	= {[FEAT_POS] = -1}
	local fbtop		= {[FEAT_FORM] = -1, [FEAT_POS] = -1, [FEAT_DRLDEP] = -1, [FEAT_DRRDEP] = -1}
	local fbtop1	= {[FEAT_FORM] = -1, [FEAT_POS] = -1}
	local fbtop2	= {[FEAT_POS] = -1}
	local fbtop3	= {[FEAT_POS] = -1}

	self:get_feature_stack_buffer(sent, state, state.stack_pos, fstop, state.stack)
	self:get_feature_stack_buffer(sent, state, state.stack_pos-1, fstop1, state.stack)

	self:get_feature_stack_buffer(sent, state, state.buffer_pos, fbtop, state.buffer)
	self:get_feature_stack_buffer(sent, state, state.buffer_pos+1, fbtop1, state.buffer)
	self:get_feature_stack_buffer(sent, state, state.buffer_pos+2, fbtop2, state.buffer)
	self:get_feature_stack_buffer(sent, state, state.buffer_pos+3, fbtop3, state.buffer)

	local fs = {fstop, fstop1, fbtop, fbtop1, fbtop2, fbtop3}

-- store in vectors
	local d = {}
	d[1] = nil
	d[2] = {}
	local index = torch.IntTensor(30)
	local i = 0
	local offset = 0

	-- singular features
	for _,f in ipairs(fs) do
		for k = 1,5 do -- FEAT_FORM,...,FEAT_FORM_H
			if f[k] ~= nil and f[k] >= 0 then
				i = i + 1
				index[i] = offset + f[k]
			end
			if f[k] ~= nil then 
				if 		k == FEAT_FORM 		then offset = offset + self.voca_dic.size + 1
				elseif	k == FEAT_POS		then offset = offset + self.pos_dic.size + 1
				elseif	k == FEAT_DRLDEP	then offset = offset + self.deprel_dic.size 
				elseif 	k == FEAT_DRRDEP	then offset = offset + self.deprel_dic.size
				elseif 	k == FEAT_FORM_H	then offset = offset + self.voca_dic.size + 1
				end
			end
		end
	end

	-- merged features
	if fstop[FEAT_POS] >= 0 and fbtop[FEAT_POS] >= 0 then
		i = i + 1
		index[i] = (fstop[FEAT_POS]-1) * (self.pos_dic.size+1) + fbtop[FEAT_POS]
	end
	offset = offset + (self.pos_dic.size+1)*(self.pos_dic.size+1)

	if fstop[FEAT_POS] >= 0 and fstop1[FEAT_POS] >= 0 and fbtop[FEAT_POS] >= 0 then
		i = i + 1
		index[i] = (index[i-1]-1)*(self.pos_dic.size+1) + fstop1[FEAT_POS]
	end
	offset = offset + (self.pos_dic.size+1)*(self.pos_dic.size+1)*(self.pos_dic.size+1)

	if fstop[FEAT_POS] >= 0 and fbtop[FEAT_POS] >= 0 and fbtop1[FEAT_POS] >= 0 then
		i = i + 1
		index[i] = (index[i-1]-1)*(self.pos_dic.size+1) + fbtop1[FEAT_POS]
	end
	offset = offset + (self.pos_dic.size+1)*(self.pos_dic.size+1)*(self.pos_dic.size+1)

	if fbtop[FEAT_POS] >= 0 and fbtop1[FEAT_POS] >= 0 and fbtop2[FEAT_POS] >= 0 then
		i = i + 1
		index[i] = ((fbtop[FEAT_POS]-1)*(self.pos_dic.size+1) + fbtop1[FEAT_POS]-1)*(self.pos_dic.size+1) + fbtop2[FEAT_POS]
	end
	offset = offset + (self.pos_dic.size+1)*(self.pos_dic.size+1)*(self.pos_dic.size+1)

	if fbtop1[FEAT_POS] >= 0 and fbtop2[FEAT_POS] >= 0 and fbtop3[FEAT_POS] >= 0 then
		i = i + 1
		index[i] = ((fbtop1[FEAT_POS]-1)*(self.pos_dic.size+1) + fbtop2[FEAT_POS]-1)*(self.pos_dic.size+1) + fbtop3[FEAT_POS]
	end
	offset = offset + (self.pos_dic.size+1)*(self.pos_dic.size+1)*(self.pos_dic.size+1)

	if fstop[FEAT_POS] >= 0 and fstop[FEAT_DRLDEP] >= 0 and fstop[FEAT_DRRDEP] >0 then
		i = i + 1
		index[i] = ((fstop[FEAT_POS]-1)*self.deprel_dic.size + fstop[FEAT_DRLDEP] - 1)*self.deprel_dic.size + fstop[FEAT_DRRDEP]
	end
	offset = offset + (self.pos_dic.size+1)*self.deprel_dic.size*self.deprel_dic.size
	
	if fbtop[FEAT_POS] >= 0 and fbtop[FEAT_DRLDEP] >= 0 and fbtop[FEAT_DRRDEP] >0 then
		i = i + 1
		index[i] = ((fbtop[FEAT_POS]-1)*self.deprel_dic.size + fbtop[FEAT_DRLDEP] - 1)*self.deprel_dic.size + fbtop[FEAT_DRRDEP]
	end
	offset = offset + (self.pos_dic.size+1)*self.deprel_dic.size*self.deprel_dic.size

	d[2][1] = index[{{1,i}}]:clone()
	d[2][2] = torch.FloatTensor(i):fill(1)

	return d
end

function Depparser:extract_training_examples(ds, examples)
	local examples = examples or {}

	-- init
	local n_words = ds.n_words
	local state = { stack = torch.LongTensor(n_words+1):fill(-1), stack_pos = 1, 
					buffer = torch.linspace(1,n_words,n_words), buffer_pos = 1,
					head = torch.LongTensor(n_words):fill(-1),
					deprel = torch.LongTensor(n_words):fill(-1),
					score = 0
				}
	state.stack[1] = 0 -- ROOT

	-- extract 
	while state.buffer_pos <= n_words do

		if state.stack_pos < 0 then 
			error('invalid dep-struct')
		end

		local i = nil
		if state.stack_pos > 0 then
			i = state.stack[state.stack_pos]
		end
		j = state.buffer[state.buffer_pos]

		local d = self:get_feature(ds, state)
		local class = nil

		-- check if left-arc
		if i ~= 0 and ds.head_id[i] == j then
			class = ds.deprel_id[i]
			state.stack_pos = state.stack_pos - 1
			state.head[i] = j
			--print('la')
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
				class = ds.deprel_id[j] + self.deprel_dic.size -- plus offset
				state.stack_pos = state.stack_pos - 1
				state.buffer[state.buffer_pos] = i
				state.head[j] = i
				--print('ra')

			else -- shift
				class = self.deprel_dic.size * 2 + 1 -- offset
				state.stack_pos = state.stack_pos + 1
				state.stack[state.stack_pos] = j
				state.buffer_pos = state.buffer_pos + 1
				--print('sh')
			end
		end	
		
		-- add examples
		d[1] = class
		examples[#examples+1] = d
	end

	return examples
end

function Depparser:train(treebank_path)
-- extract examples
	local examples = {}
	local i = 0

	local tokens = {}
	for line in io.lines(treebank_path) do
		line = trim_string(line)
		if line == '' then
			if pcall( function() ds,sent = Depstruct:create_from_strings(tokens, self.voca_dic, self.pos_dic, self.deprel_dic) end ) then
				examples = self:extract_training_examples(ds, examples)
				if math.mod(i,1000) == 0 then print(i) end
				i = i + 1
			else 
				print('error')
				print(tokens)
			end
			tokens = {}
		else 
			tokens[#tokens+1] = line
		end
	end
	
	self.label_map = {}
	local t = 0
	for i,d in ipairs(examples) do
		if self.label_map[d[1]] == nil then 
			t = t + 1
			self.label_map[d[1]] = t
		end
		if t == 2*self.deprel_dic.size + 1 then break end
	end
--	print(self.label_map)

-- train svm
	print(#examples)
	self.model = liblinear.train(examples, '-s 4 -c 0.1')
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
