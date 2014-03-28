require 'depstruct'
require 'svm'
require 'utils'
require 'dict'

Depparser = {}
Depparser_mt = { __index = Depparser }

torch.setnumthreads(1)

-- setting for beam-search
Depparser.MAX_NSTATES = 32
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
						head = state.head:clone(), 
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

	local d = self:get_feature(sent, state)
	_,_,dec = liblinear.predict({d}, self.model)

	-- left arc
	if i ~= nil and j ~= nil and i ~= 0 then
		local state_la = self:clone_state(state)
		state_la.head[i] = j
		state_la.stack_pos = state_la.stack_pos - 1
		state_la.score = state.score + dec[{1,self.label_map[Depparser.LA_ID]}]
		new_states[#new_states+1] = state_la
	end

	-- right arc
	if i ~= nil and j ~= nil then
		local state_ra = self:clone_state(state)
		state_ra.head[j] = i
		state_ra.stack_pos = state_ra.stack_pos - 1
		state_ra.buffer[state_ra.buffer_pos] = i
		state_ra.score = state.score + dec[{1,self.label_map[Depparser.RA_ID]}]
		new_states[#new_states+1] = state_ra
	end

	-- shift
	if j ~= nil then
		local state_sh = self:clone_state(state)
		state_sh.stack_pos = state_sh.stack_pos + 1
		state_sh.stack[state_sh.stack_pos] = j
		state_sh.buffer_pos = state_sh.buffer_pos + 1
		state_sh.score = state.score + dec[{1,self.label_map[Depparser.SH_ID]}]
		new_states[#new_states+1] = state_sh
	end

	return new_states
end

function Depparser:parse(sent)
-- init
	local n_words = sent.n_words
	local state = { stack = torch.LongTensor(n_words+1):fill(-1), stack_pos = 1, 
					buffer = torch.linspace(1,n_words,n_words), buffer_pos = 1,
					head = torch.LongTensor(n_words):fill(-1),
					score = 0
				}
	state.stack[1] = 0 -- ROOT
	local states = { state }

-- parse
	local stop = false
	while not stop do
		local new_states = {}
		local scores = torch.Tensor(3*#states)
		stop = true

		for _,state in ipairs(states) do
			if state.buffer_pos == sent.n_words + 1 then -- buff empty
				if state.stack_pos == 1 and state.stack[state.stack_pos] == 0 then -- valid projective dep-struct
					new_states[#new_states+1] = state
					scores[#new_states] = state.score
				end
			else 
				stop = false
				local local_new_states = self:trans(sent, state)
				for _,st in ipairs(local_new_states) do
					new_states[#new_states+1] = st
					scores[#new_states] = st.score
				end
			end
		end
	
		-- extract best states
		local n = #new_states
		top_scores,top_id = scores[{{1,n}}]:sort(1,true)
		states = {}
		for i = 1,math.min(Depparser.MAX_NSTATES,n) do
			states[i] = new_states[top_id[i]]
		end		
	end
	self.states = states
end

function Depparser:get_feature(sent, state)
	local feature = torch.Tensor(7):fill(-1)

	if state.stack_pos > 0 then
		local i = state.stack[state.stack_pos]
		-- STK[0], LDEP[0], RDEP[0]
		if i == 0 then --ROOT
			feature[1] = self.voca_dic:size() + 1
		else 
			feature[1] = sent.word_id[i]
		end
		for k = 1,i-1 do
			if state.head[k] == i then
				feature[2] = 1
				break
			end
		end
		for k = sent.n_words,i+1,-1 do
			if state.head[k] == i then
				feature[3] = 1
				break
			end
		end
	end 

	if state.buffer_pos <= sent.n_words then
		local j = state.buffer[state.buffer_pos]
		-- BUF[0], LDEP[0], RDEP[0]
		if j == 0 then -- ROOT
			feature[4] = self.voca_dic:size() + 1
		else
			feature[4] = sent.word_id[j]
		end
		for k = 1,j-1 do
			if state.head[k] == j then
				feature[5] = 1
				break
			end
		end
		for k = sent.n_words,j+1,-1 do
			if state.head[k] == j then
				feature[6] = 1
				break
			end
		end
	end

	if state.buffer_pos <= sent.n_words -1 then
		feature[7] = sent.word_id[state.buffer[state.buffer_pos+1]]
	end

	local d = {}
	d[1] = nil
	d[2] = {}
	local n = feature:ge(0):double():sum()
	d[2][1] = torch.IntTensor(n)
	d[2][2] = torch.FloatTensor(n):fill(1)

	local k = 0
	local offset = 0
	for j = 1,feature:numel() do
		if feature[j] >= 0 then
			k = k + 1
			d[2][1][k] = offset + feature[j]
		end
		if j == 1 or j == 4 or j == 7 then 
			offset = offset + self.voca_dic:size() + 1
		else
			offset = offset + 1
		end
	end
	return d
end

function Depparser:extract_training_examples(ds, examples)
	local examples = examples or {}

	-- init
	local n_words = ds.n_words
	local state = { stack = torch.LongTensor(n_words+1):fill(-1), stack_pos = 1, 
					buffer = torch.linspace(1,n_words,n_words), buffer_pos = 1,
					head = torch.LongTensor(n_words):fill(-1),
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
			class = Depparser.LA_ID
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
				class = Depparser.RA_ID
				state.stack_pos = state.stack_pos - 1
				state.buffer[state.buffer_pos] = i
				state.head[j] = i
				--print('ra')

			else -- shift
				class = Depparser.SH_ID
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

function Depparser:train_liblinear_classifier(treebank)
-- extract examples
	local examples = {}
	for i,ds in ipairs(treebank) do
		examples = self:extract_training_examples(ds, examples)
		if math.mod(i,1000) == 0 then print(i) end
	end

	self.label_map = {}
	local t = 0
	for i,d in ipairs(examples) do
		if self.label_map[d[1]] == nil then 
			t = t + 1
			self.label_map[d[1]] = t
		end
		if t == 3 then break end
	end
	

-- train svm
	print(#examples)
	self.model = liblinear.train(examples, '-s 4')
end

function load_treebank(path, voca_dic, pos_dic, deprel_dic)
	local treebank = {}

	tokens = {}
	for line in io.lines(path) do
		--print(line)
		line = trim_string(line)
		if line == '' then
			if pcall( function() ds = Depstruct:create_from_strings(tokens, voca_dic, pos_dic, deprel_dic) end ) then
				treebank[#treebank+1] = ds
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

print('load dics')
local voca_dic = Dict:new(collobert_template)
voca_dic:load('../data/wsj-dep/dic/collobert/words.lst')
 
local pos_dic = Dict:new()
pos_dic:load("../data/wsj-dep/dic/pos.lst")

local deprel_dic = Dict:new()
deprel_dic:load('../data/wsj-dep/dic/deprel.lst')

print('load treebank')
local treebank = load_treebank('../data/wsj-dep/data/train.conll', voca_dic, pos_dic, deprel_dic)

print('training...')
local parser = Depparser:new(voca_dic, pos_dic, deprel_dic)
parser:train_liblinear_classifier(treebank)



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
