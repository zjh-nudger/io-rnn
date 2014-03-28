require 'utils'

Depstruct = {}
Depstruct_mt = { __index=Depstruct }

N_DEPS = 40

function Depstruct:new( input )
	local len = #input
	local ds = {
		n_words 	= len,
		word_id		= torch.zeros(len),
		pos_id		= torch.zeros(len),
		head_id		= torch.zeros(len),
		deprel_id	= torch.zeros(len),
		n_deps		= torch.zeros(len),
		dep_id		= torch.zeros(N_DEPS, len),

		root_dep_id	= torch.zeros(N_DEPS),
		root_n_deps = 0
	}

	setmetatable(ds, Depstruct_mt)

	-- set data
	for i,row in ipairs(input) do
		ds.word_id[i]	 = row[1]
		ds.pos_id[i]	 = 1--row[2]
		ds.head_id[i]	 = row[3]
		ds.deprel_id[i]  = 1--row[4]
		
		local hid = row[3]
		if hid == 0 then -- root
			ds.root_n_deps = ds.root_n_deps + 1
			ds.root_dep_id[ds.root_n_deps] = i
		else 
			ds.n_deps[hid] = ds.n_deps[hid] + 1
			ds.dep_id[{ds.n_deps[hid],hid}] = i
		end
	end

	return ds
end

function Depstruct:create_from_strings(input, voca_dic, pos_dic, deprel_dic)
	for i,row in ipairs(input) do
		local comps = split_string(row)
		row = { voca_dic:get_id(comps[2]),
				pos_dic:get_id(comps[5]),
				tonumber(comps[7]),
				deprel_dic:get_id(comps[8])
			}
		input[i] = row	
	end

	return Depstruct:new(input)
end

--[[ test
require 'dict'

torch.setnumthreads(1)

local voca_dic = Dict:new(collobert_template)
voca_dic:load('../data/wsj-dep/dic/collobert/words.lst')
 
local pos_dic = Dict:new(cfg_template)
pos_dic:load("../data/wsj-dep/dic/pos.lst")

local deprel_dic = Dict:new()
deprel_dic:load('../data/wsj-dep/dic/deprel.lst')

tokens = {}
for line in io.lines('../data/wsj-dep/data/test.conll') do
	line = trim_string(line)
	if line == '' then
		print(tokens)
		local ds = Depstruct:create_from_strings(tokens, voca_dic, pos_dic, deprel_dic)
		tokens = {}
	else 
		tokens[#tokens+1] = line
	end
end
]]
