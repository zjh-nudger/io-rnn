require 'dict'
require 'utils'

Lex = {}
Lex_mt = {__index = Lex}

function Lex:new(wfunc)
	local lex = {	voca = Dict:new(wfunc), 
					class = Dict:new(), 
					word_in_class = {}, 
					class_of_word = {},
				}
	setmetatable(lex, Lex_mt)
	return lex
end

function Lex:setmetatable(lex)
	setmetatable(lex, Lex_mt)
	setmetatable(lex.voca, Dict_mt)
	setmetatable(lex.class, Dict_mt)
	for i,wc in ipairs(lex.word_in_class) do
		setmetatable(wc, Dict_mt)
	end
	return lex
end

function Lex:load(voca_path, class_path, classword_path)
	self.voca:load(voca_path)
	self.class:load(class_path)

	for line in io.lines(classword_path) do
		local comps = split_string(line)
		local class_id = self.class:get_id(comps[1])
		local word_id = self.voca:get_id(comps[2])
		
		--print(class_id .. ' ' .. word_id)

		local local_wc = self.word_in_class[class_id]
		if local_wc == nil then 
			self.word_in_class[class_id] = Dict:new()
			local_wc = self.word_in_class[class_id]
		end

		local_wc:addword(word_id)
		self.class_of_word[word_id] = class_id
		--print(local_wc)
		--print(word_id)
	end
end

--[[ for testing
local lex = Lex:new(collobert_template)
lex:load(arg[1], arg[2], arg[3])
print(lex)
]]

