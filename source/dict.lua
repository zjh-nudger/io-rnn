
Dict = {}
Dict_mt = {__index=Dict}

--***************** construction **************
function Dict:new(wfunc)
	local d = {}
	d.word2id = {['UNKNOWN'] = 1}
	d.id2word = {'UNKNOWN'}
	d.func = wfunc

	setmetatable(d, Dict_mt)

	return d
end

--**************** load from file ************
function Dict:load(filename)
	local i = 1
	for line in io.lines(filename) do
		self:addword(line)
	end
end

--***************** size ****************
function Dict:size()
	return #self.id2word
end


function Dict:addword(word)
	if self.func ~= nil then 
		word = self.func(word)
	end

	local id = self.word2id[word]
	if id == nil then
		id = self:size() + 1
		self.word2id[word] = id
		self.id2word[id] = word
	end
	return id
end

function Dict:get_id(word)
	if func ~= nil then 
		word = self.func(word)
	end
	local ret = self.word2id[word]
	if ret == nil then 
		ret = self.word2id['UNKNOWN']
	end
	return ret
end

--***************** template function *************--
function huang_template(word)
	word = string.lower(word)
	local ret = word
	if tonumber(word) ~= nil then
		if string.len(word) == 4 then
			ret = 'CDCDCDCD'
		else
			ret = 'CD'
		end
	end

	return ret
end

function collobert_template(word)
	local ret = string.lower(word)

	if tonumber(ret) ~= nil then 
		ret = '0'
	end
	return ret
end


