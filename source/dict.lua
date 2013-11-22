
Dict = {}
Dict_mt = {__index=Dict}

--***************** construction **************
function Dict:new()
	local d = {}
	d.word2id = {}
	d.id2word = {}

	setmetatable(d, Dict_mt)
	return d
end

--**************** load from file ************
function Dict:load( filename , with_id )
	local with_id = with_id or false

	local i = 1
	for line in io.lines(filename) do
		self.word2id[line] = i
		self.id2word[i] = line
		i = i + 1
	end
end

--***************** size ****************
function Dict:size()
	return #self.id2word
end


function Dict:addword( word )
	local id = self.word2id[word]
	if id == nil then
		id = self:size() + 1
		self.word2id[word] = id
		self.id2word[id] = word
	end
	return id
end

