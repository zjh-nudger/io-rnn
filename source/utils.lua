function split_string( str , pattern )
	local pattern = pattern or "[^\t ]+"
	local toks = {}
	for k,v in string.gmatch(str, pattern) do
		toks[#toks+1] = k
	end
	return toks
end

function get_all_filenames( dir )
	local f = io.popen("ls " .. dir)
	local files = f:read("*a")
	f:close()

	return split_string(files, "[^\n]+")
end

function get_word_id(word2id, word)
	if word == nil then 
		return word2id['NULL']
	else
		word = string.lower(word)
		if tonumber(word) ~= nil then 
			return word2id['0']
		elseif word2id[word] == nil then 
			return word2id['UNKNOWN']
		else 
			return word2id[word]
		end
	end
end

function safe_compute_softmax(A)
	maxA,_ = A:max(1)
	maxA = torch.repeatTensor(maxA, A:size(1), 1)
	local B = torch.repeatTensor((A - maxA):exp():sum(1):log(), A:size(1), 1)
	return (A - maxA - B):exp()
end
