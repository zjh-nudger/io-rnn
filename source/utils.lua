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

function get_word_id(dic, word)
	if word == nil then 
		return dic.word2id['NULL']
	else
		word = string.lower(word)
		if tonumber(word) ~= nil then 
			return dic.word2id['0']
		elseif dic.word2id[word] == nil then 
			return dic.word2id['UNKNOWN']
		else 
			return dic.word2id[word]
		end
	end
end


