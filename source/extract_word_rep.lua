require 'iornn'
require 'dict'
require 'knn'

k = 10

if #arg == 4 then
	local net_path = arg[1]
	local we_path = arg[2]
	local tgwords_path = arg[3]
	local output_path = arg[4]
 
	-- load net
	local net = IORNN:load(net_path)
	local embs = net.L

	-- load dic
	local f = torch.DiskFile(we_path, 'r')
	dic = f:readObject()
	setmetatable(dic, Dict_mt)
	embs = f:readObject()
	f:close()

--[[
	print(dic:size())
	print(embs:size())

	local f = io.open('temp', 'w')
	for i = 1,dic:size() do
		f:write(dic.id2word[i] .. '\n')
	end
	f:close()
]]

	-- load target words
	local tgdic = Dict:new(collobert_template)
	tgdic:load(tgwords_path)
	
	-- for knn
	local entries = {labels = dic.id2word, reps = embs}

	-- extract 
	f = torch.DiskFile(output_path, 'w')
	f:noAutoSpacing()

	for i = 1,tgdic:size() do
		local word = tgdic.id2word[i]
		local id = dic.word2id[word]
		if id ~= nil then
			f:writeString(word .. ' ')
			emb = embs[{{},{id}}] 
			for j = 1,emb:numel() do
				f:writeDouble(emb[{j,1}])
				f:writeString(' ')
			end
			f:writeString('\n')

			-- find knn
			local nn = knn(entries, emb, k)
			local str = word .. ' :\t'
			for _,id in ipairs(nn) do
				str = str .. entries.labels[id] .. ' '
			end
			print(str)
		end
	end
	f:close()

else
	print("[iornn] [wembs] [target words] [output]")
end


