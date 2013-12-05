require 'knn'
require 'dict'
require 'tree'

function extract_all_phrases_and_reps(treebank, dic)
	local n = 0
	for _, tree in ipairs(treebank) do
		n = n + tree.n_nodes
	end

	local labels = {}
	local reps = torch.Tensor(treebank[1].inner:size(1), n)
	n = 0
	for _, tree in ipairs(treebank) do
		reps[{{},{n+1,n+tree.n_nodes}}]:copy(tree.inner)
		local phrases = extract_all_phrases(tree, dic)
		for i = 1,tree.n_nodes do
			labels[#labels+1] = phrases[i]
		end
		n = n + tree.n_nodes
	end

	return {labels = labels, reps = reps}
end

if #arg == 3 then 
	local we_path = arg[2]
	local parsed_treebank_path = arg[1]
	local output_path = arg[3]

	-- load dic
	print('load dic...')
	local f = torch.DiskFile(we_path, 'r')
	local dic = f:readObject()
	setmetatable(dic, Dict_mt)
	f:close()

	-- load treebank
	print('load parsed treebank...')
	f = torch.DiskFile(parsed_treebank_path, 'r')
	f:binary()
	local treebank = f:readObject()
	f:close()

	-- extract 
	local entries = extract_all_phrases_and_reps(treebank, dic)
--[[
	for i = 1,20000 do
		print(entries.labels[i])
	end
]]
	-- find nn
	local id = 1253
	local rep = entries.reps[{{},{id}}]
	print(entries.labels[id])
	local nn = knn(entries, rep, 50)
	print(nn)
	for _,i in ipairs(nn) do
		print(entries.labels[i])
	end

else 
	print("[parsed treebank] [we] [output]")
end
