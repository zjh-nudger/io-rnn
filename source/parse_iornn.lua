require 'iornn'
require 'tree' 

if #arg == 4 then
	torch.setnumthreads(1)

	local net_path = arg[1]
	local we_path = arg[2]
	local treebank_path = arg[3]
	local output_path = arg[4]

-- load dic
	print('load dic...')
	local f = torch.DiskFile(we_path, 'r')
	local dic = f:readObject(); setmetatable(dic, Dict_mt)
	f:close()

-- load treebank
	print('load treebank...')
	local treebank = {}
	for line in io.lines(treebank_path) do
		tree = Tree:create_from_string(line)
		tree = tree:to_torch_matrices(dic, 1)
		treebank[#treebank + 1] = tree
	end

-- load iornn
	print('load iornn...')
	local net = IORNN:load(net_path)

-- parse
	print('parsing...')
	treebank = net:parse(treebank)

-- save
	print('save parses')
	f = torch.DiskFile(output_path, 'w')
	f:binary()
	f:writeObject(treebank)
	f:close()

else
	print("[net] [dic] [treebank] [output]")

end
