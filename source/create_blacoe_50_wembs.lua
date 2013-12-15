require 'dict'
require 'utils'

if #arg == 2 then
	embs_path = arg[1] .. '/embeddings.txt'
	dic_path = arg[1] .. '/words.lst'
	output_path = arg[2]

	-- load dic --
	local dic = Dict:new(turian_template)
	dic:load(dic_path, true)

	-- load org embs --
	f = torch.DiskFile(embs_path, 'r')
	local info = f:readInt(2)
	local nword = info[1]	
	local embdim = info[2]	
	local embs = torch.Tensor(f:readDouble(nword*embdim))
					:resize(nword, embdim):t()
	embs = embs * 1/(embs:std())
	f:close()

	-- output --
	f = torch.DiskFile(output_path, 'w')
	f:writeObject(dic)
	f:writeObject(embs)
	f:close()

	print(dic:size())
	print(embs:size())

else
	print(
		"invalid arguments: <wordembs_dir> <output_path>" )
		
end

