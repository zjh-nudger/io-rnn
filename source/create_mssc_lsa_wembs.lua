require 'dict'
require 'utils'

if #arg == 3 then
	embs_path = arg[1] .. '/embs.txt'
	dic_path = arg[1] .. '/words.lst'
	output_path = arg[2]

	-- load dic --
	local dic = Dict:new(collobert_template)
	dic:load(dic_path, true)

	-- load org embs --
	f = torch.DiskFile(embs_path, 'r')
	local info = f:readInt(2)
	local nword = info[1]	
	local embdim = info[2]	
	local embs = torch.Tensor(f:readDouble(nword*embdim))
					:resize(nword, embdim):t()
	f:close()

	-- L2-normalize to 
	if arg[3] == 'normalize' then
		embs = torch.cdiv(embs, torch.repeatTensor(torch.norm(embs,2,1), embs:size(1),1))
	end

	-- output --
	f = torch.DiskFile(output_path, 'w')
	f:writeObject(dic)
	f:writeObject(embs)
	f:close()

else
	print("[wordembs_dir] [output_path] [normalize/org]" )
		
end

