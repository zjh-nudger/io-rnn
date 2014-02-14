require 'dict'
require 'utils'

if #arg == 2 then
	embs_path = arg[1] .. '/wordembs.txt'
	dic_path = arg[1] .. '/words.txt'
	output_path = arg[2]

	-- load dic --
	local dic = Dict:new(collobert_template)
	dic:load(dic_path, true)

	-- load org embs --
	f = torch.DiskFile(embs_path, 'r')
	local info = f:readInt(2)
	local nword = info[1]	
	local embdim = info[2]	
	local org_embs = torch.Tensor(f:readDouble(nword*embdim))
					:resize(nword, embdim):t()
	org_embs = org_embs - org_embs:mean()
	org_embs = org_embs * 1/(org_embs:std())
	f:close()

	local embs = torch.Tensor(embdim, nword+1)
	embs[{{1,embdim},{2,1+nword}}]:copy(org_embs)

	-- output --
	f = torch.DiskFile(output_path, 'w')
	f:writeObject(dic)
	f:writeObject(embs)
	f:close()

else
	print(
		"invalid arguments: <wordembs_dir> <output_path>" )
		
end

