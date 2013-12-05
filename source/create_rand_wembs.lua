require 'tree'
require 'dict'
require 'utils'

if #arg == 3 then
	
	local wordlistpath = arg[1]
	local outputpath = arg[3]
	local dim = tonumber(arg[2])
	
	local dic = Dict:new(collobert_template)	
	dic:load(wordlistpath)

	local embs = (torch.rand(dim, dic:size()) - 0.5) * 2 * 0.0001
	
	local f = torch.DiskFile(outputpath, 'w')
	f:writeObject(dic)
	f:writeObject(embs)
	f:close()	
else 
	print("invalide arguments: [word list path] [dim] [output path]")
end
