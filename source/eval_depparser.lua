require 'depparser'
require 'utils'
require 'dict'
require 'xlua'
require 'dpiornn'

torch.setnumthreads(1)

if #arg == 2 then
	net_path = arg[1]
	treebank_path = arg[2]

	print('load net')
	local net = IORNN:load(net_path)

	print('create parser')
	local parser = Depparser:new(net.L, net.voca_dic, net.pos_dic, net.deprel_dic)
	parser.net = net

	print('eval')
	parser:eval(treebank_path, '/tmp/parsed.conll')

else
	print("[net path] [treebank]")
end
