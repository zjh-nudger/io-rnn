require 'depparser_rerank'
require 'utils'
require 'dict'
require 'xlua'
require 'dpiornn_gen'

torch.setnumthreads(1)

if #arg >= 3 then
	treebank_path = arg[2]
	kbesttreebank_path = arg[3]
	K = tonumber(arg[4]) or 10

	print('load net')
	local net = IORNN:load(arg[1])

	print('create parser')
	local parser = Depparser:new(net.voca_dic, net.pos_dic, net.deprel_dic)

	local u = arg[1]:find('/model')
	if u == nil then parser.mail_subject = path
	else parser.mail_subject = arg[1]:sub(1,u-1) end

	print(parser.mail_subject)

	print('eval')
--[[
	print('oracle-best')
	parser:eval('best', kbesttreebank_path, treebank_path, nil, K)

	print('oracle-worst')
	parser:eval('worst', kbesttreebank_path, treebank_path, nil, K)

	print('first')
	parser:eval('first', kbesttreebank_path, treebank_path, nil, K)
]]
	print('IORNN')
	parser:eval(net, kbesttreebank_path, treebank_path, nil, K)

else
	print("[net path] [gold]  [kbest] [K] ")
end
