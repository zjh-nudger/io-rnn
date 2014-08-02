require 'unsup_depparser'
require 'utils'
require 'dict'
require 'xlua'
require 'dpiornn_gen'

torch.setnumthreads(1)

if #arg == 6 then
	net_path 			= arg[1]
	kbestparser 		= arg[2]
	kbestparser_model 	= arg[3]
	input				= arg[4]
	output 				= arg[5]
	K					= arg[6]

	print('load net')
	local net = IORNN:load(net_path)

	print('create parser')
	local parser = UDepparser:new(net.voca_dic, net.pos_dic, net.deprel_dic)

	parser:parse(net_path, kbestparser, kbestparser_model, input, output, K)

else
	print("[net] [kbestparser] [kbestparser model] [input] [output] [K] ")
end
