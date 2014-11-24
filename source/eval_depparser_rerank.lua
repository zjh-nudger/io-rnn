require 'depparser_rerank'
require 'utils'
require 'dict'
require 'xlua'
require 'dpiornn_gen'
require 'dp_spec'

torch.setnumthreads(1)
marked_file = '/tmp/eval_is_open'

posix = require('posix')

if #arg >= 3 then
	treebank_path = arg[2]
	kbesttreebank_path = arg[3]
	output = arg[4]

	if marked_file ~= nil then
		while check_file_exist(marked_file) == true do
			print('is busy, wait...')
			posix.sleep(10)
		end
	end

	local f = io.open(marked_file, 'w')

	print('load net')
	local net = IORNN:load(arg[1])
	print(net.complete_inside)
--	print(net.Wh:size())

	print('create parser')
	local parser = Depparser:new(net.voca_dic, net.pos_dic, net.deprel_dic)

	local u = arg[1]:find('/model')
	if u == nil then parser.mail_subject = path
	else parser.mail_subject = arg[1]:sub(1,u-1) end

	--print(parser.mail_subject)
--
	print('eval')

	print('\n\n--- oracle-best ---')
--	parser:eval('best', kbesttreebank_path, treebank_path, output..'.oracle-best')

	print('\n\n--- oracle-worst ---')
--	parser:eval('worst', kbesttreebank_path, treebank_path, output..'.oracle-worst')

	print('\n\n--- first ---')
--	parser:eval('first', kbesttreebank_path, treebank_path, output..'.first')

	print('\n\n--- rescore ---')
	parser:eval(net, kbesttreebank_path, treebank_path, kbesttreebank_path..'.iornnscores')

	print('\n\n--- mix. reranking ---')
	parser:eval(kbesttreebank_path..'.iornnscores', kbesttreebank_path, treebank_path, output..'.reranked')

	f:close()
	os.remove(marked_file)
else
	print("[net] [gold/input] [kbest] [output]")
end
