require 'depparser'
require 'utils'
require 'dict'
require 'xlua'
require 'dpiornn'

torch.setnumthreads(1)

if #arg >= 2 then
	treebank_path = arg[#arg]

	print('load net')
	local nets = {}
	for i = 1,#arg-1 do
		nets[i] = IORNN:load(arg[i])
	end

	print('create parser')
	local parser = Depparser:new(nets[1].voca_dic, nets[1].pos_dic, nets[1].deprel_dic)
	parser:load_possible_word_dr('../data/wsj-dep/universal/dic/word_deprel.lst')

	local u = arg[1]:find('/model')
	if u == nil then parser.mail_subject = path
	else parser.mail_subject = arg[1]:sub(1,u-1) end

	print(parser.mail_subject)

	print('eval')
	parser:eval(nets, treebank_path, '/tmp/parsed.conll')

else
	print("[net path] [...] [treebank]")
end
