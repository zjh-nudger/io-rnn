require 'iornnlm'
require 'tree'
require 'utils'
require 'dict'
require 'optim'

if #arg == 4 then
	torch.setnumthreads(1)

	we_path = arg[1]
	senbank_dir = arg[2]
	learn_rate = tonumber(arg[3])
	local model_dir = arg[4]

-- load word emb and grammar rules
	print('load wordembeddngs...')
	local f = torch.DiskFile(we_path, 'r')
	local vocaDic = f:readObject(); setmetatable(vocaDic, Dict_mt)
	local wembs = f:readObject()
	f:close()

-- create net
	print('create iornn...')
	local struct = {Lookup = wembs, func = tanh, funcPrime = tanhPrime }
	local net = IORNNLM:new(struct, rules)
	--local net = IORNNLM:load('model_completeccg_bnc_shuf_1/model_6_1')

	net.update_L = true
	net.L = uniform(net.L:size(1), net.L:size(2), -1e-5, 1e-5)

	lambda = 1e-4
	lambda_L = 1e-8
	batchsize = 100
	maxnepoch = 100

-- train
	local filenames = get_all_filenames(senbank_dir)
	local adagrad_config = {learningRate = learn_rate}
	local adagrad_state = {}

	net:save(model_dir .. '/model_0')

	for nepoch = 1,maxnepoch do
		for i,fn in ipairs(filenames) do
			local prefix = model_dir..'/model_'..tostring(nepoch)
			local senbank = {}
			print(prefix .. '_' .. i)
				
			print('load sentences in file ' .. fn)
			for line in io.lines(senbank_dir .. '/' .. fn) do
				local words = split_string(line)
				local sen = {vocaDic:get_id('PADDING')}
				for i,w in ipairs(words) do
					sen[i+1] = vocaDic:get_id(w)
				end
				sen[#sen+1] = vocaDic:get_id('PADDING')
				senbank[#senbank+1] = sen
			end

			adagrad_config, adagrad_state = 
				net:train_with_adagrad(senbank, batchsize, 1, {lambda = lambda, lambda_L = lambda_L}, 
										prefix,	adagrad_config, adagrad_state, bag_of_subtrees)
		end
	end

else
	print("[wordemb path] [senbank dir] [learning rate] [model dir]")
end
