require 'sfiornnlm'
require 'tree'
require 'utils'
require 'dict'
require 'optim'

if #arg == 4 then
	torch.setnumthreads(1)

	word_lst_path = arg[1]
	senbank_dir = arg[2]
	learn_rate = tonumber(arg[3])
	local model_dir = arg[4]

-- load word list
	print('load words...')
	local vocaDic = Dict:new()
	vocaDic:load(word_lst_path)	
	vocaDic:addword('<s>')
	vocaDic:addword('</s>')

-- create net
	print('create iornn...')
	local struct = {Lookup = uniform(200, vocaDic:size(), -1, 1),
					func = tanh, funcPrime = tanhPrime, n_leaves = 6 }
	local net = SFIORNNLM:new(struct, rules)
	--local net = SFIORNNLM:load('model_completeccg_bnc_shuf_1/model_6_1')

	net.update_L = true

	lambda = 1e-6
	lambda_L = 1e-6
	batchsize = 10
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
				local sen = {}
				for i = 1,net.n_leaves - 1 do
					sen[i] = vocaDic:get_id('<s>')
				end
				for i,w in ipairs(words) do
					sen[#sen+1] = vocaDic:get_id(w)
				end
				sen[#sen+1] = vocaDic:get_id('</s>')
				senbank[#senbank+1] = sen
				--print(sen)
			end

			adagrad_config, adagrad_state = 
				net:train_with_adagrad(senbank, batchsize, 1, {lambda = lambda, lambda_L = lambda_L}, 
										prefix,	adagrad_config, adagrad_state, bag_of_subtrees)
		end
	end

else
	print("[words path] [senbank dir] [learning rate] [model dir]")
end
