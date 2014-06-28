require 'depstruct'
require 'dpiornn_gen'
require 'utils'
require 'dict'
require 'xlua'
require 'dp_spec'
require 'posix'

p = xlua.Profiler()

UDepparser = {}
UDepparser_mt = { __index = UDepparser }

function UDepparser:new(voca_dic, pos_dic, deprel_dic)
	local parser = { voca_dic = voca_dic, pos_dic = pos_dic, deprel_dic = deprel_dic}
	setmetatable(parser, UDepparser_mt)
	return parser
end

function UDepparser:load_dsbank(path)
	local dsbank = {}

	tokens = {}
	local i = 1
	for line in io.lines(path) do
		--print(line)
		line = trim_string(line)
		if line == '' then
			--local status, err =  pcall( function() 
					ds = Depstruct:create_from_strings(tokens, 
						self.voca_dic, self.pos_dic, self.deprel_dic) 
			--	end ) 
			if true then --status then
				ds.weight = 1
				dsbank[#dsbank+1] = ds
			else 
				print(err)
			end
			tokens = {}

			if math.mod(i,1000) == 0 then io.write('.');io.flush() end
			i = i + 1
		else 
			tokens[#tokens+1] = line
		end
	end

	return dsbank
end

function UDepparser:load_kbestdsbank(path, golddsbank, K, do_sampling)
	local dsbank = self:load_dsbank(path)
	local kbestdsbank = {}
	local K = K or 10000000

	if do_sampling then 
		print('do sampling')
	end

	-- load kbest-scores given by the MSTParser
	local id = 1
	for line in io.lines(path..'.mstscores') do
		local comps = split_string(line)
		local scores = torch.zeros(#comps)
		for i,c in ipairs(comps) do
			scores[i] = tonumber(c)
		end
		scores = torch.exp(scores - log_sum_of_exp(scores))
	
		local temp = {}
		for i = 1, scores:numel() do
			temp[i] = dsbank[id]; id = id + 1
			temp[i].mstscore = scores[i]
			temp[i].weight = 1
		end
		local group = {}
		if do_sampling then
			local newid = roulette_wheel_selection(scores, K)
			for i = 1, K do
				group[i] = temp[newid[i]]
			end 
		else
			for i = 1, math.min(K,#temp) do
				group[i] = temp[i]
			end
		end

		kbestdsbank[#kbestdsbank+1] = group
	end

	for i,group in ipairs(kbestdsbank) do
		local goldds = golddsbank[i]
		for _,ds in ipairs(group) do
			if ds.n_words ~= goldds.n_words then 
				print(#group)
				print(#kbestdsbank)
				print(ds.n_words)
				print(goldds.n_words)
				error("not match")
			end
			ds.word = goldds.word:clone()
			ds.cap = goldds.cap:clone()
			ds.sent = goldds.sent 
		end
	end
	
	return kbestdsbank
end

function UDepparser:compute_weights(net, kbestdsbank)
	for i,parses in ipairs(kbestdsbank) do
		local log_probs = net:compute_log_prob(parses)
		local scores = torch.exp(log_probs - log_sum_of_exp(log_probs)) -- apprx P_theta'(d|s)

		for j,ds in ipairs(parses) do
			ds.iornnscore = scores[j]
			ds.weight = ds.iornnscore / ds.mstscore
		end

		if math.mod(i,10) then io.write(i..' ');io.flush() end
	end

	return kbestdsbank
end

function UDepparser:one_step_train(net, traindsbank_path, trainkbestdsbank_path, golddevdsbank_path, kbestdevdsbank_path, model_dir)
	print('load train dsbank ' .. traindsbank_path .. ' ' .. trainkbestdsbank_path)
	local temp = self:load_dsbank(traindsbank_path)
	local trainkbestdsbank = self:load_kbestdsbank(trainkbestdsbank_path, temp, TRAIN_SAMPLE_SIZE, true)
	
	print('compute weights')
	self:compute_weights(net, trainkbestdsbank)
	
	-- shuf the traindsbank
	print('shufing train dsbank')
	local traindsbank = {}
	for _,dss in ipairs(trainkbestdsbank) do
		for _,ds in ipairs(dss) do
			traindsbank[#traindsbank+1] = ds
		end
	end
	local new_i = torch.randperm(#traindsbank)
	temp = {}
	for i = 1,#traindsbank do
		temp[i] = traindsbank[new_i[i]]
	end
	traindsbank = temp

	-- train
	local adagrad_config = {	weight_learningRate	= TRAIN_WEIGHT_LEARNING_RATE,
								voca_learningRate	= TRAIN_VOCA_LEARNING_RATE	}
	local adagrad_state = {}

	net:save(model_dir .. '/model_0')
	local prefix = model_dir..'/model'

	print('train net')
	adagrad_config, adagrad_state = net:train_with_adagrad(traindsbank, TRAIN_BATCHSIZE,
										TRAIN_1STEP_MAX_N_EPOCHS, {lambda = TRAIN_LAMBDA, lambda_L = TRAIN_LAMBDA_L}, 
										prefix, adagrad_config, adagrad_state, 
										golddevdsbank_path, kbestdevdsbank_path)
	return net
end

function UDepparser:warm_up_train(net, traindsbank_path, golddevdsbank_path, kbestdevdsbank_path, model_dir)
	print('load train dsbank ' .. traindsbank_path)
	local dsbank = self:load_dsbank(traindsbank_path)
	
	-- shuf the traindsbank
	print('shufing train dsbank')
	local new_i = torch.randperm(#dsbank)
	traindsbank = {}
	for i = 1,#dsbank do
		traindsbank[i] = dsbank[new_i[i]]
	end

	-- train
	local adagrad_config = {	weight_learningRate	= TRAIN_WEIGHT_LEARNING_RATE,
								voca_learningRate	= TRAIN_VOCA_LEARNING_RATE	}
	local adagrad_state = {}

	net:save(model_dir .. '/model_0')
	local prefix = model_dir..'/model'

	print('train net')
	adagrad_config, adagrad_state = net:train_with_adagrad(traindsbank, TRAIN_BATCHSIZE,
										TRAIN_WARM_UP_N_EPOCHS, {lambda = TRAIN_LAMBDA, lambda_L = TRAIN_LAMBDA_L}, 
										prefix, adagrad_config, adagrad_state, 
										golddevdsbank_path, kbestdevdsbank_path)
	return net
end

function execute(cmd)
	print("\nEXECUTE: " .. cmd)
	os.execute(cmd)
end


function UDepparser:train(net_struct, traindsbank_path, golddevdsbank_path, model_dir)

	local execMST = 'java -classpath "../tools/mstparser/:../tools/mstparser/lib/trove.jar" -Xmx32g -Djava.io.tmpdir=./ mstparser.DependencyParser ' 

	execute('mkdir ' .. model_dir)
	execute('mkdir ' .. model_dir..'/warm_up/')
	local net = nil
	local temp_file = model_dir .. '/temp'
	
	-- train MSTparser with dsbank
	local mst_dir = model_dir..'/MST-1/'
	execute('mkdir ' .. mst_dir)
	execute('cp ' .. traindsbank_path .. ' ' .. mst_dir ..'train.conll')
	traindsbank_path = mst_dir..'train.conll'
	trainkbestdsbank_path = mst_dir .. 'train-'..TRAIN_MST_K_BEST..'-best-mst2ndorder.conll'
	kbestdevdsbank_path = mst_dir .. 'dev-'..TRAIN_MST_K_BEST..'-best-mst2ndorder.conll'

	execute(execMST .. 
			'train train-file:'..traindsbank_path..' training-k:5 order:2 loss-type:nopunc model-name:'..mst_dir..'model '..
			'test test-file:'..traindsbank_path..' testing-k:'..TRAIN_MST_K_BEST..' output-file:'..trainkbestdsbank_path)
	execute('cp '..trainkbestdsbank_path..' '..temp_file)
	execute("cat "..temp_file.." | sed 's/<no-type>/NOLABEL/g' > "..trainkbestdsbank_path)

	execute(execMST .. 'test order:2 model-name:'..mst_dir..'model test-file:'..golddevdsbank_path..' testing-k:'..TRAIN_MST_K_BEST..' output-file:'..kbestdevdsbank_path)
	execute(execMST .. 'test order:2 model-name:'..mst_dir..'model test-file:'..golddevdsbank_path..' testing-k:1 output-file:/tmp/x eval gold-file:'..golddevdsbank_path)

	execute('cp '..kbestdevdsbank_path..' '..temp_file)
	execute("cat "..temp_file.." | sed 's/<no-type>/NOLABEL/g' > "..kbestdevdsbank_path)


	for it = 1, TRAIN_MAX_N_EPOCHS do
		local submodel_dir = model_dir..'/'..it..'/'
		execute('mkdir ' .. submodel_dir)
		
		-- warm-up
		local warm_up_dir = submodel_dir..'/warm_up/'
		execute('mkdir '..warm_up_dir)
		net = IORNN:new(net_struct)
		self:warm_up_train(net, traindsbank_path, golddevdsbank_path, kbestdevdsbank_path, warm_up_dir)

		-- training MSTParser
		print('load train dsbank ' .. traindsbank_path .. ' ' .. trainkbestdsbank_path)

		local mst_dir = model_dir .. '/MST-' .. (it+1) ..'/' 
		execute('mkdir '..mst_dir)
		old_traindsbank_path = traindsbank_path
		traindsbank_path = mst_dir .. 'train.conll'
		old_trainkbestdsbank_path = trainkbestdsbank_path
		trainkbestdsbank_path = mst_dir .. 'train-'..TRAIN_MST_K_BEST..'-best-mst2ndorder.conll'
		kbestdevdsbank_path = mst_dir .. 'dev-'..TRAIN_MST_K_BEST..'-best-mst2ndorder.conll'

		self:rerank_parallel(warm_up_dir..'/model_'..TRAIN_WARM_UP_N_EPOCHS, old_traindsbank_path, old_trainkbestdsbank_path, traindsbank_path, TRAIN_N_PROC or 10)

		execute(execMST .. 
				'train train-file:'..traindsbank_path..' training-k:5 order:2 loss-type:nopunc model-name:'..mst_dir..'model ' .. 
				'test test-file:'..traindsbank_path..' testing-k:'..TRAIN_MST_K_BEST..' output-file:'..trainkbestdsbank_path)
		execute('cp '..trainkbestdsbank_path..' '..temp_file)
		execute("cat "..temp_file.." | sed 's/<no-type>/NOLABEL/g' > "..trainkbestdsbank_path)

		execute(execMST .. 'test order:2 model-name:'..mst_dir..'model test-file:'..golddevdsbank_path..' testing-k:'..TRAIN_MST_K_BEST..' output-file:'..kbestdevdsbank_path)
		execute(execMST .. 'test order:2 model-name:'..mst_dir..'model test-file:'..golddevdsbank_path..' testing-k:1 output-file:/tmp/x eval gold-file:'..golddevdsbank_path)

		execute('cp '..kbestdevdsbank_path..' '..temp_file)
		execute("cat "..temp_file.." | sed 's/<no-type>/NOLABEL/g' > "..kbestdevdsbank_path)
	end
end

function UDepparser:train_only_net(net_struct, traindsbank_path, golddevdsbank_path, kbestdevdsbank_path, model_dir)
	execute('mkdir '..model_dir)
	local subdir = model_dir..'/only_net/'
	execute('mkdir '..subdir)
	net = IORNN:new(net_struct)
	TRAIN_WARM_UP_N_EPOCHS = 20
	self:warm_up_train(net, traindsbank_path, golddevdsbank_path, kbestdevdsbank_path, subdir)
end

function UDepparser:compute_scores(test_ds, gold_ds, punc)
	local unlabel = 0
	local label = 0
	local n = 0

	if test_ds.n_words ~= gold_ds.n_words then
		error('not match')
	else
		for j = 2,test_ds.n_words do -- ignore ROOT
			local count = true
			--local w = self.voca_dic.id2word[gold_ds.word[j]]:gsub("%p",'')
						
			if count then
				n = n + 1
				if test_ds.head[j] == gold_ds.head[j] then
					unlabel = unlabel + 1
					if test_ds.deprel[j] == gold_ds.deprel[j] then
						label = label + 1
					end
				end
			end
		end
	end
	return { unlabel = unlabel , label = label , n = n }
end

function UDepparser:rerank_oracle(kbestdsbank, golddsbank, typ, K)
	local typ = typ or 'best'
	local K = K or 10

	if #kbestdsbank ~= #golddsbank then 
		error('size not match')
	end

	local ret = {}

	for i,parses in ipairs(kbestdsbank) do
		if typ == 'first' then ret[i] = parses[1] 
		else
			local gold = golddsbank[i]
			local best_parse = nil
			local best_score = nil

			for k,parse in ipairs(parses) do
				if k > K then break end
				local score = self:compute_scores(parse, gold)
				if typ == 'worst' then score.unlabel = -score.unlabel end
				if best_score == nil or score.unlabel > best_score then
					best_parse = parse
					best_score = score.unlabel
				end
			end
			ret[i] = best_parse
		end
	end

	return ret
end

function UDepparser:rerank(net, kbestdsbank, output)
	local K = K or 10000
	local ret = {}
	local sum_sen_log_p = 0
	local sum_n_words = 0
	local alpha = TRAIN_WEIGHT_MIX
	--local alpha = self.weight_mix

	local f = nil
	if output then
		f = torch.DiskFile(output, 'w')
		f:writeInt(#kbestdsbank)
	end

	for i,org_parses in ipairs(kbestdsbank) do
		if math.mod(i, 1) == 0 then io.write(i..' ');io.flush() end
		local parses = {}
		for k = 1, math.min(K, #org_parses) do
			parses[k] = org_parses[k]
		end

		local best_parse = nil
		local best_score = nil
		local log_probs = net:compute_log_prob(parses)

		if f then f:writeObject(log_probs) end

		for j,parse in ipairs(parses) do
			local score = alpha*log_probs[j] + (1-alpha)*math.log(parse.mstscore)
			if best_score == nil or score > best_score then
				best_parse = parse
				best_score = score
			end
		end
		ret[i] = best_parse
		sum_sen_log_p = sum_sen_log_p + log_sum_of_exp(torch.Tensor(log_probs))
		sum_n_words = sum_n_words + parses[1].n_words - 1
	end
	local ppl = math.pow(2, -sum_sen_log_p / math.log(2) / sum_n_words)

	if f then f:close() end
	
	return ret, ppl
end

function UDepparser:rerank_parallel(net_path, dsbank_path, kbestdsbank_path, output, n_procs)
	local dir = 'tmp'..math.floor(math.random()*1000)..'/'
	execute('mkdir ' .. dir)

	local dsbank = self:load_dsbank(dsbank_path)
	local kbestbank = self:load_kbestdsbank(kbestdsbank_path, dsbank)
	local subn = math.ceil(#dsbank / n_procs)

	execute('split -l '..subn..' -d -a 1 '..kbestdsbank_path..'.mstscores '..dir..'/mstscores')

	for i = 1,n_procs do
		local subkbestbank = {}
		local subbank = {}
		for j = (i-1)*subn+1,math.min(i*subn,#dsbank) do
			local id = j - (i-1)*subn
			subbank[id] = dsbank[j]
			for _,ds in ipairs(kbestbank[j]) do
				subkbestbank[#subkbestbank+1] = ds
			end
		end

		-- write subbank
		local subbank_path = dir..'/'..i..'-sub'
		local subkbestbank_path = dir..'/'..i..'-sub-best'
		local result_path = dir..'/'..i..'-results'
		self:print_parses(subbank, subbank_path)
		self:print_parses(subkbestbank, subkbestbank_path)
		execute('mv '..dir..'/mstscores'..(i-1)..' '..subkbestbank_path..'.mstscores')

		-- parse
		execute("th eval_unsup_dp.lua "..net_path..' '..subbank_path..' '.. subkbestbank_path..' 1000000 '..result_path..' &')
	end

	while true do
		local done = true
		for i = 1,n_procs do
			local f = io.open(dir..'/'..i..'-results', 'r')
			if f == nil then 
				done = false 
				break
			else 
				f:close() 
			end
		end
		if done then break 
		else posix.sleep(1) end 
	end

	-- combine
	local ret = {}
	for i = 1,n_procs do
		local ret_path = dir..'/'..i..'-results'
		local subbank = self:load_dsbank(ret_path)
		for _,ds in ipairs(subbank) do
			ret[#ret+1] = ds
		end
	end
	self:print_parses(ret, output)
end

function UDepparser:perplexity(net, dsbank)
	local log_probs = net:compute_log_prob(dsbank)
	local sum = 0
	local nwords = 0
	for i,log_p in ipairs(log_probs) do
		sum = sum + log_p / math.log(2)
		nwords = nwords + dsbank[i].n_words - 1
	end
	return math.pow(2, -sum / nwords)
end

function UDepparser:print_parses(parses, output)
	local f = io.open(output, 'w')
	for i, parse in ipairs(parses) do
		local sent = parse.sent
		for j = 2,#sent.word do
			f:write((j-1)..'\t'..sent.word[j]..'\t_\t'..sent.pos[j]..'\t'..sent.pos[j]..'\t_\t'..(parse.head[j]-1)..'\t'..self.deprel_dic.id2word[parse.deprel[j] ]..'\t_\t_\n')
		end
		f:write('\n')	
	end
	f:close()
end

function UDepparser:computeAScores(parses, golddsbank, punc, output)
	if output then
		self:print_parses(parses, output)
	end

	-- compute scores
	local total = 0
	local label = 0
	local unlabel = 0

	for i,parse in ipairs(parses) do
		local gold = golddsbank[i]
		local ret = self:compute_scores(parse, gold, punc)
		total 	= total + ret.n
		label 	= label + ret.label
		unlabel = unlabel + ret.unlabel
	end
	--print(total)

	local LAS = label / total * 100
	local UAS = unlabel / total * 100
	return LAS, UAS
end

punc = true

-- should not call it directly when training, there's a mem-leak problem!!!
function UDepparser:eval(typ, kbestpath, goldpath, output, K)
	local str = ''

	print('load ' .. goldpath)
	local golddsbank = self:load_dsbank(goldpath)

	print('load ' .. kbestpath)
	local kbestdsbank, kbestdsscore  = self:load_kbestdsbank(kbestpath, golddsbank, K)

	print('parsing...')

	-- compute perplexity
	if type(typ) ~= 'string' then
		--str = str .. 'ds-ppl ' .. self:perplexity(typ, golddsbank) .. '\n'
	end

	-- reranking
	local parses = nil
	local ppl = nil
	if type(typ) == 'string' then
		if typ == 'best' or typ == 'worst' or typ == 'first' then 
			parses = self:rerank_oracle(kbestdsbank, golddsbank, typ, K)
			LAS, UAS = self:computeAScores(parses, golddsbank, punc) --'/tmp/univ-test-result.conll.'..typ)
			str = 'LAS = ' .. string.format("%.2f",LAS)..'\nUAS = ' ..string.format("%.2f",UAS)
			print(str)
		end
	else 
		local net = typ
		parses,ppl = self:rerank(net, kbestdsbank, kbestpath..'.iornnscores')
		LAS, UAS = self:computeAScores(parses, golddsbank, punc)
		str = 'LAS = ' .. string.format("%.2f",LAS)..'\nUAS = ' ..string.format("%.2f",UAS)
		print(str)

		if output then
			self:print_parses(parses, output)
			local f = io.open(output..'.done', 'w') -- for notification only
			f.close()
		end

		-- mail
		if EVAL_EMAIL_ADDR and self.mail_subject then
			os.execute('echo "'..str..'" | mail -s '..self.mail_subject..' '..EVAL_EMAIL_ADDR)
		end
		print('sen-ppl ' .. ppl ..'\n')
	end	
end

--[[ for testing
torch.setnumthreads(1)

print('load dictionaries')
local data_path = '../data/wsj-dep/universal/'

local voca_dic = Dict:new()
voca_dic:load(data_path .. '/dic/collobert/words.lst')
local pos_dic = Dict:new()
pos_dic:load(data_path .. '/dic/cpos.lst')
local deprel_dic = Dict:new()
deprel_dic:load(data_path .. '/dic/deprel.lst')

local parser = UDepparser:new(voca_dic, pos_dic, deprel_dic)

print('eval...')
parser:eval(nil, data_path..'/data/dev-small-10best-mst.conll', data_path..'/data/dev-small.conll')
]]
