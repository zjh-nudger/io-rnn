require 'depstruct'
require 'utils'
require 'dict'
require 'xlua'
require 'dp_spec'

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
			local status, err =  pcall( function() 
					ds = Depstruct:create_from_strings(tokens, 
						self.voca_dic, self.pos_dic, self.deprel_dic) 
				end ) 
			if status then
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
	return net, trainkbestdsbank
end

function UDepparser:train(net, traindsbank_path, trainkbestdsbank_path, golddevdsbank_path, kbestdevdsbank_path, model_dir)
	os.execute('mkdir ' .. model_dir)

	for it = 1, TRAIN_MAX_N_EPOCHS do
		-- training IORNN
		local submodel_dir = model_dir..'/'..it..'/'
		os.execute('mkdir ' .. submodel_dir)
		net = self:one_step_train(net, traindsbank_path, trainkbestdsbank_path, golddevdsbank_path, kbestdevdsbank_path, submodel_dir)

		-- training MSTParser
		print('load train dsbank ' .. traindsbank_path .. ' ' .. trainkbestdsbank_path)
		local temp = self:load_dsbank(traindsbank_path)
		local trainkbestdsbank = self:load_kbestdsbank(trainkbestdsbank_path, temp, 10)

		local mst_dir = model_dir .. '/MST-' .. (it+1) ..'/' 
		os.execute('mkdir '..mst_dir)
		traindsbank_path = mst_dir .. 'train.conll'
		trainkbestdsbank_path = mst_dir .. 'train-'..TRAIN_MST_K_BEST..'-best-mst2ndorder.conll'

		local parses,_ = self:rerank(net, trainkbestdsbank)
		self:print_parses(parses, traindsbank_path)

		os.execute('java -classpath "../tools/mstparser/:../tools/mstparser/lib/trove.jar" -Xmx32g -Djava.io.tmpdir=./ mstparser.DependencyParser train train-file:'..traindsbank_path..' training-k:5 order:2 loss-type:nopunc model-name:'..mst_dir..'model test test-file:'..traindsbank_path..' testing-k:'..TRAIN_MST_K_BEST..' output-file:'..trainkbestdsbank_path)
		os.execute('cp '..trainkbestdsbank_path..' /tmp/kbest.txt')
		os.execute("cat /tmp/kbest.txt | sed 's/<no-type>/NOLABEL/g' > "..trainkbestdsbank_path)

		os.execute('java -classpath "../tools/mstparser/:../tools/mstparser/lib/trove.jar" -Xmx32g -Djava.io.tmpdir=./ mstparser.DependencyParser test order:2 model-name:'..mst_dir..'model test-file:'..golddevdsbank_path..' testing-k:'..TRAIN_MST_K_BEST..' output-file:'..kbestdevdsbank_path)
		os.execute('cp '..kbestdevdsbank_path..' /tmp/kbest.txt')
		os.execute("cat /tmp/kbest.txt | sed 's/<no-type>/NOLABEL/g' > "..kbestdevdsbank_path)

	end
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
			if best_score == nil or log_probs[j] > best_score then
				best_parse = parse
				best_score = log_probs[j]
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

punc = false

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
