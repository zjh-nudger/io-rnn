require 'depstruct'
require 'utils'
require 'dict'
require 'xlua'
require 'dp_spec'

p = xlua.Profiler()

Depparser = {}
Depparser_mt = { __index = Depparser }

function Depparser:new(voca_dic, pos_dic, deprel_dic)
	local parser = { voca_dic = voca_dic, pos_dic = pos_dic, deprel_dic = deprel_dic}
	setmetatable(parser, Depparser_mt)
	return parser
end

function Depparser:load_dsbank(path)
	local dsbank = {}
	local raw = {}

	tokens = {}
	for line in io.lines(path) do
		--print(line)
		line = trim_string(line)
		if line == '' then
			if pcall( function() 
					ds,sent = Depstruct:create_from_strings(tokens, 
						self.voca_dic, self.pos_dic, self.deprel_dic) 
				end ) 
			then
				dsbank[#dsbank+1] = ds
				raw[#raw+1] = sent
			else 
				print('error')
				print(tokens)
			end
			tokens = {}
		else 
			tokens[#tokens+1] = line
		end
	end

	return dsbank, raw
end

function Depparser:load_kbestdsbank(path)
	local dsbank = self:load_dsbank(path)
	local kbestdsbank = {}
	local group = nil
	
	for _,ds in ipairs(dsbank) do
		if group == nil or group[1].n_words ~= ds.n_words or (group[1].word - ds.word):abs():sum() > 0 then
			group = { ds }
			kbestdsbank[#kbestdsbank+1] = group
		else
			group[#group+1] = ds		
		end
	end
	
	return kbestdsbank
end

function Depparser:train(net, traintrebank_path, devdsbank_path, kbestdevdsbank_path, model_dir)
	print('load train dsbank')
	local traindsbank,_ = self:load_dsbank(traintrebank_path)
	
	net.update_L = TRAIN_UPDATE_L

	-- shuf the traindsbank
	print('shufing train dsbank')
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
										TRAIN_MAX_N_EPOCHS, {lambda = TRAIN_LAMBDA, lambda_L = TRAIN_LAMBDA_L}, 
										prefix, adagrad_config, adagrad_state, 
										devdsbank_path, kbestdevdsbank_path)
	return net
end

function Depparser:compute_scores(test_ds, gold_ds)
	local unlabel = 0
	local label = 0

	if test_ds.n_words ~= gold_ds.n_words then
		error('not match')
	else
		for j = 2,test_ds.n_words do -- ignore ROOT
			if test_ds.head[j] == gold_ds.head[j] then
				unlabel = unlabel + 1
				if test_ds.deprel[j] == gold_ds.deprel[j] then
					label = label + 1
				end
			end
		end
	end
	return { unlabel = unlabel , label = label }
end

function Depparser:rerank_oracle(kbestdsbank, golddsbank, typ, K)
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

function Depparser:rerank(net, kbestdsbank, K)
	local K = K or 10
	local ret = {}

	for i,org_parses in ipairs(kbestdsbank) do
		if math.mod(i, 100) == 0 then print(i) end
		local parses = {}
		for k = 1, math.min(K, #org_parses) do
			parses[k] = org_parses[k]
		end

		local best_parse = nil
		local best_score = nil
		local log_probs = net:compute_log_prob(parses)

		for j,parse in ipairs(parses) do
			if best_score == nil or log_probs[j] > best_score then
				best_parse = parse
				best_score = log_probs[j]
			end
		end
		ret[i] = best_parse
	end

	return ret
end

function Depparser:perplexity(net, dsbank)
	local log_probs = net:compute_log_prob(dsbank)
	local sum = 0
	local nwords = 0
	for i,log_p in ipairs(log_probs) do
		sum = sum + log_p / math.log(2)
		nwords = nwords + dsbank[i].n_words - 1
	end
	return math.pow(2, -sum / nwords)
end

-- should not call it directly when training, there's a mem-leak problem!!!
function Depparser:eval(typ, kbestpath, goldpath, output, K)
	local str = ''

	print('load data')
	local golddsbank, raw = self:load_dsbank(goldpath)
	local kbestdsbank, _  = self:load_kbestdsbank(kbestpath)

	print('parsing...')

	-- compute perplexity
	if type(typ) ~= 'string' then
		str = str .. 'perplexity ' .. self:perplexity(typ, golddsbank) .. '\n'
	end

	-- reranking
	local parses = nil
	if type(typ) == 'string' then 
		parses = self:rerank_oracle(kbestdsbank, golddsbank, typ, K)
	else 
		local net = typ
		parses = self:rerank(net, kbestdsbank, K)
	end
	
	if output then
		print('print to '..output)
		local f = io.open(output, 'w')
		for i, parse in ipairs(parses) do
			local sent = raw[i]
			for j = 2,#sent do
				f:write((j-1)..'\t'..sent[j]..'\t_\t_\t_\t_\t'..(parse.head[j]-1)..'\t'..self.deprel_dic.id2word[parse.deprel[j] ]..'\t_\t_\n')
			end
			f:write('\n')	
		end
		f:close()
	end

	-- compute scores
	local total = 0
	local label = 0
	local unlabel = 0

	for i,parse in ipairs(parses) do
		local gold = golddsbank[i]
		local ret = self:compute_scores(parse, gold)
		total 	= total + parse.n_words - 1
		label 	= label + ret.label
		unlabel = unlabel + ret.unlabel
	end

	local LAS = label / total
	local UAS = unlabel / total
	str = str .. 'LAS = ' .. string.format("%.2f",LAS*100)..'\nUAS = ' ..string.format("%.2f",UAS*100)

	print(str)

	-- mail
	if EVAL_EMAIL_ADDR and self.mail_subject then
		os.execute('echo "'..str..'" | mail -s '..self.mail_subject..' '..EVAL_EMAIL_ADDR)
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

local parser = Depparser:new(voca_dic, pos_dic, deprel_dic)

print('eval...')
parser:eval(nil, data_path..'/data/dev-small-10best-mst.conll', data_path..'/data/dev-small.conll')
]]
