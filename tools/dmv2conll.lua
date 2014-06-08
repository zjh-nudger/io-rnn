
function split_string( str , pattern )
	local pattern = pattern or "[^\t ]+"
	local toks = {}
	for k,v in string.gmatch(str, pattern) do
		toks[#toks+1] = k
	end
	return toks
end

if #arg ~= 3 then
	error('[ph-file] [pwh-file] [output]')
end

ph_file		= arg[1]
pwh_file	= arg[2]
output		= arg[3]

pos		= {}
head	= {}
word	= {}

i = 0
for line in io.lines(ph_file) do
	i = i + 1
	t = math.mod(i,3)
	if		t == 1 then pos[#pos+1] = split_string(line)
	elseif	t == 2 then head[#head+1] = split_string(line)
	end 
end

i = 0
for line in io.lines(pwh_file) do
	i = i + 1
	t = math.mod(i,4)
	if t == 1 then word[#word+1] = split_string(line) end 
end

f = io.open(output,'w')
for i,p in ipairs(pos) do
	w = word[i]
	h = head[i]
	--print(p)
	--print(w)
	--print(h)
	if #p ~= #w or #p ~= #h then 
		error('not match')
	end

	for j = 1,#p do
		local deprel = 'NOLABEL'
		if h[j] == '0' then deprel = 'ROOT' end 
		f:write(j..'\t'..w[j]..'\t_\t'..p[j]..'\t'..p[j]..'\t_\t'..h[j]..'\t'..deprel..'\n')
	end
	f:write('\n')
end
f:close()


