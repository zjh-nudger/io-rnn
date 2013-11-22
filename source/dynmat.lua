DynMat = {}
DynMat_mt = {__index = DynMat}

function DynMat:new( ncols, blocksize )
	local blocksize = blocksize or 10000
	local mat = { 
					blocks = {torch.Tensor(blocksize, ncols)}, 
					blocksize = blocksize, nrows = 0 , 
					ncols = ncols 
				}

	setmetatable(mat, DynMat_mt)
	return mat
end

function DynMat:num_rows()
	return self.nrows
end

function DynMat:num_cols() 
	return self.ncols
end

function DynMat:insert_row( datum, irow )
	local irow = irow or self.nrows + 1 
	local iblock = math.floor((irow-1) / self.blocksize) + 1 
	local ofset = irow - (iblock-1) * self.blocksize 

	local block = self.blocks[iblock]
	if block == nil then 
		block = torch.Tensor(self.blocksize, self.ncols)
		self.blocks[iblock] = block
	end
	
	block[{ofset,{}}]:copy(datum)
	if self.nrows < irow then 
		self.nrows = irow
	end
end

function DynMat:get_row(irow)
	local iblock = math.floor((irow-1) / self.blocksize) + 1
	local ofset = irow - (iblock-1) * self.blocksize

	local block = self.blocks[iblock]
	if block == nil then return nil 
	else return block[{ofset,{}}] end
end
