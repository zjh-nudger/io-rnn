#include <stdio.h>
#include <string.h>
#include "lua.h"
#include "luaT.h"
#include "TH/TH.h"
#include "lauxlib.h"
#include "lualib.h"

#include "gsl/gsl_statistics.h"

static double _spearman_rho(THDoubleTensor* a, THDoubleTensor* b) {
	int n = THDoubleTensor_size(a, 0);
	double* a_data = THDoubleTensor_data(a);
	double* b_data = THDoubleTensor_data(b);
	double* work = malloc(2*n*sizeof(double));
	double rho = gsl_stats_spearman(a_data, 1, b_data, 1, n, work);
	return rho;
}

static int spearman_rho(lua_State* L) {
	THDoubleTensor* a = luaT_toudata(L, -2, "torch.DoubleTensor");
	THDoubleTensor* b = luaT_toudata(L, -1, "torch.DoubleTensor");
	double rho = _spearman_rho(a, b);
	lua_pushnumber(L, rho);
	return 1;
}

static int multi(lua_State* L) {
	THDoubleTensor* a = luaT_toudata(L, -2, "torch.DoubleTensor");
	THDoubleTensor* b = luaT_toudata(L, -1, "torch.DoubleTensor");
	THDoubleTensor* sum = luaT_toudata(L, -3, "torch.DoubleTensor");
	int i,j;
	int k,x;
	for (i=0;i<3000;i++) {
		for (j=0;j<1;j++) {
			THDoubleTensor_cadd(sum, a, 1, b);
			//for (k=0;k<2500;k++) x = 1 + 3;
		}
	}
	lua_pushnumber(L, 0);
	return 1;
}

int luaopen_cutils(lua_State *L) {
	lua_register(L, "spearman_rho", spearman_rho);
	lua_register(L, "multi", multi);
	return 0;
}
