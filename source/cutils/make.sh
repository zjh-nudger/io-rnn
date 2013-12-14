#gcc -fopenmp -Wall -shared -fPIC -o cutils.so -I/datastore/phong/usr/include/torch -I/datastore/phong/usr/include -L/datastore/phong/usr/lib/libluaT.so -L/datastore/phong/usr/lib/libTH.so -L/datastore/phong/usr/lib/libgsl.so -L/datastore/phong/usr/lib/libgslcblas.so -L/datastore/phong/usr/lib/libransampl.so cutils.c

gcc -lgsl -lgslcblas -lransampl -fopenmp -Wall -shared -fPIC -o cutils.so -I/datastore/phong/usr/include/torch -I/datastore/phong/usr/include -L/datastore/phong/usr/lib cutils.c
