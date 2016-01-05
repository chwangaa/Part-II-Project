#ifndef MATRIX_SETTING_H
#define MATRIX_SETTING_H

typedef int Dtype;
#define DEBUG 0

#ifdef DEBUG
#define debug_assert(...) do{ assert(__VA_ARGS__); } while( false )
#else
#define debug_assert(...) do{ } while ( false )
#endif


#endif