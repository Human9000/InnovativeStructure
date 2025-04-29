cmake_minimum_required(VERSION 3.27)
project(Q8_6 C)

set(CMAKE_C_STANDARD 11)

include_directories(h)
include_directories(h_sw)

add_executable(Q8_6
        h/layer.h
        h/layer_ex.h
        h/net.h
        h/opt.h
        h/opt_ex.h
        h/tool.h
#        test/sw_seg_20_12_500_100.c
#        test/net_sw.c
#        net_sw.c
        test/cls_net_sw_125.c
        #        test/test_softmax.c
)
