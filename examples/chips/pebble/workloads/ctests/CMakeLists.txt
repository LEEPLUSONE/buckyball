add_buckyball_linux_ctest(
  lenet_int8_bemu
  ${CMAKE_CURRENT_SOURCE_DIR}
  lenet_int8_bemu.c)

add_dependencies(pebble-ctests-build lenet_int8_bemu-linux-build)
