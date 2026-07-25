add_buckyball_linux_ctest(
  lenet_int8_bemu
  ${CMAKE_CURRENT_SOURCE_DIR}
  lenet_int8_bemu.c)

add_buckyball_linux_ctest(
  lenet_int8_bemu_eval
  ${CMAKE_CURRENT_SOURCE_DIR}
  lenet_int8_bemu_eval.c)

add_buckyball_linux_ctest(
  lenet_int8_channel_bemu_eval
  ${CMAKE_CURRENT_SOURCE_DIR}
  lenet_int8_channel_bemu_eval.c)

add_dependencies(pebble-ctests-build lenet_int8_bemu-linux-build)
add_dependencies(pebble-ctests-build lenet_int8_bemu_eval-linux-build)
add_dependencies(pebble-ctests-build lenet_int8_channel_bemu_eval-linux-build)
