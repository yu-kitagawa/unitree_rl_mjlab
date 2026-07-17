cmake_minimum_required(VERSION 3.22)
project(ZmqOpenCcReceiver)

set(CMAKE_CXX_STANDARD 17)

# 1. パッケージの検索
find_package(OpenCV REQUIRED)
find_package(cppzmq REQUIRED)

# 2. 実行ファイルの追加
add_executable(zmq_receiver main.cpp)

# 3. ターゲットにインクルードとリンクを紐付け
target_include_directories(zmq_receiver PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(zmq_receiver PRIVATE ${OpenCV_LIBS} zmq)
