cmake_minimum_required(VERSION 3.17)
project(Jakubovsky_projekt)

set(CMAKE_CXX_STANDARD 14)

find_package(Threads)

set(SOURCE_FILES
        ArrayException.h
        DynamicArray.h
        Item.h
        AssociativeArray.h
        ReadJSON.h
        WriteJSON.h

        WriteJSON.cpp
        ReadJSON.cpp

        testDynamicArray.cpp
        testItem.cpp
        testAssociativeArray.cpp
        testToJSON.cpp
        testReadJSON.cpp

        GoogleTest/gtest-death-test.cc
        GoogleTest/gtest-filepath.cc
        GoogleTest/gtest-port.cc
        GoogleTest/gtest-printers.cc
        GoogleTest/gtest-test-part.cc
        GoogleTest/gtest-typed-test.cc
        GoogleTest/gtest.cc
        GoogleTest/gtest_main.cc)

include_directories(GoogleTest)

add_executable(Jakubovsky_projekt ${SOURCE_FILES})

target_link_libraries(Jakubovsky_projekt ${CMAKE_THREAD_LIBS_INIT})

add_custom_command(TARGET Jakubovsky_projekt PRE_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/test_files/text1.txt
        ${CMAKE_SOURCE_DIR}/test_files/text2.txt
        ${CMAKE_SOURCE_DIR}/test_files/text3.txt
        ${CMAKE_SOURCE_DIR}/test_files/simple_int.json
        ${CMAKE_SOURCE_DIR}/test_files/simple_str.json
        ${CMAKE_SOURCE_DIR}/test_files/dict.json
        ${CMAKE_SOURCE_DIR}/test_files/deeper_dict.json
        ${CMAKE_SOURCE_DIR}/test_files/special_chars.json
        ${CMAKE_SOURCE_DIR}/test_files/indented_deeper_dict.json
        $<TARGET_FILE_DIR:Jakubovsky_projekt>
        VERBATIM)