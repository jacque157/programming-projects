#include <iostream>
#include "gtest/gtest.h"
#include <string>
#include "fstream"

#include "WriteJSON.h"

using namespace std;
using namespace ::testing;

TEST(TestJSON, ReadJSONint)
{
    Dict<string, int> dict(0.6);
    read_JSON(dict, "simple_int.json");

    ASSERT_EQ(dict["0"], 10);
    ASSERT_EQ(dict["1"], 11);
    ASSERT_EQ(dict["2"], 12);
    ASSERT_EQ(dict["3"], 13);
}

TEST(TestJSON, ReadJSONstr)
{
    Dict<string, string> dict(0.6);
    read_JSON(dict, "simple_str.json");

    ASSERT_EQ(dict["0"], "I");
    ASSERT_EQ(dict["1"], "look");
    ASSERT_EQ(dict["2"], "at");
    ASSERT_EQ(dict["3"], "you");
    ASSERT_EQ(dict["19"], "They're");
    ASSERT_EQ(dict["23"], "the");
    ASSERT_EQ(dict["24"], "cornfields");
    ASSERT_EQ(dict["35"], "Will");
    ASSERT_EQ(dict["40"], "here.");
}

TEST(TestJSON, ReadJSONdict)
{
    Dict<string, Dict<string, string>> dict(0.6);
    read_JSON(dict, "dict.json");

    ASSERT_EQ(dict["0"]["0"], "e");
    ASSERT_EQ(dict["0"]["1"], "c");
    ASSERT_EQ(dict["0"]["2"], "b");
    ASSERT_EQ(dict["0"]["3"], "e");

    ASSERT_EQ(dict["1"]["0"], "a");

    ASSERT_EQ(dict["2"]["0"], "f");
    ASSERT_EQ(dict["2"]["1"], "d");
    ASSERT_EQ(dict["2"]["2"], "c");

    ASSERT_EQ(dict["3"]["0"], "e");
    ASSERT_EQ(dict["3"]["1"], "b");
}

TEST(TestJSON, ReadJSONdeep)
{
    Dict<string, Dict<string, Dict<string, int>>> dict(0.6);
    read_JSON(dict, "deeper_dict.json");

    ASSERT_EQ(dict["0"]["0"]["0"], 10);
    ASSERT_EQ(dict["0"]["0"]["1"], 11);

    ASSERT_EQ(dict["0"]["1"]["0"], 200);
    ASSERT_EQ(dict["0"]["1"]["1"], 201);

    ASSERT_EQ(dict["1"]["0"]["0"], 4000);
    ASSERT_EQ(dict["1"]["0"]["1"], 4001);

    ASSERT_EQ(dict["1"]["1"]["0"], 80000);
    ASSERT_EQ(dict["1"]["1"]["1"], 80001);
}

TEST(TestJSON, ReadJSONdeepToString)
{
    Dict<string, Dict<string, Dict<string, string>>> dict(0.6);
    read_JSON(dict, "deeper_dict.json");

    ASSERT_EQ(dict["0"]["0"]["0"], "10");
    ASSERT_EQ(dict["0"]["0"]["1"], "11");

    ASSERT_EQ(dict["0"]["1"]["0"], "200");
    ASSERT_EQ(dict["0"]["1"]["1"], "201");

    ASSERT_EQ(dict["1"]["0"]["0"], "4000");
    ASSERT_EQ(dict["1"]["0"]["1"], "4001");

    ASSERT_EQ(dict["1"]["1"]["0"], "80000");
    ASSERT_EQ(dict["1"]["1"]["1"], "80001");
}

TEST(TestJSON, ReadJSONindentedDeepToString)
{
    Dict<string, Dict<string, Dict<string, string>>> dict(0.6);
    read_JSON(dict, "indented_deeper_dict.json");

    ASSERT_EQ(dict["0"]["0"]["0"], "10");
    ASSERT_EQ(dict["0"]["0"]["1"], "11");

    ASSERT_EQ(dict["0"]["1"]["0"], "200");
    ASSERT_EQ(dict["0"]["1"]["1"], "201");

    ASSERT_EQ(dict["1"]["0"]["0"], "4000");
    ASSERT_EQ(dict["1"]["0"]["1"], "4001");

    ASSERT_EQ(dict["1"]["1"]["0"], "80000");
    ASSERT_EQ(dict["1"]["1"]["1"], "80001");
}

TEST(TestJSON, ReadJSONspecialChars)
{
    Dict<string, string> dict(0.6);
    try
    {
        read_JSON(dict, "special_chars.json");
    }
    catch(Exception &e)
    {
        cout << e.getMessage() << endl;
        ASSERT_TRUE(false);
    }

    ASSERT_EQ(dict["1"], "{");
    ASSERT_EQ(dict["2"], "}");
    ASSERT_EQ(dict["3"], ":");
    ASSERT_EQ(dict["4"], ",");
    ASSERT_EQ(dict["5"], "\"");

}

TEST(TestJSON, TestNonExisting)
{
    Dict<string, Dict<string, Dict<string, string>>> dict(0.6);

    try
    {
        read_JSON(dict, "beep-boop.json");
        ASSERT_TRUE(false);
    }
    catch (Exception &e)
    {
        ASSERT_EQ("file: beep-boop.json was not found.", e.getMessage());
    }
}

