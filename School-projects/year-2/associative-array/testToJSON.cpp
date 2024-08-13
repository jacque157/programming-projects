#include "gtest/gtest.h"
#include "fstream"

#include "WriteJSON.h"


using namespace std;
using namespace ::testing;

TEST(TestJSON, ListOfKeys)
{
    Dict<int, int> dict(0.6);
    map<string, int> m;
    string message, word1, word2;

    ostringstream output;

    for (int i = 0; i < 4; i++)
    {
        try
        {
            dict[i] = i + 10;
        }
        catch (Exception &e)
        {
            message = e.getMessage();
            cout << message << endl;
            ASSERT_TRUE(false);
        }
    }

    DynamicArray<int> *a = &dict.keys();
    for (int i = 0; i < 4; i++)
    {
        ASSERT_TRUE(dict.contains((*a)[i]));
    }
}

TEST(TestJSON, SimpleDict)
{
    Dict<int, int> dict(0.6);
    map<string, int> m;
    string message, word1, word2;

    for (int i = 0; i < 4; i++)
    {
        try
        {
            dict[i] = i + 10;
        }
        catch (Exception &e)
        {
            message = e.getMessage();
            cout << message << endl;
            ASSERT_TRUE(false);
        }
    }
    try
    {
        to_JSON(dict, "test.json");
    }
    catch (Exception &e)
    {
        cout << e.getMessage() << endl;
        ASSERT_TRUE(false);
    }

    ifstream correctFile("simple_int.json");
    ifstream testFile("test.json");

    if (correctFile.fail() || testFile.fail())
    {
        cout << "Unable to open files" << endl;
        ASSERT_TRUE(false);
    }

    set<string> s1;
    set<string> s2;
    while (getline(correctFile, word1, ',') && getline(testFile, word2, ','))
    {
        word1.erase(0, 1);
        word2.erase(0, 1);
        if (word1[word1.size() - 1] == '}') word1.erase(word1.size() - 1, 1);
        if (word2[word2.size() - 1] == '}') word2.erase(word2.size() - 1, 1);

        s1.insert(word1);
        s2.insert(word2);
    }

    for (auto iter = s1.begin(); iter != s1.end(); iter++)
    {
        ASSERT_FALSE(s2.find(*iter) == s2.end());
    }
}

TEST(TestJSON, SimpleStringDict)
{
    Dict<int, string> dict(0.6);
    string message, word1, word2, word;

    ifstream inputFile("text3.txt");

    if (inputFile.fail())
    {
        cout << "Unable to open file: test3.txt" << endl;
        ASSERT_TRUE(false);
    }
    int i = 0;
    while (inputFile >> word)
    {
        try
        {
            dict[i] = word;
            i++;
        }
        catch (Exception &e)
        {
            message = e.getMessage();
            ASSERT_TRUE(false);
        }
    }

    try
    {
        to_JSON(dict, "test.json");
    }
    catch (Exception &e)
    {
        cout << e.getMessage() << endl;
        ASSERT_TRUE(false);
    }

    ifstream correctFile("simple_str.json");
    ifstream testFile("test.json");

    if (correctFile.fail() || testFile.fail())
    {
        cout << "Unable to open files" << endl;
        ASSERT_TRUE(false);
    }

    set<string> s1;
    set<string> s2;

    while (getline(correctFile, word1, ',') && getline(testFile, word2, ','))
    {
        word1.erase(0, 1);
        word2.erase(0, 1);
        if (word1[word1.size() - 1] == '}') word1.erase(word1.size() - 1, 1);
        if (word2[word2.size() - 1] == '}') word2.erase(word2.size() - 1, 1);

        s1.insert(word1);
        s2.insert(word2);
    }

    for (auto iter = s1.begin(); iter != s1.end(); iter++)
    {
        ASSERT_FALSE(s2.find(*iter) == s2.end());
    }
}

TEST(TestJSON, InnerDict)
{
    Dict<string, Dict<string, string>> dict(0.6);
    string message, sentence1, sentence2, word1, word2, word;

    try
    {
        dict["0"]["0"] = "e";
        dict["0"]["1"] = "c";
        dict["0"]["2"] = "b";
        dict["0"]["3"] = "e";

        dict["1"]["0"] = "a";

        dict["2"]["0"] = "f";
        dict["2"]["1"] = "d";
        dict["2"]["2"] = "c";

        dict["3"]["0"] = "e";
        dict["3"]["1"] = "b";
    }
    catch (Exception &e)
    {
        message = e.getMessage();
        cout << message << endl;
        ASSERT_TRUE(false);
    }

    try
    {
        to_JSON(dict, "test.json");
    }
    catch (Exception &e)
    {
        cout << e.getMessage() << endl;
        ASSERT_TRUE(false);
    }

    Dict<string, Dict<string, string>> dict2(0.6);

    read_JSON(dict2, "test.json");

    DynamicArray<string> *listOfKeys = &dict.keys();

    for (auto iter = listOfKeys->begin(); iter != listOfKeys->end(); iter++)
    {
        ASSERT_TRUE(dict2.contains(*iter));

        DynamicArray<string> *listOfKeys2 = &dict[*iter].keys();

        for (auto iter2 = listOfKeys2->begin(); iter2 != listOfKeys2->end(); iter2++)
        {
            ASSERT_TRUE(dict2[*iter].contains(*iter2));
        }
    }
}

TEST(TestJSON, FinalTest)
{
    Dict<string, Dict<string, int>> dict1(0.6);
    Dict<string, Dict<string, int>> dict2(0.6);
    try
    {
        dict1["\""]["{"] = 1;
        dict1["\n"]["{"] = 2;
        dict1["}"]["{"] = 3;
        dict1[":"]["{"] = 4;
        dict1["\\"]["{"] = 5;

        to_JSON(dict1, "test.json");



        read_JSON(dict2, "test.json");

        ASSERT_EQ(dict2["\""]["{"], 1);
        ASSERT_EQ(dict2["\n"]["{"], 2);
        ASSERT_EQ(dict2["}"]["{"], 3);
        ASSERT_EQ(dict2[":"]["{"], 4);
        ASSERT_EQ(dict2["\\"]["{"], 5);
    }
    catch (Exception &e)
    {
        cout << e.getMessage() << endl;
        ASSERT_TRUE(false);
    }

}