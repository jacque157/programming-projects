#include <iostream>
#include "gtest/gtest.h"
#include <string>
#include <fstream>

#include "AssociativeArray.h"

using namespace std;
using namespace ::testing;

TEST(TestDict, TestAssignment)
{
    string sentence[] = {"Mama", "ma,", "emu", "a", "Ema", "ma", "mamu"};
    Dict<string, int> d(0.9);

    for (int i = 0; i < 7; i++)
    {
        d.insert(sentence[i], 0);
    }

    for (int i = 0; i < 7; i++)
    {
        ASSERT_EQ(d.get_value(sentence[i]), 0);
    }

    ASSERT_EQ(d.size(), 7);
}

TEST(TestDict, ImproperCapacity1)
{
    string message;
    try
    {
        Dict<string, int> d(-5);
        ASSERT_TRUE(false);
    }
    catch (const Exception &e)
    {
        message = e.getMessage();
    }
    ASSERT_EQ("Load capacity must be in interval 0 to 1 not included.", message);
}

TEST(TestDict, ImproperCapacity2)
{
    string message;
    try
    {
        Dict<string, int> d(1);
        ASSERT_TRUE(false);
    }
    catch (const Exception &e)
    {
        message = e.getMessage();
    }
    ASSERT_EQ("Load capacity must be in interval 0 to 1 not included.", message);
}

TEST(TestDict, TestExceedingSize)
{
    string sentence[] = {"a", "set", "of", "words", "that", "is", "complete", "in", "itself,",
                         "typically", "containing", "a", "subject", "and", "predicate,", "conveying", "a",
                         "statement,", "question,", "exclamation,", "or", "command,", "and", "consisting",
                         "of", "a", "main", "clause", "and", "sometimes", "one", "or", "more", "subordinate",
                         "clauses."};

    Dict<string, int> dict(0.6);
    map<string, int> m;
    for (int i = 0; i < 35; i++)
    {
        dict.insert(sentence[i], i);
        m[sentence[i]] = i;
    }

    for (int i = 0; i < 35; i++)
    {
        ASSERT_EQ(dict.get_value(sentence[i]), m[sentence[i]]);
    }
    ASSERT_EQ(m.size(), dict.size());
}

TEST(TestDict, FrequencyTable1)
{
    Dict<string, int> dict(0.6);
    map<string, int> m;
    ifstream file("text1.txt");
    if (!file)
    {
        cout << "File does not exist." << endl;
        ASSERT_TRUE(false);
    }
    string word;
    while (file >> word)
    {
        if (dict.contains(word))
        {
            dict[word] += 1;
            m[word] += 1;
        }
        else
        {
            dict.insert(word, 1);
            m[word] = 1;
        }
    }

    for (auto iter = m.begin(); iter != m.end() ; iter++)
    {
        ASSERT_EQ(iter->second, dict.get_value(iter->first));
    }
    ASSERT_EQ(m.size(), dict.size());
}

TEST(TestDict, KeyError)
{
    string sentence[] = {"a", "set", "of", "words", "that", "is", "complete", "in", "itself,",
                         "typically", "containing", "a", "subject", "and", "predicate,", "conveying", "a",
                         "statement,", "question,", "exclamation,", "or", "command,", "and", "consisting",
                         "of", "a", "main", "clause", "and", "sometimes", "one", "or", "more", "subordinate",
                         "clauses."};

    Dict<string, int> dict(0.6);
    map<string, int> m;
    for (int i = 0; i < 35; i++)
    {
        dict.insert(sentence[i], i);
        m[sentence[i]] = i;
    }

    ASSERT_EQ(m.size(), dict.size());

    dict.remove("a");

    ASSERT_EQ(m.size() - 1, dict.size());

    string message1;
    try
    {
        dict.remove("a");
        ASSERT_TRUE(false);
    }
    catch (const Exception &e1)
    {
        message1 = e1.getMessage();
    }
    ASSERT_EQ("Key does not exist.", message1);
    string message2;
    try
    {
        dict.get_value("a");
        ASSERT_TRUE(false);
    }
    catch (const Exception &e2)
    {
        message2 = e2.getMessage();
    }
    ASSERT_EQ("Key does not exist.", message2);
}

TEST(TestDict, FrequencyTableOperator)
{
    Dict<string, int> dict(0.6);
    map<string, int> m;
    string message;
    ifstream file("text1.txt");
    if (!file)
    {
        cout << "File does not exist" << endl;
        ASSERT_TRUE(false);
    }
    string word;
    while (file >> word)
    {
        try
        {
            if (m.find(word) != m.end())
            {
                dict[word] += 1;
                m[word] += 1;
            }
            else
            {
                dict[word] = 1;
                m[word] = 1;
            }
        }
        catch (const Exception &e)
        {
            message = e.getMessage();
            cout << message.c_str() << endl;
            ASSERT_FALSE(true);
        }
    }
    for (auto iter = m.begin(); iter != m.end() ; iter++)
    {
        try
        {
            ASSERT_EQ(iter->second, dict.get_value(iter->first));
        }
        catch (const Exception &e)
        {
            message = e.getMessage();
            cout << iter->first << endl;
            cout << message.c_str() << endl;
            ASSERT_TRUE(false);
        }
    }
}

TEST(TestDict, FrequencyTable2)
{
    Dict<string, int> dict(0.6);
    map<string, int> m;
    ifstream file("text2.txt");
    string message;
    if (!file)
    {
        cout << "File does not exist" << endl;
        ASSERT_TRUE(false);
    }
    string word;
    while (file >> word)
    {
        try
        {
            if (m.find(word) != m.end())
            {
                dict[word] += 1;
                m[word] += 1;
            }
            else
            {
                dict[word] = 1;
                m[word] = 1;
            }
        }
        catch (const Exception &e)
        {
            message = e.getMessage();
            cout << message.c_str() << endl;
            ASSERT_TRUE(false);
        }
    }

    ASSERT_EQ(m.size(), dict.size());

    for (auto iter = m.begin(); iter != m.end() ; iter++)
    {
        try
        {
            ASSERT_EQ(iter->second, dict.get_value(iter->first));
        }
        catch (const Exception &e)
        {
            message = e.getMessage();
            cout << message.c_str() << endl;
            ASSERT_TRUE(false);
        }
    }
}

TEST(TestDict, Remove)
{
    Dict<string, int> dict(0.6);
    map<string, int> m1;
    ifstream file1("text1.txt");
    string message;
    if (!file1)
    {
        cout << "File does not exist" << endl;
        ASSERT_TRUE(false);
    }
    string word;
    while (file1 >> word)
    {
        try
        {
            if (dict.contains(word))
            {
                dict[word] += 1;
                m1[word] += 1;
            }
            else
            {
                dict[word] = 1;
                m1[word] = 1;
            }
        }
        catch (const Exception &e)
        {
            message = e.getMessage();
            cout << message.c_str() << endl;
            ASSERT_FALSE(true);
        }

    }

    for (auto iter = m1.begin(); iter != m1.end() ; iter++)
    {
        try
        {
            dict.remove(iter->first);
        }
        catch (const Exception &e)
        {
            message = e.getMessage();
            cout << message.c_str() << endl;
            ASSERT_FALSE(true);
        }
    }

    ASSERT_EQ(0, dict.size());

    map<string, int> m2;
    ifstream file2("text2.txt");
    if (!file2)
    {
        cout << "File does not exist" << endl;
        ASSERT_TRUE(false);
    }

    while (file2 >> word)
    {
        try
        {
            if (dict.contains(word))
            {
                dict[word] += 1;
                m2[word] += 1;
            }
            else
            {
                dict[word] = 1;
                m2[word] = 1;
            }
        }
        catch (const Exception &e)
        {
            message = e.getMessage();
            cout << message.c_str() << endl;
            ASSERT_FALSE(true);
        }
    }

    for (auto iter = m2.begin(); iter != m2.end() ; iter++)
    {
        ASSERT_EQ(iter->second, dict.get_value(iter->first));
    }
    ASSERT_EQ(m2.size(), dict.size());
}

