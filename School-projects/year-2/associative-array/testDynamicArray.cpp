#include <iostream>
#include "gtest/gtest.h"

#include "DynamicArray.h"
#include "ArrayException.h"

using namespace std;
using namespace ::testing;

TEST(TestDynamicArray, ReadingWriting)
{
    DynamicArray<int> a(5);
    for (int i = 0; i < 5; i++)
    {
        a[i] = 10 + i;
    }
    for (int i = 0; i < 5; i++)
    {
        ASSERT_EQ(10 + i, a[i]);
    }
}

TEST(TestDynamicArray, getSize)
{
    DynamicArray<int> a(5);
    for (int i = 0; i < 5; i++)
    {
        a[i] = 10 + i;
    }
    ASSERT_EQ(5, a.get_size());
}

TEST(TestDynamicArray, OutOfRange)
{
    DynamicArray<int> a(5);
    string message;
    try
    {
        a[6];
        ASSERT_TRUE(false);
    }
    catch (const Exception &e)
    {
        message = e.getMessage();
    }
    ASSERT_EQ("index out of range", message);
}

TEST(TestDynamicArray, BelowZero)
{
    DynamicArray<int> a(5);
    string message;
    try
    {
        a[-20];
        ASSERT_TRUE(false);
    }
    catch (const Exception &e)
    {
        message = e.getMessage();
    }
    ASSERT_EQ("index below zero", message);
}

TEST(TestDynamicArray, WrongSize)
{
    DynamicArray<int> a(5);
    string message;
    try
    {
        a.resize(-90);
        ASSERT_TRUE(false);
    }
    catch (const Exception &e)
    {
        message = e.getMessage();
    }
    ASSERT_EQ("new size cannot be below 0", message);
}

TEST(TestDynamicArray, Resize)
{
    DynamicArray<int> a(5);

    for (int i = 0; i < 5; i++)
    {
        a[i] = 10 + i;
    }
    ASSERT_EQ(5, a.get_size());

    a.resize(10);
    for (int i = 0; i < 10; i++)
    {
        a[i] = 10 + i;
    }
    ASSERT_EQ(10, a.get_size());

    for (int i = 0; i < 10; i++)
    {
        ASSERT_EQ(10 + i, a[i]);
    }
}

TEST(TestDynamicArray, RandLRefferences)
{
    DynamicArray<int> a(5);
    for (int i = 0; i < a.get_size(); i++)
    {
        a[i] = 10 + i;
    }
    a.resize(10);
    for (int i = 0; i < a.get_size(); i++)
    {
        a[i] = 10 + i;
    }

    for (int i = 0; i < a.get_size(); i++)
    {
        a[i] = 10 + a[i];
        ASSERT_EQ(20 + i, a[i]);
    }
}

TEST(TestDynamicArray, StringArray)
{
    string s[] = {"I", "hope", "this", "works", "as intended"};
    DynamicArray<string> a(5);
    for (int i = 0; i < a.get_size(); i++)
    {
        a[i] = s[i];
    }

    for (int i = 0; i < a.get_size(); i++)
    {
        ASSERT_EQ(s[i], a[i]);
    }
}

TEST(TestDynamicArray, Iterator)
{
    string s[] = {"I", "hope", "this", "works", "as intended"};
    DynamicArray<string> a(5);
    int i = 0;
    for (auto iter = a.begin(); iter != a.end(); i++, iter++)
    {
        *iter = s[i];
    }

    i = 0;
    for (auto iter = a.begin(); iter != a.end(); i++, iter++)
    {
        ASSERT_EQ(s[i], *iter);
    }
}

TEST(TestDynamicArray, IteratorForEmpty)
{
    DynamicArray<string> a(0);

    for (auto iter = a.begin(); iter != a.end(); iter++)
    {
        ASSERT_TRUE(false);
    }
    ASSERT_TRUE(true);
}

TEST(TestDynamicArray, Asignment)
{
    DynamicArray<int> a(0);
    DynamicArray<int> b(10);
    for (int i = 0; i < 10; i++)
    {
        b[i] = 10 + i;
    }
    a = b;
    for (int i = 0; i < 10; i++)
    {
        ASSERT_EQ(10 + i, a[i]);
        ASSERT_EQ(10 + i, b[i]);
    }
}

TEST(TestDynamicArray, CopyConstructor)
{
    DynamicArray<int> b(10);
    for (int i = 0; i < 10; i++)
    {
        b[i] = 10 + i;
    }
    DynamicArray<int>  a(b);
    for (int i = 0; i < 10; i++)
    {
        ASSERT_EQ(10 + i, a[i]);
        ASSERT_EQ(10 + i, b[i]);
    }
}

TEST(TestDynamicArray, MoveConstructor)
{
    DynamicArray<int> b(10);
    for (int i = 0; i < 10; i++)
    {
        b[i] = 10 + i;
    }

    DynamicArray<int>  a(move(b));

    ASSERT_EQ(a.get_size(), 10);
    ASSERT_EQ(b.get_size(), 0);

    for (int i = 0; i < 10; i++)
    {
        ASSERT_EQ(10 + i, a[i]);
    }
}