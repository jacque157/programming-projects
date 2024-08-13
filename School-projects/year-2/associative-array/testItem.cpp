#include <iostream>
#include "gtest/gtest.h"

#include "DynamicArray.h"
#include "Item.h"

using namespace std;
using namespace ::testing;

TEST(TestItemDynamicArray, ItemTest)
{
    Item<string, int> tuple("abeceda", 1);
    ASSERT_EQ("abeceda", tuple.get_key());
    ASSERT_EQ(1, tuple.get_value());

    tuple.value() = tuple.value() + 1;
    ASSERT_EQ(2, tuple.get_value());
}

TEST (TestItemDynamicArray, ItemArray)
{
    string alphabet[] = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"};
    DynamicArray<Item<string, int>> d(10);

    for (int i = 0; i < d.get_size(); i++)
    {
        Item<string, int> tuple(alphabet[i], i);
        d[i] = tuple;
    }

    ASSERT_EQ(10, d.get_size());
    for (int i = 0; i < d.get_size(); i++)
    {
        ASSERT_EQ(alphabet[i], d[i].get_key());
        ASSERT_EQ(i, d[i].get_value());
    }

    d.resize();
    ASSERT_EQ(20, d.get_size());
}

TEST (TestItemDynamicArray, ItemArrayEmpty)
{
    DynamicArray<Item<string, int>> d(10);

    for (int i = 0; i < d.get_size(); i++)
    {
        ASSERT_TRUE(d[i].get_type() == ItemType::EMPTY);
    }
}