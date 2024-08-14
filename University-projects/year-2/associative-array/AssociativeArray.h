#ifndef PROGRAMOVANIE_3_PROJEKT_ASSOCIATIVEARRAY_H
#define PROGRAMOVANIE_3_PROJEKT_ASSOCIATIVEARRAY_H

#include <iostream>
#include <string>
#include "functional"

#include "DynamicArray.h"
#include "Item.h"

using namespace std;

template <typename KeyType, typename ValueType> class Dict
{
    DynamicArray<Item<KeyType, ValueType>> array;
    int numberOfItems = 0;
    double loadCapacity;

    void _resize(const int new_size)
    {
        numberOfItems = 0;
        int sizeBefore = array.get_size();
        DynamicArray<Item<KeyType, ValueType>> temporaryArray(sizeBefore);

        for (int i = 0; i < sizeBefore; i++)
        {
            temporaryArray[i] = array[i];
        }
        array.resize(new_size);

        for (int i = 0; i < sizeBefore; i++)
        {
            if (temporaryArray[i].is_full())
            {
                insert(temporaryArray[i].key(), temporaryArray[i].value());
            }
        }
    }

    Item<KeyType, ValueType>& _find(const KeyType key)
    {
        hash<KeyType> key_hash;
        int index = key_hash(key) % array.get_size();
        int start = index;
        int firstAvailable = -1;

        while (true)
        {
            if (array[index].get_type() == ItemType::EMPTY)
            {
                if (firstAvailable == -1)
                {
                    firstAvailable = index;
                }
                return array[firstAvailable];
            }
            if (array[index].get_type() == ItemType::AVAIL)
            {
                if (firstAvailable == -1)
                {
                    firstAvailable = index;
                }
            }
            else if (array[index].get_type() == ItemType::FULL && array[index].get_key() == key)
            {
                return array[index];
            }

            index = (index + 1) % array.get_size();
            if (index == start) throw Exception("Array full.");
        }
    }

public:
    Dict(const double capacity=0.6)
    {
        if (capacity >= 1.0 || capacity <= 0) throw Exception("Load capacity must be in interval 0 to 1 not included.") ;
        array.resize(10);
        loadCapacity = capacity;
    }

    int size() const { return numberOfItems; };

    void insert(const KeyType key, ValueType value)
    {
        if ((double)(numberOfItems) > loadCapacity * array.get_size())
        {
            _resize(array.get_size() * 2);
        }

        Item<KeyType, ValueType>& item = _find(key);

        item.set_value(value);

        if (item.is_empty())
        {
            item.set_key(key);
            numberOfItems++;
        }
        item.set_type(ItemType::FULL);
    }

    ValueType& get_value(const KeyType key)
    {
        Item<KeyType, ValueType>& item = _find(key);

        if (item.is_empty()) throw Exception("Key does not exist.");

        return item.value();
    }

    void remove(const KeyType key)
    {
        Item<KeyType, ValueType>& item = _find(key);

        if (item.is_empty()) throw Exception("Key does not exist.");
        numberOfItems--;
        item.set_type(ItemType::AVAIL);

    }

    bool contains(const KeyType key)
    {
        Item<KeyType, ValueType>& item = _find(key);
        if (item.is_empty()) return false;
        return true;
    }

    ValueType& operator[](const KeyType key)
    {
        Item<KeyType, ValueType>* item = &_find(key);
        if (item->is_empty())
        {
            if ((double)(numberOfItems) > loadCapacity * array.get_size())
            {
                _resize(array.get_size() * 2);
            }
            numberOfItems++;

            item = &_find(key);
            item->set_key(key);
            item->set_type(ItemType::FULL);
        }
        return item->value();
    }

    DynamicArray<KeyType>& keys()
    {
        DynamicArray<KeyType> *listOfKeys = new DynamicArray<KeyType>(numberOfItems);

        for (int index = 0, itemsCount = 0; itemsCount < numberOfItems; index++)
        {
            if (array[index].is_full())
            {
                (*listOfKeys)[itemsCount] = array[index].get_key();
                itemsCount++;
            }
        }
        return *listOfKeys;
    }
};

#endif //PROGRAMOVANIE_3_PROJEKT_ASSOCIATIVEARRAY_H
