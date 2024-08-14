#ifndef JAKUBOVSKY_PROJEKT_DYNAMICARRAY_H
#define JAKUBOVSKY_PROJEKT_DYNAMICARRAY_H

#include "ArrayException.h"

template <typename Type> class DynamicArray
{
    Type *array;
    int size = 0;

public:
    DynamicArray(int length=10)
    {
        size = length;
        array = new Type[length + 1];
    }

    DynamicArray(DynamicArray<Type> &other)
    {
        if (this != &other)
        {
            size = other.get_size();
            array = new Type[size + 1];

            for (int i = 0; i < size; i++)
            {
                (*this)[i] = other[i];
            }
        }
    }

    DynamicArray(DynamicArray<Type> &&other)
    {
        if (this != &other)
        {
            size = other.get_size();
            array = new Type[size + 1];

            for (int i = 0; i < size; i++)
            {
                (*this)[i] = other[i];
            }
            other.resize(0);
        }
    }

    Type& operator[](int index)
    {
        if (index >= size) throw Exception("index out of range");
        if (index < 0) throw Exception("index below zero");
        return array[index];
    }

    DynamicArray<Type>& operator=(DynamicArray<Type> &other)
    {
        resize(other.get_size());

        for (int i = 0; i < size; i++)
        {
            (*this)[i] = other[i];
        }

        return *this;
    }

    void resize(int new_size)
    {
        if (new_size < 0) throw Exception("new size cannot be below 0");
        Type *new_array = new Type[new_size + 1];

        size = new_size;
        delete [] array;
        array = new_array;
    }

    void resize()
    {
        resize(2 * size);
    }

    int get_size() const { return size; };
    Type* begin() { return array; };
    Type* end()
    {
        if (size == 0) return begin();
        return &array[size];
    };

    ~DynamicArray() { delete[] array; };
};

#endif //JAKUBOVSKY_PROJEKT_DYNAMICARRAY_H
