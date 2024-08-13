#ifndef JAKUBOVSKY_PROJEKT_IOSTREAMFUNCTIONS_H
#define JAKUBOVSKY_PROJEKT_IOSTREAMFUNCTIONS_H

#include "ReadJSON.h"

using namespace std;

template <typename Type>
Type convertFromSpecialCharacters(Type value) // In case the value is not a string the function should not do anything.
{
    return value;
}

string convertFromSpecialCharacters(char* text);

string convertFromSpecialCharacters(string text);

template <typename KeyType>
ostream& operator<< (ostream &os, Dict<KeyType, char*> &dict)
{
    DynamicArray<KeyType> *listOfKeys = &dict.keys();

    os << "{";

    for (auto iter = listOfKeys->begin(); iter != listOfKeys->end(); iter++)
    {
        if ( ! (os << "\"" << convertFromSpecialCharacters(*(iter)) << "\": \"" << convertFromSpecialCharacters(dict[*(iter)]) << "\""))
            throw Exception("Types of Dict could not be processed to stream.");
        if (iter + 1 != listOfKeys->end()) os << ", ";
    }

    return os << "}";
}

template <typename KeyType>
ostream& operator<< (ostream &os, Dict<KeyType, string> &dict)
{
    DynamicArray<KeyType> *listOfKeys = &dict.keys();

    os << "{";

    for (auto iter = listOfKeys->begin(); iter != listOfKeys->end(); iter++)
    {
        if ( ! (os << "\"" << (convertFromSpecialCharacters(*iter)) << "\": \"" << convertFromSpecialCharacters(dict[*(iter)]) << "\""))
            throw Exception("Types of Dict could not be processed to stream.");
        if (iter + 1 != listOfKeys->end()) os << ", ";
    }

    return os << "}";
}

template <typename KeyType, typename ValueType>
ostream& operator<< (ostream &os, Dict<KeyType, ValueType> &dict)
{
    string key, value;
    DynamicArray<KeyType> *listOfKeys = &dict.keys();

    os << "{";

    for (auto iter = listOfKeys->begin(); iter != listOfKeys->end(); iter++)
    {
        os << "\"" << convertFromSpecialCharacters(*(iter)) << "\": " << convertFromSpecialCharacters(dict[*(iter)]);
        if (iter + 1 != listOfKeys->end()) os << ", ";
    }
    os << "}";

    return os;
}

template <typename KeyType, typename ValueType>
void to_JSON(Dict<KeyType, ValueType> &dict, const char* fileName)
{
    ofstream file(fileName);

    if (file.fail()) throw Exception("Unable to access file:" + string(fileName));

    if ( ! (file << dict)) throw Exception("Could not output text.");
}
#endif //JAKUBOVSKY_PROJEKT_IOSTREAMFUNCTIONS_H