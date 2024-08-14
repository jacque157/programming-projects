#ifndef JAKUBOVSKY_PROJEKT_WRITEJSON_H
#define JAKUBOVSKY_PROJEKT_WRITEJSON_H

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

#include "AssociativeArray.h"

#pragma once

using namespace std;

string convertToSpecialCharacters(string text);

string trim (string text, char character);

template <typename ValueType>
istringstream& operator>> (istringstream& pairStream, Dict<string, ValueType> &dict)
{
    string key;
    ValueType value;

    if ( ! (pairStream >> key >> value))
        throw Exception("Stream could not be parsed to types of Dict.");

    key = trim(key, '\"');
    dict.insert(convertToSpecialCharacters(key), value);
    return pairStream;
}

istringstream& operator>> (istringstream& pairStream, Dict<string, string> &dict);

template <typename ValueType>
bool insertPair(Dict<string, ValueType> &dict, string pair)
{
    string key;
    ValueType value;
    bool readingString = false;
    bool backslash = false;
    int amountOfColons = 0;

    if (pair.find(':') != string::npos)
    {
        for (int i = 0; i < pair.size(); i++)
        {

            if ( (pair[i] == '\"') && ! backslash)
                readingString = ! readingString; // switches true to false and vice versa

            if ((pair[i] == '{' || pair[i] == '}' || pair[i] == ',') && ! readingString)
                return false;

            if (pair[i] == ':' && ! readingString)
            {
                amountOfColons++;
                if (amountOfColons > 1) return false;
                pair[i] = ' ';
            }

            if (pair[i] == '\\') // enables or disables the meaning of backslash
                backslash = ! backslash;
            else
                backslash = false;
        }

        istringstream pairStream(pair);
        if (pairStream >> dict) return true;
    }
    return false;
}

template <typename ValueType>
void read_JSON(Dict<string, ValueType> &dict, const char* fileName)
{
    ostringstream outputString;
    ifstream file(fileName);

    string pair, key, line, list, text;
    ValueType value;

    if (file.fail()) throw Exception("file: " + string(fileName) + " was not found.");

    while (file >> line)
    {
        text += line; // loads file into one line
    };
    deconstructToPairs(text, dict);
}

template <typename ValueType>
void insertPairs(string listOfPairs, Dict<string, ValueType> &dict)
{
    string pair;
    bool readingString = false;
    bool backslash = false;
    int start = 0;

    for (int i = 0; i < listOfPairs.size(); i++)
    {
        if ( (listOfPairs[i] == '\"') && ! backslash)
            readingString = ! readingString; // switches true to false and vice versa

        if (listOfPairs[i] == ',' && ! readingString)
        {
            pair = listOfPairs.substr(start, i - start);
            pair = trim(pair, ' ');
            if ( ! insertPair(dict, pair)) throw Exception("Incorrect format of JSON file.");
            start = i + 1;
        }

        if (i == listOfPairs.size() - 1)
        {
            pair = listOfPairs.substr(start, i - start + 1);
            pair = trim(pair, ' ');
            if ( ! insertPair(dict, pair)) throw Exception("Incorrect format of JSON file.");
        }

        if (listOfPairs[i] == '\\') // enables or disables the meaning of backslash
            backslash = ! backslash;
        else
            backslash = false;
    }
}

template <typename ValueType>
void deconstructToPairs(string listOfValues, Dict<string,ValueType> &dict)
{
    listOfValues = trim(listOfValues, '{');
    listOfValues = trim(listOfValues, '}');

    insertPairs(listOfValues, dict);
}

template <typename ValueType>
void deconstructToPairs(string listOfValues, Dict<string, Dict<string, ValueType>> &dict)
{
    listOfValues = trim(listOfValues, '{');
    listOfValues = trim(listOfValues, '}');

    if ( listOfValues.find('{') == string::npos)
        throw Exception("Incorrect format of JSON file.");

    int keyStart = 0;   // starting index of key
    int valuesStart = 0;    // starting index of value
    int keySize= 0;     // length of string key
    int valuesSize = 0;     // length of string value
    int cbCount = 0;    // Curly brackets count: +1 for '{' -1 for '}'

    bool readingString = false;
    bool backslash = false;

    string values, key;

    for (int i = 0; i < listOfValues.size(); i++)
    {
        if ( (listOfValues[i] == '\"') && ! backslash)
            readingString = ! readingString; // switches true to false and vice versa

        if (listOfValues[i] == '{' && ! readingString)
        {
            if (cbCount == 0)
            {
                key = listOfValues.substr(keyStart + 1, keySize - 1);
                valuesStart = keyStart + keySize;
                keySize = 0;

                key = trim(key, ',');
                key = trim(key, ' ');
                key = trim(key, ':');
                key = trim(key, '\"');
                // trims: <, "<key>": > to: <<key>>
            }
            cbCount++;
        }
        if (listOfValues[i] == '}' && ! readingString)
        {
            cbCount--;
            if (cbCount == 0)
            {
                values = listOfValues.substr(valuesStart, valuesSize + 1);
                keyStart = valuesStart + valuesSize;
                valuesSize = 0;

                deconstructToPairs(values, dict[convertToSpecialCharacters(key)]);
            }
        }
        if (cbCount == 0)
        {
            keySize++;
        }
        else
        {
            valuesSize++;
        }

        if (listOfValues[i] == '\\') // enables or disables the meaning of backslash
            backslash = ! backslash;
        else
            backslash = false;
    }
    if (cbCount != 0) throw Exception("Incorrect format of JSON file.");
}

#endif //JAKUBOVSKY_PROJEKT_WRITEJSON_H

