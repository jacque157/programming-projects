# include "ReadJSON.h"

string convertToSpecialCharacters(string text)
{
    int position;
    while(true)
    {
        if ((position = text.find("\\\"")) != string::npos) text.replace(position, 2, "\"");
        else if ((position = text.find("\\n")) != string::npos) text.replace(position, 2, "\n");
        else if ((position = text.find("\\t")) != string::npos) text.replace(position, 2, "\t");
        else if ((position = text.find("\\v")) != string::npos) text.replace(position, 2, "\v");
        else if ((position = text.find("\\b")) != string::npos) text.replace(position, 2, "\b");
        else if ((position = text.find("\\r")) != string::npos) text.replace(position, 2, "\r");
        else if ((position = text.find("\\f")) != string::npos) text.replace(position, 2, "\f");
        else if ((position = text.find("\\a")) != string::npos) text.replace(position, 2, "\a");
        else if ((position = text.find("\\\\")) != string::npos) text.replace(position, 2, "\\");
        else return text;
    }
}

string trim (string text, char character)
{
    if (text.length() > 0 && (text[0] == character)) text = text.substr(1);
    if (text.length() > 0 && text[text.length() - 1] == character) text = text.substr(0, text.length() - 1);
    return text;
}

istringstream& operator>> (istringstream& pairStream, Dict<string, string> &dict)
{
    string key, value;

    if ( ! (pairStream >> key >> value))
        throw Exception("Stream could not be parsed to types of Dict.");

    key = trim(key, '\"');
    value = trim(value, '\"');
    dict.insert(convertToSpecialCharacters(key), convertToSpecialCharacters(value));
    return pairStream;
}