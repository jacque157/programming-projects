# include "WriteJSON.h"

string convertFromSpecialCharacters(char* text)
{
    return convertFromSpecialCharacters(string(text));
}

string convertFromSpecialCharacters(string text)
{
    for (int i = 0; i < text.size(); i++)
    {
        switch(text[i])
        {
            case '\"':
            {
                text.replace(i, 1, "\\\"");
                i++;
                break;
            }
            case '\n':
            {
                text.replace(i, 1, "\\n");
                i++;
                break;
            }
            case '\t':
            {
                text.replace(i, 1, "\\t");
                i++;
                break;
            }
            case '\v':
            {
                text.replace(i, 1, "\\v");
                i++;
                break;
            }
            case '\b':
            {
                text.replace(i, 1, "\\b");
                i++;
                break;
            }
            case '\r':
            {
                text.replace(i, 1, "\\r");
                i++;
                break;
            }
            case '\f':
            {
                text.replace(i, 1, "\\f");
                i++;
                break;
            }
            case '\a':
            {
                text.replace(i, 1, "\\a");
                i++;
                break;
            }
            case '\\':
            {
                text.replace(i, 1, "\\\\");
                i++;
                break;
            }
        }
    }
    return text;
}

