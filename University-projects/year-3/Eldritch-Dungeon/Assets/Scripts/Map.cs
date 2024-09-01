using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class Map
{
    public static List<List<string>> map = new List<List<string>>();
    public static bool loaded = false;
    public static float scale = 2f;
    public static BoxCollider exitTrigger;

    public static string GetCell(int row, int col)
    {
        if (row < 0 || col < 0)
            return "";
        if (row < map.Count && col < map[row].Count)
            return map[row][col];
        return "";
    }

    public static (float x, float z)CellCoordinates(int row, int column)
    {
        if (row < 0 || column < 0)
            return (0, 0);
        if (row > map.Count || column > map[row].Count)
            return (0, 0);

        float z = (float)row * scale;
        float x = (float)column * scale;
        return (x, z);
    }
}
