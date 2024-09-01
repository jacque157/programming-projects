using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HolyWater : Item
{
    public static float healAmount = 75;

    override public void Use()
    {
        if (Player.sanity >= Player.maxSanity)
            return;
        Player.HealHorror(healAmount);
        Destroy(gameObject);
    }
}
