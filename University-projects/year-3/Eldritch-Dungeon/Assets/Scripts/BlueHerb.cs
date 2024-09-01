using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BlueHerb : Item
{
    public static float healAmount = 25;

    override public void Use()
    {
        if (Player.sanity >= Player.maxSanity)
            return;
        Player.HealHorror(healAmount);
        Destroy(gameObject);
    }
}
